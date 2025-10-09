#!/usr/bin/env python3
"""
Script to extract video and audio segments based on time periods from a CSV file
and combine them into a final output video using OpenCV.

Modified to also accept JSON input with bounding box data for speaker-focused extraction.
"""
from datetime import datetime, timezone

import pandas as pd
import cv2
import numpy as np
from pydub import AudioSegment
from pydub.utils import make_chunks
import os
import subprocess
import tempfile
from pathlib import Path
import json
import argparse
import time
from tqdm import tqdm
import bisect
import math
import shlex

def probe_fps(path):
    out = subprocess.check_output([
        "ffprobe","-v","error","-select_streams","v:0",
        "-show_entries","stream=avg_frame_rate","-of","default=nw=1:nk=1", path
    ]).decode().strip()  # e.g. "30000/1001"
    num, den = map(int, out.split("/"))
    return num/den  # float and rational string


def extract_audio_segments(word_segment_times, audio_file, output_audio_file, fade_samples:int=3000):
    """
    Faster version of extract_audio_segments().
    Avoids repeated array reallocations and redundant conversions.
    """

    # Load and extract metadata
    audio = AudioSegment.from_file(audio_file)
    sr = audio.frame_rate
    ch = audio.channels
    sw = audio.sample_width  # bytes per sample (1,2,3,4)

    dtype = {1: np.int8, 2: np.int16, 3: np.int32, 4: np.int32}[min(sw, 4)]
    full_scale = float((1 << (8 * min(sw, 3) - 1)) - 1)

    # Zero-copy decode
    raw = np.frombuffer(audio.raw_data, dtype=dtype)
    frames = raw.reshape((-1, ch)).astype(np.float32) / full_scale

    # Precompute crossfade windows
    t = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)[:, None]
    fade_out = np.cos(t * np.pi / 2.0)
    fade_in = np.sin(t * np.pi / 2.0)

    segments = []

    for i, row in enumerate(tqdm(word_segment_times, desc="Extracting audio")):
        start = int(row["start"] * sr)
        end = int(row["end"] * sr)
        if start >= end:
            raise Exception(f"Start >= End {row} {start} {end}")
        seg = frames[start:end]

        prev = []
        if segments:
            prev = segments[-1]

        if len(prev)>fade_samples:
            overlap_in = frames[start-fade_samples:start] * fade_in
            prev[-fade_samples:] = prev[-fade_samples:] * fade_out + overlap_in

        segments += [seg]

    # Combine all at once
    out = np.concatenate(segments, axis=0)

    # Convert to int16
    out_i16 = np.clip(out * 32767.0, -32768, 32767).astype(np.int16)

    pcm = out_i16.reshape(-1) # interleaved 
    out_seg = AudioSegment( data=pcm.tobytes(), frame_rate=sr, sample_width=2, # 16-bit 
                           channels=ch, ) 
    out_seg.export(output_audio_file, format="wav") 
    print(f"✅ Audio segments extracted to: {output_audio_file}") 
    return output_audio_file


def add_text_with_shadow(frame, text, font_scale=4.0):
    """
    Add text with drop shadow to a frame, centered at the bottom like closed captions.
    
    Args:
        frame: The video frame to add text to
        text: The text to display
        font_scale: Font size scale
        thickness: Text thickness
    
    Returns:
        Frame with text overlay
    """
    # Get frame dimensions
    height, width = frame.shape[:2]
    # Scale text width if needed
    thickness = int(font_scale * 2)
    
    # Use a readable font
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Calculate text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Position text at bottom center (like closed captions)
    x = (width - text_width) // 2
    y = height - 150  # Position from bottom
    
    # Add drop shadow (darker text slightly offset)
    shadow_offset = 3
    shadow_color = (0, 0, 0)  # Black shadow
    text_color = (255, 255, 255)  # White text
    
    # Draw shadow
    cv2.putText(frame, text, (x + shadow_offset, y + shadow_offset), 
                font, font_scale, shadow_color, thickness, cv2.LINE_AA)
    
    # Draw main text
    cv2.putText(frame, text, (x, y), 
                font, font_scale, text_color, thickness, cv2.LINE_AA)
    
    return frame

def extract_cropped_segment(frame, speaker_box, output_width, output_height, last_center=None, y_offset = 100):
    """
    Extract a cropped segment from the frame centered on the speaker's bounding box.
    Implements camera movement: allow big movements (>20% diagonal) and small movements (<1% diagonal),
    but restrict medium movements to small movements.
    
    Args:
        frame: The video frame
        speaker_box: Bounding box [x1, y1, x2, y2] of the speaker
        output_width: Width of the output window
        output_height: Height of the output window
        last_center: Previous center position [x, y] for smoothing (optional)
    
    Returns:
        Cropped frame centered on speaker
    """
    # Calculate center of bounding box
    target_center_x = (speaker_box[0] + speaker_box[2]) / 2
    target_center_y = (speaker_box[1] + speaker_box[3]) / 2 + y_offset
    
    # Calculate screen diagonal for movement thresholds
    screen_diagonal = np.sqrt(output_width**2 + output_height**2)
    small_movement_threshold = 0.01 * screen_diagonal  # 1% of diagonal
    big_movement_threshold = 0.20 * screen_diagonal    # 20% of diagonal
    
    # If we have a previous center, apply movement constraints
    if last_center is not None:
        last_center_x, last_center_y = last_center
        
        # Calculate distance to target
        distance = np.sqrt((target_center_x - last_center_x)**2 + (target_center_y - last_center_y)**2)
        
        # Apply movement constraints
        if distance > 0:
            if distance <= small_movement_threshold:
                # Small movement: allow it (move directly to target)
                center_x, center_y = target_center_x, target_center_y
            elif distance >= big_movement_threshold:
                # Big movement: allow it (move directly to target)
                center_x, center_y = target_center_x, target_center_y
            else:
                # Medium movement: restrict to small movement
                direction_x = (target_center_x - last_center_x) / distance
                direction_y = (target_center_y - last_center_y) / distance
                center_x = last_center_x + direction_x * small_movement_threshold
                center_y = last_center_y + direction_y * small_movement_threshold
        else:
            # No movement needed
            center_x, center_y = last_center_x, last_center_y
    else:
        # No previous center, use target directly
        center_x, center_y = target_center_x, target_center_y
    
    # Calculate crop boundaries
    crop_x1 = int(center_x - output_width / 2)
    crop_y1 = int(center_y - output_height / 2)
    crop_x2 = int(center_x + output_width / 2)
    crop_y2 = int(center_y + output_height / 2)
    
    # Get frame dimensions
    frame_height, frame_width = frame.shape[:2]
    
    # Adjust crop boundaries if they go outside frame boundaries
    if crop_x1 < 0:
        crop_x2 -= crop_x1
        crop_x1 = 0
    if crop_y1 < 0:
        crop_y2 -= crop_y1
        crop_y1 = 0
    if crop_x2 > frame_width:
        crop_x1 -= (crop_x2 - frame_width)
        crop_x2 = frame_width
    if crop_y2 > frame_height:
        crop_y1 -= (crop_y2 - frame_height)
        crop_y2 = frame_height
    
    # Ensure boundaries are within frame
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(frame_width, crop_x2)
    crop_y2 = min(frame_height, crop_y2)
    
    # Extract cropped region
    cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    
    return cropped_frame, (center_x, center_y)

def visualize_extraction_process(frame, current_time, word, speaker_info=None, original_frame=None, show_original=False):
    """
    Display the extraction process in real-time with visualization overlays.
    
    Args:
        frame: The current frame being processed
        current_time: Current time in seconds
        word: Current word being processed
        speaker_info: Optional speaker information for visualization
        original_frame: Optional original frame for side-by-side comparison
        show_original: Whether to show original frame alongside processed frame
    """
    display_frame = frame.copy()
    
    # Add time and word information
    info_text = f"Time: {current_time:.2f}s | Word: '{word}'"
    
    # Add speaker info if available
    if speaker_info:
        info_text += f" | Speaker: {speaker_info['id']}"
    
    # Add info text to top of frame
    cv2.putText(display_frame, info_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(display_frame, info_text, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    
    # Add processing status
    status_text = "EXTRACTING"
    cv2.putText(display_frame, status_text, (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    
    if show_original and original_frame is not None:
        # Create side-by-side comparison
        h1, w1 = display_frame.shape[:2]
        h2, w2 = original_frame.shape[:2]
        
        # Resize frames to same height
        if h1 != h2:
            scale = h1 / h2
            original_frame = cv2.resize(original_frame, (int(w2 * scale), h1))
            h2, w2 = original_frame.shape[:2]
        
        # Create combined frame
        combined_width = w1 + w2
        combined_frame = np.zeros((h1, combined_width, 3), dtype=np.uint8)
        combined_frame[:, :w1] = display_frame
        combined_frame[:, w1:w1+w2] = original_frame
        
        # Add labels
        cv2.putText(combined_frame, "PROCESSED", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(combined_frame, "ORIGINAL", (w1 + 10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        
        display_frame = combined_frame
    
    # Display the frame
    cv2.imshow('Extraction Visualizer', display_frame)
    
    # Wait for key press (1ms delay) to allow window to update
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        return False  # Signal to stop
    elif key == ord('p'):
        # Pause until any key is pressed
        cv2.waitKey(0)
    
    return True  # Continue processing

def preprocess_speaker_data(speaking_times, window_size=3.0):
    """
    Preprocess speaker data with a centered mode filter to smooth speaker transitions.
    Combines preprocessing and speaker computation into a single function.
    Optimized with NumPy for better performance.
    
    Args:
        speaking_times: List of JSON entries with time and people data
        window_size: Size of the moving window in seconds
    
    Returns:
        List of JSON entries with smoothed speaker data and times
    """
    
    # Sort by time
    speaking_times_averaged = sorted(speaking_times, key=lambda x: x['time'])
    
    # Extract times for faster window calculations
    times = np.array([entry['time'] for entry in speaking_times_averaged])
    
    # Precompute all speaker data for faster processing
    all_speaker_data = []
    for entry in speaking_times_averaged:
        speakers = {}
        for person in entry['people']:
            if person.get('speaking', False):
                speakers[person['id']] = 1
        all_speaker_data.append(speakers)
    
    # Create a new list for processed data
    processed_data = []
    
    for i, entry in enumerate(speaking_times_averaged):
        current_time = entry['time']
        
        # Find entries within the window using NumPy vectorization
        window_start = current_time - window_size / 2
        window_end = current_time + window_size / 2
        
        # Use NumPy for fast window selection
        in_window = (times >= window_start) & (times <= window_end)
        window_indices = np.where(in_window)[0]
        
        # Collect speaker counts within the window
        speaker_counts = {}
        for j in window_indices:
            for speaker_id in all_speaker_data[j]:
                speaker_counts[speaker_id] = speaker_counts.get(speaker_id, 0) + 1
        
        # Create a copy of the current entry
        processed_entry = entry.copy()
        processed_entry['people'] = [person.copy() for person in entry['people']]
        
        # Determine the dominant speaker in the window
        if speaker_counts:
            dominant_speaker = max(speaker_counts, key=speaker_counts.get)
            
            # Update speaking status based on mode filter
            for person in processed_entry['people']:
                person['speaking'] = (person['id'] == dominant_speaker)
        
        processed_data.append(processed_entry)
    
    return processed_data, times.tolist()

import numpy as np

def fix_zero_length_words(word_segment_times, sr, preappend_budget=0.2):
    """
    word_segment_times: list[dict], each dict at least has {"start": float, "end": float, ...}
    Mutates a copy of the list and returns it.
    """
    totaltime = 0
    preappend_budget = int(preappend_budget*sr)/sr
    if not word_segment_times:
        return []

    out = [dict(word_segment_times[0])]  # copy first row as-is
    for i in range(1, len(word_segment_times)):
        prev = out[-1]                               # previous (already possibly adjusted)
        curr = dict(word_segment_times[i])           # copy current

        curr_len = curr["end"] - curr["start"]
        if curr_len <= 0:
            # Compute allowable pull-back
            prev_dur = max(0.0, prev["end"] - prev["start"])
            halftime = 0.5 * prev_dur
            budget = min(preappend_budget, halftime)

            # Limit 1: at most `budget` earlier than nominal start
            candidate = curr["start"] - budget

            # Limit 2: do not advance earlier than the previous word's midpoint
            prev_midpoint_limit = prev["end"] - halftime  # == prev["start"] + halftime

            new_start = max(candidate, prev_midpoint_limit)

            # Ensure we don't go before the previous *start* (can happen if prev_dur == 0)
            new_start = max(new_start, prev["start"])

            # Adjust previous end only if we overlapped it
            if new_start < prev["end"]:
                prev["end"] = new_start

            # Commit the new start; clamp end if needed (still allow zero-length after move)
            curr["start"] = new_start
            if curr["end"] < curr["start"]:
                curr["end"] = curr["start"]
        totaltime += curr["end"] - curr["start"]
        out.append(curr)
    print(totaltime)
    return out

def precompute_segments(word_segment_times, fps):
    """
    Convert word segments to frame-aligned timing with sub-frame delta correction.
    """
    precomputed_segments = []
    frames_needed = 0.0
    last_end = 0
    for seg in tqdm(word_segment_times, desc="Precomputing segment timings"):
        start_f = seg["start"] * fps
        end_f = seg["end"] * fps
        start = math.floor(start_f)
        end = math.floor(end_f)
        frames_needed += (end_f - start_f) - (end - start)

        # if we need to add a frame, add it to the beginning
        if frames_needed >= 0.5 and start > last_end+1:
            start -= 1; frames_needed -= 1

        # if we need to give up a frame, remove it from the beginning
        elif frames_needed <= -0.5:
            start += 1; frames_needed += 1
        last_end = end
        precomputed_segments.append({
            "word": seg["word"],
            "start_frame": start,
            "end_frame": end,
            "start_sec": start / fps,
            "end_sec": end / fps
        })
    return precomputed_segments

def switch_speaker(current_speaker, current_time, entry, last_speaker_change_time,
                   last_speaker_start_time, last_visit_times,
                   min_stay_duration=5.0, cycle_after=8.0, last_speaker_box=None):
    """
    Compute the correct current speaker given time, active speaker entry, and switching rules.

    Returns updated:
      current_speaker, last_speaker_change_time, last_speaker_start_time,
      last_speaker_box, last_visit_times
    """
    # Determine active (currently speaking) person
    active_speaker = next((p for p in entry.get("people", []) if p.get("speaking", False)), None)

    should_switch = False
    if active_speaker:
        if current_speaker is None:
            should_switch = True
        elif active_speaker["id"] != current_speaker["id"]:
            if current_time - last_speaker_change_time >= min_stay_duration:
                should_switch = True

    if should_switch:
        if current_speaker:
            last_visit_times[current_speaker["id"]] = current_time
        current_speaker = active_speaker
        last_speaker_change_time = current_time
        last_speaker_start_time = current_time
        if current_speaker:
            last_speaker_box = current_speaker.get("box", last_speaker_box)
            last_visit_times[current_speaker["id"]] = current_time

    # Auto-cycle if same person too long
    if current_speaker and (current_time - last_speaker_start_time) >= cycle_after:
        available = [p for p in entry["people"] if p["id"] != current_speaker["id"]]
        if available:
            least_recent = min(available, key=lambda p: last_visit_times.get(p["id"], 0))
            last_visit_times[current_speaker["id"]] = current_time
            current_speaker = least_recent
            last_speaker_box = current_speaker.get("box", last_speaker_box)
            last_speaker_start_time = current_time
            last_speaker_change_time = current_time
            last_visit_times[current_speaker["id"]] = current_time

    return (
        current_speaker,
        last_speaker_change_time,
        last_speaker_start_time,
        last_speaker_box,
        last_visit_times
    )

def generate_frame_list(speaking_times_averaged, json_times, fps,
                                      total_frames, precomputed_segments,
                                      input_width=1920, input_height=1080,
                                      output_width=480, output_height=853,
                                      min_stay_duration=5.0, cycle_after=8.0):
    """
    Generate a list of frames with crop boundary information for speaker-focused extraction.
    Includes crop boundaries calculated based on speaker bounding boxes and movement smoothing.
    Includes word information from precomputed segments.
    Only includes frames that are also in a segment (have word information).
    """
    print("Generating frame list with crop boundaries and word information...")
    
    # Create a mapping from frame index to word and track frames in segments
    frame_to_word = {}
    frames_in_segments = set()
    for seg in precomputed_segments:
        start_frame = seg["start_frame"]
        end_frame = seg["end_frame"]
        word = seg["word"]
        for frame_idx in range(start_frame, end_frame):
            if frame_idx < total_frames:
                frame_to_word[frame_idx] = word
                frames_in_segments.add(frame_idx)

    # Create a list that only includes frames that are in segments
    frame_list = []
    
    current_speaker = None
    last_speaker_change_time = 0.0
    last_speaker_start_time = 0.0
    last_speaker_box = None
    last_visit_times = {}
    last_center = None

    # Only process frames that are in segments
    frames_to_process = sorted(frames_in_segments)
    
    for frame_idx in tqdm(frames_to_process, desc="Generating frame list"):
        current_time = frame_idx / fps

        # Find closest entry in JSON
        idx = bisect.bisect_left(json_times, current_time)
        if idx == 0:
            entry = speaking_times_averaged[0]
        elif idx == len(json_times):
            entry = speaking_times_averaged[-1]
        else:
            before, after = speaking_times_averaged[idx - 1], speaking_times_averaged[idx]
            entry = before if abs(before["time"] - current_time) <= abs(after["time"] - current_time) else after

        # Use reusable switch_speaker() logic
        current_speaker, last_speaker_change_time, last_speaker_start_time, \
        last_speaker_box, last_visit_times = switch_speaker(
            current_speaker,
            current_time,
            entry,
            last_speaker_change_time,
            last_speaker_start_time,
            last_visit_times,
            min_stay_duration,
            cycle_after,
            last_speaker_box
        )

        # Get word for this frame
        current_word = frame_to_word.get(frame_idx, "")

        # Calculate crop boundaries for this frame
        crop_boundaries = None
        if current_speaker and last_speaker_box:
            # Calculate center of bounding box
            target_center_x = (last_speaker_box[0] + last_speaker_box[2]) / 2
            target_center_y = (last_speaker_box[1] + last_speaker_box[3]) / 2 + 100  # y_offset
            
            # Calculate screen diagonal for movement thresholds
            screen_diagonal = np.sqrt(output_width**2 + output_height**2)
            small_movement_threshold = 0.01 * screen_diagonal  # 1% of diagonal
            big_movement_threshold = 0.20 * screen_diagonal    # 20% of diagonal
            
            # If we have a previous center, apply movement constraints
            if last_center is not None:
                last_center_x, last_center_y = last_center
                
                # Calculate distance to target
                distance = np.sqrt((target_center_x - last_center_x)**2 + (target_center_y - last_center_y)**2)
                
                # Apply movement constraints
                if distance > 0:
                    if distance <= small_movement_threshold:
                        # Small movement: allow it (move directly to target)
                        center_x, center_y = target_center_x, target_center_y
                    elif distance >= big_movement_threshold:
                        # Big movement: allow it (move directly to target)
                        center_x, center_y = target_center_x, target_center_y
                    else:
                        # Medium movement: restrict to small movement
                        direction_x = (target_center_x - last_center_x) / distance
                        direction_y = (target_center_y - last_center_y) / distance
                        center_x = last_center_x + direction_x * small_movement_threshold
                        center_y = last_center_y + direction_y * small_movement_threshold
                else:
                    # No movement needed
                    center_x, center_y = last_center_x, last_center_y
            else:
                # No previous center, use target directly
                center_x, center_y = target_center_x, target_center_y
            
            # Calculate crop boundaries
            crop_x1 = int(center_x - output_width / 2)
            crop_y1 = int(center_y - output_height / 2)
            crop_x2 = int(center_x + output_width / 2)
            crop_y2 = int(center_y + output_height / 2)
            
            # Ensure boundaries are valid and "doable" - stay within input frame dimensions
            # Make sure crop region has positive dimensions
            if crop_x2 <= crop_x1:
                crop_x2 = crop_x1 + output_width
            if crop_y2 <= crop_y1:
                crop_y2 = crop_y1 + output_height
            
            # Ensure crop region doesn't extend beyond input video dimensions
            crop_x1 = max(0, min(crop_x1, input_width - output_width))
            crop_y1 = max(0, min(crop_y1, input_height - output_height))
            crop_x2 = min(input_width, max(crop_x2, crop_x1 + output_width))
            crop_y2 = min(input_height, max(crop_y2, crop_y1 + output_height))
            
            # Final validation to ensure positive dimensions
            if crop_x2 <= crop_x1:
                crop_x2 = crop_x1 + output_width
            if crop_y2 <= crop_y1:
                crop_y2 = crop_y1 + output_height
            
            crop_boundaries = {
                "crop_x1": crop_x1,
                "crop_y1": crop_y1,
                "crop_x2": crop_x2,
                "crop_y2": crop_y2,
                "center_x": center_x,
                "center_y": center_y
            }
            
            last_center = (center_x, center_y)

        # Record the frame information for ALL frames in segments
        # If no current speaker, use last crop settings or center of frame
        if not current_speaker and last_center is not None:
            # Use last crop settings if available
            crop_boundaries = {
                "crop_x1": int(last_center[0] - output_width / 2),
                "crop_y1": int(last_center[1] - output_height / 2),
                "crop_x2": int(last_center[0] + output_width / 2),
                "crop_y2": int(last_center[1] + output_height / 2),
                "center_x": last_center[0],
                "center_y": last_center[1]
            }
        elif not current_speaker and last_center is None:
            # Start in center of frame if no previous settings
            # Assume frame dimensions (will be adjusted during extraction)
            crop_boundaries = {
                "crop_x1": 0,
                "crop_y1": 0,
                "crop_x2": output_width,
                "crop_y2": output_height,
                "center_x": output_width / 2,
                "center_y": output_height / 2
            }
        
        frame_list.append({
            "frame_idx": frame_idx,
            "time": current_time,
            "speaker_id": current_speaker["id"] if current_speaker else None,
            "speaker_box": last_speaker_box,
            "crop_boundaries": crop_boundaries,
            "word": current_word
        })

    # Count frames with valid information
    frames_with_info = len(frame_list)
    print(f"✅ Frame list generation complete.")
    print(f"   Total frames in video: {total_frames}")
    print(f"   Frames in segments: {len(frames_in_segments)}")
    print(f"   Frames included in frame list: {frames_with_info}")
    print(f"   Frames excluded (not in segments): {total_frames - len(frames_in_segments)}")
    return frame_list


def extract_video_segments(
    frame_list,
    video_file,
    silent_video_file,
    output_width,
    output_height,
    fps,
    font_scale=1.8,
    visualize=False
):
    """
    Efficiently extract and crop speaker-focused segments from a video.
    Reads frames sequentially without random seeks for maximum efficiency.
    """
    print("🧠 Optimized sequential OpenCV frame processing (no cap.set calls)...")

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {video_file}")
        return

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"📹 Input video: {input_width}x{input_height}, {input_fps:.2f} fps, {total_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(silent_video_file, fourcc, fps, (output_width, output_height))
    if not out.isOpened():
        print(f"❌ Error: Could not create output video {silent_video_file}")
        cap.release()
        return

    if visualize:
        cv2.namedWindow('Extraction Visualizer', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Extraction Visualizer', 1200, 800)
        print("Visualizer started. Press 'q' to quit, 'p' to pause.")

    processed_frames = 0
    skipped_frames = 0
    continue_processing = True

    # Ensure frame list is sorted
    frame_list = sorted(frame_list, key=lambda f: f["frame_idx"])

    target_iter = iter(frame_list)
    current_target = next(target_iter, None)
    current_frame_idx = 0
    
    # Store last 5 frames for fade transitions
    last_frames = []
    max_frames_to_store = 5
    fade_frames = 5  # Number of frames to fade over

    with tqdm(total=len(frame_list), desc="Processing frames") as pbar:
        while cap.isOpened() and current_target is not None and continue_processing:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames until reaching the next target frame index
            if current_frame_idx < current_target["frame_idx"]:
                current_frame_idx += 1
                continue

            # Process this frame
            frame_info = current_target
            processed_frame = frame
            crop_bounds = frame_info.get("crop_boundaries")
            if crop_bounds:
                x1, y1, x2, y2 = (
                    crop_bounds["crop_x1"],
                    crop_bounds["crop_y1"],
                    crop_bounds["crop_x2"],
                    crop_bounds["crop_y2"],
                )
                if x2 > x1 and y2 > y1:
                    processed_frame = frame[y1:y2, x1:x2]

            if processed_frame.shape[1] != output_width or processed_frame.shape[0] != output_height:
                processed_frame = cv2.resize(processed_frame, (output_width, output_height))

            word = frame_info.get("word", "")
            processed_frame = add_text_with_shadow(processed_frame, word, font_scale=font_scale)
            
            # Write the current frame
            out.write(processed_frame)
            processed_frames += 1
            pbar.update(1)
            
            # Store current frame for future transitions
            last_frames.append(processed_frame.copy())
            if len(last_frames) > max_frames_to_store:
                last_frames.pop(0)

            # Visualization
            if visualize:
                original = frame.copy()
                current_time = frame_info["frame_idx"] / input_fps
                speaker_info = {
                    "id": frame_info.get("speaker_id", "unknown"),
                    "box": frame_info.get("speaker_box")
                }
                continue_processing = visualize_extraction_process(
                    processed_frame,
                    current_time,
                    word,
                    speaker_info,
                    original,
                    show_original=True
                )
                if not continue_processing:
                    print("Visualization stopped by user.")
                    break

            # Move to next target frame
            current_target = next(target_iter, None)
            current_frame_idx += 1

    cap.release()
    out.release()
    if visualize:
        cv2.destroyAllWindows()

    print(f"✅ Processing complete. Processed {processed_frames}, skipped {skipped_frames}.")
    verify = cv2.VideoCapture(silent_video_file)
    print(f"   Output frames: {int(verify.get(cv2.CAP_PROP_FRAME_COUNT))}")
    verify.release()

def combine_video_audio(video_file, audio_file, final_output):
    """
    Combine video and audio files using ffmpeg.
    
    Args:
        video_file (str): Path to video file (without audio)
        audio_file (str): Path to audio file
        output_file (str): Path for final output file
    """
    
    # Use ffmpeg to combine video and audio
    cmd = [
        'ffmpeg',
        '-i', video_file,
        '-i', audio_file,
        '-c:v', 'libx264',
        '-c:a', 'aac',
        '-strict', 'experimental',
        '-y',  # Overwrite output file if it exists
        final_output
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Video and audio combined to: {final_output}")
    except subprocess.CalledProcessError as e:
        print(f"Error combining video and audio: {e}")
        return False
    
    return True

def main():
    """Main function to extract and combine video/audio segments."""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract video segments based on CSV with optional JSON cropping')
    parser.add_argument('--speaking_times', type=str, help='JSON file with bounding box data for speaker cropping')
    parser.add_argument('--csv', type=str, default='words.csv', help='CSV file with time segments (default: words.csv)')
    parser.add_argument('--video', type=str, default='deblurred.mp4', help='Input video file (default: deblurred.mp4)')
    parser.add_argument('--audio', type=str, default='GBRvid4_audio.wav', help='Input audio file (default: GBRvid4_audio.wav)')
    parser.add_argument('--width', type=int, default=640, help='Output window width (default: 640)')
    parser.add_argument('--height', type=int, default=480, help='Output window height (default: 480)')
    parser.add_argument('--cycle-after', type=int, default=8, help='How long to cycle off of an active speaker')
    parser.add_argument('--visualize', action='store_true', help='Show real-time visualization of extraction process')
    parser.add_argument('--font-scale', type=float, default=4.0, help='Font scale for text overlay (default: 4.0)')
    parser.add_argument('--min-stay-duration', type=float, default=5, help='Minimum time in seconds to stay on a person after switching target (default: 1.5)')
    parser.add_argument('--no-video', action='store_true', help='Skip video processing')
    parser.add_argument('--no-audio', action='store_true', help='Skip audio processing')
    parser.add_argument('--no-combine', action='store_true', help='Skip combine')
    parser.add_argument('--save-lengths', action='store_true', help='Save segment lengths summary file')
    parser.add_argument('--fade-samples', type=int, default=3000,
                        help='Crossfade length in samples for audio stitching (default: 256)')

    
    args = parser.parse_args()
    
    # File paths
    csv_file = args.csv
    video_file = args.video
    audio_file = args.audio
    final_output = video_file + "_final.mp4"
    output_audio_file = audio_file+"_extracted.wav"
    
    # Check if files exist
    if not os.path.exists(csv_file):
        print(f"Error: CSV file '{csv_file}' not found")
        return
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found")
        return
    
    print("Starting video/audio extraction...")
    
    audio = AudioSegment.from_file(audio_file)
    sr = audio.frame_rate
    
    # --- Load CSV + JSON ---
    word_segment_times_df = pd.read_csv(csv_file)
    # align all times to audio sample boundaries
    word_segment_times_df["start"] = (word_segment_times_df["start"] * sr).astype(int) / sr
    word_segment_times_df["end"] = (word_segment_times_df["end"] * sr).astype(int) / sr
    # Convert DataFrame to list of dictionaries for easier processing
    word_segment_times = word_segment_times_df.to_dict('records')
    word_segment_times = fix_zero_length_words(word_segment_times, sr)

    with open(args.speaking_times, 'r') as f:
        speaking_times = json.load(f)

    if not args.no_audio:
        print("\n=== Extracting Audio Segments ===")
        extract_audio_segments(word_segment_times, audio_file, output_audio_file, fade_samples=args.fade_samples)

    
    # Check video file only if not in audio-only mode
    if not os.path.exists(video_file):
        print(f"Error: Video file '{video_file}' not found")
        return
    
    silent_video_file = video_file + "_cropped.mp4"

    if not args.no_video:
        total_frames = int(cv2.VideoCapture(video_file).get(cv2.CAP_PROP_FRAME_COUNT))
        fps = probe_fps(video_file)
        precomputed_segments = precompute_segments(word_segment_times, fps)
        speaking_times_averaged, json_times = preprocess_speaker_data(speaking_times)
        # Get input video dimensions
        cap = cv2.VideoCapture(video_file)
        input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        frame_number_and_bounds = generate_frame_list(
            speaking_times_averaged,
            json_times,
            fps,
            total_frames,
            precomputed_segments,
            input_width=input_width,
            input_height=input_height,
            cycle_after=args.cycle_after,
            output_width=args.width,
            output_height=args.height
        )
        # Convert to serializable format
        serializable_list = []
        for frame_info in frame_number_and_bounds:
            if frame_info is not None:
                serializable_list.append(frame_info)
        # Write frame list to file for inspection
        with open("frame_number_and_bounds.json", "w") as f:
            json.dump(serializable_list, f, indent=2)
        print("✅ Frame list written to frame_number_and_bounds.json")
        
        extract_video_segments(
            frame_number_and_bounds,
            video_file,
            silent_video_file,
            output_width=args.width,
            output_height=args.height,
            fps=fps,
            visualize=args.visualize
        )

    if not args.no_combine:
        print("\n=== Combining Video and Audio ===")
        combine_video_audio(silent_video_file, output_audio_file, final_output)

if __name__ == "__main__":
    main()
