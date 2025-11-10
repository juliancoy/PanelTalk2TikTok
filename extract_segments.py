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
from keyframes import load_and_interpolate_camera_keyframes, generate_frame_list_from_camera_keyframes


def probe_fps(path):
    out = (
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate",
                "-of",
                "default=nw=1:nk=1",
                path,
            ]
        )
        .decode()
        .strip()
    )  # e.g. "30000/1001"
    num, den = map(int, out.split("/"))
    return num / den  # float and rational string


def extract_audio_segments(
    word_segment_times, audio_file, output_audio_file, fade_samples: int = 3000
):
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

    # Precompute crossfade windows (expanded to match channel count)
    # t has shape (fade_samples, 1). We compute 1-D fades then expand to (fade_samples, ch)
    t = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)[:, None]
    fade_out_1d = np.cos(t * np.pi / 2.0)
    fade_in_1d = np.sin(t * np.pi / 2.0)
    # Expand fades to match number of channels so operations broadcast correctly
    # `ch` is number of audio channels (e.g., 2 for stereo)
    fade_out = np.repeat(fade_out_1d, ch, axis=1)
    fade_in = np.repeat(fade_in_1d, ch, axis=1)

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

        if len(prev) > fade_samples:
            overlap_in = frames[start - fade_samples : start] * fade_in
            prev[-fade_samples:] = prev[-fade_samples:] * fade_out + overlap_in

        segments.append(seg.copy())


    # Combine all at once
    out = np.concatenate(segments, axis=0)

    # Convert to int16
    out_i16 = np.clip(out * 32767.0, -32768, 32767).astype(np.int16)

    pcm = out_i16.reshape(-1)  # interleaved
    out_seg = AudioSegment(
        data=pcm.tobytes(),
        frame_rate=sr,
        sample_width=2,  # 16-bit
        channels=ch,
    )
    out_seg.export(output_audio_file, format="wav")
    print(f"✅ Audio segments extracted to: {output_audio_file}")
    return output_audio_file


def add_text_with_shadow(frame, text, font_scale=4.0, text_height_from_bottom=300):
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
    # Ensure text is a string and handle None / NaN values
    if text is None:
        text = ""
    else:
        try:
            # Treat NaN or Inf floats as empty text
            if isinstance(text, float) and (np.isnan(text) or np.isinf(text)):
                text = ""
            else:
                text = str(text)
        except Exception:
            text = ""
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )

    # Position text at bottom center (like closed captions)
    x = (width - text_width) // 2
    y = height - text_height_from_bottom  # Position from bottom

    # Add drop shadow (darker text slightly offset)
    shadow_offset = 3
    shadow_color = (0, 0, 0)  # Black shadow
    text_color = (255, 255, 255)  # White text

    # Draw shadow
    cv2.putText(
        frame,
        text,
        (x + shadow_offset, y + shadow_offset),
        font,
        font_scale,
        shadow_color,
        thickness,
        cv2.LINE_AA,
    )

    # Draw main text
    cv2.putText(
        frame, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA
    )

    return frame

def preprocess_injections(injections_list, output_width, output_height):
    """
    Preprocess injection images by loading them from disk and precomputing
    their scaled versions for the target output dimensions.
    This avoids repeated disk I/O and image processing during frame processing.
    """
    if not injections_list:
        return []
    
    processed_injections = []
    for inj in injections_list:
        filename = inj["filename"]
        if not os.path.exists(filename):
            print(f"⚠️ Warning: Injection image '{filename}' not found, skipping")
            continue
            
        # Load image once
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"⚠️ Warning: Could not load injection image '{filename}', skipping")
            continue
            
        # Ensure BGRA format
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            bgr = img
            alpha = np.full((img.shape[0], img.shape[1], 1), 255, dtype=np.uint8)
            img = np.concatenate([bgr, alpha], axis=2)
        
        # Store the preloaded image and other metadata
        processed_inj = inj.copy()
        processed_inj["image"] = img
        processed_injections.append(processed_inj)
    
    return processed_injections


def overlay_injections_on_frame(frame, current_time, injections, fps=30.0):
    """
    Overlay any active injection images onto `frame` (BGR uint8) based on current_time.
    - injections: list of dicts produced by preprocess_injections (each has 'start','end','image')
    - fps: frames per second for fade calculations
    Returns a new BGR uint8 frame with overlays applied (alpha composited).
    """
    if not injections:
        return frame
    out = frame.astype(np.float32)
    h, w = frame.shape[:2]
    for inj in injections:
        start = inj.get("start")
        end = inj.get("end")
        if start is None or end is None:
            continue
        if current_time < start or current_time > end:
            continue
        
        # Use preloaded image instead of reading from disk
        img = inj.get("image")
        if img is None:
            continue

        # Calculate fade alpha based on fade_in_frames and fade_out_frames
        fade_in_frames = inj.get("fade_in_frames", 0)
        fade_out_frames = inj.get("fade_out_frames", 0)
        
        # Calculate fade alpha (0.0 to 1.0)
        fade_alpha = 1.0
        
        # Fade in calculation
        if fade_in_frames > 0 and current_time < start + (fade_in_frames / fps):
            fade_progress = (current_time - start) * fps / fade_in_frames
            fade_alpha = min(fade_alpha, max(0.0, fade_progress))
        
        # Fade out calculation
        if fade_out_frames > 0 and current_time > end - (fade_out_frames / fps):
            fade_progress = (end - current_time) * fps / fade_out_frames
            fade_alpha = min(fade_alpha, max(0.0, fade_progress))
        
        # Skip if completely faded out
        if fade_alpha <= 0.0:
            continue

        ih, iw = img.shape[:2]
        
        # Get positioning parameters (default to center if not specified)
        center_x = inj.get("center_x", w // 2)
        center_y = inj.get("center_y", h // 2)
        zoom = inj.get("zoom", 1.0)
        
        # Apply zoom factor to scale calculation
        base_scale = min(float(w) / float(iw), float(h) / float(ih))
        scale = base_scale * zoom
        
        new_w = max(1, int(round(iw * scale)))
        new_h = max(1, int(round(ih * scale)))
        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)

        # Create transparent canvas and position the resized image based on center_x, center_y
        canvas = np.zeros((h, w, 4), dtype=np.uint8)
        x = int(center_x - new_w // 2)
        y = int(center_y - new_h // 2)
        
        # Ensure the image stays within frame boundaries
        x = max(0, min(x, w - new_w))
        y = max(0, min(y, h - new_h))
        
        # Calculate the actual region where we can place the image
        # This handles cases where the resized image is larger than the available space
        canvas_y_start = max(0, y)
        canvas_y_end = min(h, y + new_h)
        canvas_x_start = max(0, x)
        canvas_x_end = min(w, x + new_w)
        
        # Calculate corresponding regions in the resized image
        resized_y_start = max(0, -y)
        resized_y_end = resized_y_start + (canvas_y_end - canvas_y_start)
        resized_x_start = max(0, -x)
        resized_x_end = resized_x_start + (canvas_x_end - canvas_x_start)
        
        # Ensure we don't exceed the resized image dimensions
        resized_y_end = min(new_h, resized_y_end)
        resized_x_end = min(new_w, resized_x_end)
        
        # Only place the portion of the resized image that fits in the canvas
        if (canvas_y_end > canvas_y_start and canvas_x_end > canvas_x_start and
            resized_y_end > resized_y_start and resized_x_end > resized_x_start):
            canvas[canvas_y_start:canvas_y_end, canvas_x_start:canvas_x_end, :] = \
                resized[resized_y_start:resized_y_end, resized_x_start:resized_x_end, :]

        # Apply alpha compositing with fade alpha
        bgr = canvas[..., :3].astype(np.float32)
        alpha = canvas[..., 3].astype(np.float32) / 255.0
        alpha = alpha[:, :, None] * fade_alpha  # Apply fade alpha
        out = bgr * alpha + out * (1.0 - alpha)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def extract_cropped_segment(
    frame,
    speaker_box,
    output_width,
    output_height,
    last_center=None,
    last_zoom=1.0,
    y_offset=100,
    zoom_speed=0.02,
):
    """
    Extract a cropped segment from the frame centered on the speaker's bounding box
    with optional smooth zooming toward the speaker.

    Behavior:
      - If `speaker_box` is provided, compute a desired zoom so that the speaker's
        bounding box occupies a fixed fraction of the output frame, then smoothly
        interpolate the zoom level from `last_zoom` toward `desired_zoom` using
        `zoom_speed`.
      - If `speaker_box` is None (non-speaker frame), do NOT zoom: zoom is reset
        immediately to 1.0 (no magnification).
      - Movement smoothing for the center is preserved (same small/medium/big
        thresholds as before).

    Returns:
        cropped_frame: The cropped image (may be smaller than output; caller should
                       resize to output_width/output_height)
        center: (center_x, center_y) used for the crop
        new_zoom: the updated zoom value to carry forward for the next frame
    """
    # Validate speaker_box presence (if empty or None we will not zoom)
    has_speaker = speaker_box is not None

    # Determine target center from speaker box if available
    if has_speaker:
        target_center_x = (speaker_box[0] + speaker_box[2]) / 2
        target_center_y = (speaker_box[1] + speaker_box[3]) / 2 + y_offset
    else:
        # If no speaker box provided, default target is last_center or center of frame
        frame_h, frame_w = frame.shape[:2]
        target_center_x = last_center[0] if last_center is not None else frame_w / 2
        target_center_y = last_center[1] if last_center is not None else frame_h / 2

    # Movement smoothing (reuse existing thresholds)
    screen_diagonal = np.sqrt(output_width**2 + output_height**2)
    small_movement_threshold = 0.01 * screen_diagonal  # 1% of diagonal
    big_movement_threshold = 0.20 * screen_diagonal  # 20% of diagonal

    if last_center is not None:
        last_center_x, last_center_y = last_center

        # Distance to target
        distance = np.sqrt(
            (target_center_x - last_center_x) ** 2
            + (target_center_y - last_center_y) ** 2
        )

        if distance > 0:
            if distance <= small_movement_threshold:
                center_x, center_y = target_center_x, target_center_y
            elif distance >= big_movement_threshold:
                center_x, center_y = target_center_x, target_center_y
            else:
                direction_x = (target_center_x - last_center_x) / distance
                direction_y = (target_center_y - last_center_y) / distance
                center_x = last_center_x + direction_x * small_movement_threshold
                center_y = last_center_y + direction_y * small_movement_threshold
        else:
            center_x, center_y = last_center_x, last_center_y
    else:
        center_x, center_y = target_center_x, target_center_y

    # Get frame dims
    frame_height, frame_width = frame.shape[:2]

    # Zoom computation
    if has_speaker:
        box_w = max(1.0, float(speaker_box[2] - speaker_box[0]))
        # fraction of output width we want the box to occupy (tweakable)
        target_fraction = 0.45
        # desired zoom = how much the final frame should magnify the box
        # S = output_width / crop_width and we want box_w / crop_width == target_fraction
        # => crop_width = box_w / target_fraction => S = output_width * target_fraction / box_w
        desired_zoom = (output_width * target_fraction) / box_w
        desired_zoom = float(max(1.0, min(desired_zoom, 3.0)))  # clamp reasonable zoom
        if last_zoom is None:
            last_zoom = 1.0
        # Smoothly interpolate zoom toward desired_zoom
        new_zoom = last_zoom + (desired_zoom - last_zoom) * float(zoom_speed)
    else:
        # Don't zoom on non-speakers: reset immediately
        new_zoom = 1.0

    # Compute crop dimensions based on zoom (smaller crop -> more zoom when resized to output)
    crop_w = int(round(output_width / new_zoom))
    crop_h = int(round(output_height / new_zoom))

    # Fallback minimums and maximums
    crop_w = max(2, min(crop_w, frame_width))
    crop_h = max(2, min(crop_h, frame_height))

    # Calculate crop box from center
    crop_x1 = int(center_x - crop_w / 2)
    crop_y1 = int(center_y - crop_h / 2)
    crop_x2 = crop_x1 + crop_w
    crop_y2 = crop_y1 + crop_h

    # Adjust crop boundaries if they go outside frame boundaries
    if crop_x1 < 0:
        crop_x2 -= crop_x1
        crop_x1 = 0
    if crop_y1 < 0:
        crop_y2 -= crop_y1
        crop_y1 = 0
    if crop_x2 > frame_width:
        crop_x1 -= crop_x2 - frame_width
        crop_x2 = frame_width
    if crop_y2 > frame_height:
        crop_y1 -= crop_y2 - frame_height
        crop_y2 = frame_height

    # Ensure boundaries are within frame
    crop_x1 = max(0, crop_x1)
    crop_y1 = max(0, crop_y1)
    crop_x2 = min(frame_width, crop_x2)
    crop_y2 = min(frame_height, crop_y2)

    # Extract cropped region
    cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]

    return cropped_frame, (center_x, center_y), new_zoom


def visualize_extraction_process(
    frame,
    current_time,
    word,
    speaker_info=None,
    original_frame=None,
    show_original=False,
    get_frame=False
):
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
    cv2.putText(
        display_frame,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        display_frame,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )

    # Add processing status
    status_text = "EXTRACTING"
    cv2.putText(
        display_frame,
        status_text,
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

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
        combined_frame[:, w1 : w1 + w2] = original_frame

        # Add labels
        cv2.putText(
            combined_frame,
            "PROCESSED",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            combined_frame,
            "ORIGINAL",
            (w1 + 10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        display_frame = combined_frame

    # Display the frame
    cv2.imshow("Extraction Visualizer", display_frame)

    # Wait for key press (1ms delay) to allow window to update
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        return False  # Signal to stop
    elif key == ord("p") or get_frame:
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
    speaking_times_averaged = sorted(speaking_times, key=lambda x: x["time"])

    # Extract times for faster window calculations
    times = np.array([entry["time"] for entry in speaking_times_averaged])

    # Precompute all speaker data for faster processing
    all_speaker_data = []
    for entry in speaking_times_averaged:
        speakers = {}
        for person in entry["people"]:
            if person.get("speaking", False):
                speakers[person["id"]] = 1
        all_speaker_data.append(speakers)

    # Create a new list for processed data
    processed_data = []

    for i, entry in enumerate(speaking_times_averaged):
        current_time = entry["time"]

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
        processed_entry["people"] = [person.copy() for person in entry["people"]]

        # Determine the dominant speaker in the window
        if speaker_counts:
            dominant_speaker = max(speaker_counts, key=speaker_counts.get)

            # Update speaking status based on mode filter
            for person in processed_entry["people"]:
                person["speaking"] = person["id"] == dominant_speaker

        processed_data.append(processed_entry)

    return processed_data, times.tolist()


import numpy as np


def fix_zero_length_words(
    word_segment_times, sr, preappend_budget=0.2, shift_seconds=0.15
):
    """
    word_segment_times: list[dict], each dict at least has {"start": float, "end": float, ...}
    Mutates a copy of the list and returns it.
    """
    totaltime = 0
    preappend_budget = int(preappend_budget * sr) / sr
    if not word_segment_times:
        return []

    out = [dict(word_segment_times[0])]  # copy first row as-is
    for i in range(1, len(word_segment_times)):
        prev = out[-1]  # previous (already possibly adjusted)
        curr = dict(word_segment_times[i])  # copy current
        # apply shift
        curr["start"] = curr["start"] - shift_seconds

        # lengthening is ok
        if i < len(word_segment_times) - 1:
            curr["end"] = min(
                curr["end"], word_segment_times[i + 1]["start"] - shift_seconds
            )

        if curr["start"] < 0:
            continue

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
    finalout = []
    for segment in out:
        if segment["end"] - segment["start"] > 0.01:
            finalout.append(segment)
        else:
            print(f"Segment too short {segment}")
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
        if frames_needed >= 0.5 and start > last_end + 1:
            start -= 1
            frames_needed -= 1

        # if we need to give up a frame, remove it from the beginning
        elif frames_needed <= -0.5:
            start += 1
            frames_needed += 1
        last_end = end
        precomputed_segments.append(
            {
                "word": seg["word"],
                "start_frame": start,
                "end_frame": end,
                "start_sec": start / fps,
                "end_sec": end / fps,
            }
        )
    return precomputed_segments


def switch_speaker(
    current_speaker,
    current_time,
    entry,
    last_speaker_change_time,
    last_speaker_start_time,
    last_visit_times,
    min_stay_duration=5.0,
    cycle_after=8.0,
    last_speaker_box=None,
):
    """
    Compute the correct current speaker given time, active speaker entry, and switching rules.

    Returns updated:
      current_speaker, last_speaker_change_time, last_speaker_start_time,
      last_speaker_box, last_visit_times
    """
    # Determine active (currently speaking) person
    active_speaker = next(
        (p for p in entry.get("people", []) if p.get("speaking", False)), None
    )

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
            least_recent = min(
                available, key=lambda p: last_visit_times.get(p["id"], 0)
            )
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
        last_visit_times,
    )


def generate_frame_list(
    speaking_times_averaged,
    json_times,
    fps,
    total_frames,
    precomputed_segments,
    input_width=1920,
    input_height=1080,
    output_width=480,
    output_height=853,
    min_stay_duration=5.0,
    cycle_after=8.0,
):
    """
    Generate a list of frames with crop boundary information for speaker-focused extraction.
    Includes crop boundaries calculated based on speaker bounding boxes and movement smoothing.
    Includes word information from precomputed segments.
    Only includes frames that are also in a segment (have word information).
    """
    print("Generating frame list with crop boundaries and word information...")

    # Create a list that only includes frames that are in segments
    frame_list = []

    current_speaker = None
    last_speaker_change_time = 0.0
    last_speaker_start_time = 0.0
    last_speaker_box = None
    last_visit_times = {}
    last_center = None
    frames_with_info = 0

    for seg in tqdm(precomputed_segments, desc="Generating frame list"):
        seg_list = []
        start_frame = seg["start_frame"]
        end_frame = seg["end_frame"]
        word = seg["word"]
        for frame_idx in range(start_frame, end_frame):
            current_time = frame_idx / fps

            # Find closest entry in JSON
            idx = bisect.bisect_left(json_times, current_time)
            if idx == 0:
                entry = speaking_times_averaged[0]
            elif idx == len(json_times):
                entry = speaking_times_averaged[-1]
            else:
                before, after = (
                    speaking_times_averaged[idx - 1],
                    speaking_times_averaged[idx],
                )
                entry = (
                    before
                    if abs(before["time"] - current_time)
                    <= abs(after["time"] - current_time)
                    else after
                )

            # Use reusable switch_speaker() logic
            (
                current_speaker,
                last_speaker_change_time,
                last_speaker_start_time,
                last_speaker_box,
                last_visit_times,
            ) = switch_speaker(
                current_speaker,
                current_time,
                entry,
                last_speaker_change_time,
                last_speaker_start_time,
                last_visit_times,
                min_stay_duration,
                cycle_after,
                last_speaker_box,
            )

            # Get word for this frame
            current_word = word

            # Calculate crop boundaries for this frame
            crop_boundaries = None
            if current_speaker and last_speaker_box:
                # Calculate center of bounding box
                target_center_x = (last_speaker_box[0] + last_speaker_box[2]) / 2
                target_center_y = (
                    last_speaker_box[1] + last_speaker_box[3]
                ) / 2 + 100  # y_offset

                # Calculate screen diagonal for movement thresholds
                screen_diagonal = np.sqrt(output_width**2 + output_height**2)
                small_movement_threshold = 0.01 * screen_diagonal  # 1% of diagonal
                big_movement_threshold = 0.20 * screen_diagonal  # 20% of diagonal

                # If we have a previous center, apply movement constraints
                if last_center is not None:
                    last_center_x, last_center_y = last_center

                    # Calculate distance to target
                    distance = np.sqrt(
                        (target_center_x - last_center_x) ** 2
                        + (target_center_y - last_center_y) ** 2
                    )

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
                            center_x = (
                                last_center_x + direction_x * small_movement_threshold
                            )
                            center_y = (
                                last_center_y + direction_y * small_movement_threshold
                            )
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
                    "center_y": center_y,
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
                    "center_y": last_center[1],
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
                    "center_y": output_height / 2,
                }

            seg_list.append(
                {
                    "frame_idx": frame_idx,
                    "time": current_time,
                    "speaker_id": current_speaker["id"] if current_speaker else None,
                    "speaker_box": last_speaker_box,
                    "crop_boundaries": crop_boundaries,
                    "word": current_word,
                }
            )
            frames_with_info += 1
        frame_list.append(seg_list)
    # Count frames with valid information
    print(f"✅ Frame list generation complete.")
    print(f"   Total frames in video: {total_frames}")
    print(f"   Frames included in frame list: {frames_with_info}")
    return frame_list


def extract_video_segments(
    frame_list,
    video_file,
    silent_video_file,
    output_width,
    output_height,
    fps,
    font_scale=1.8,
    visualize=False,
    fade_frames=4,
    read_shift_frames=7,
    text_height=300,
    zoom_speed=0.02,
    injections=None,
    get_frame=False,
    video_start_frame: int = 0,
):
    """
    Efficiently extract and crop speaker-focused segments from a video.
    Smoothly fades between segments using crossfade blending.
    Uses extract_cropped_segment(...) to perform smooth center movement and
    optional smooth zooming on speakers only.
    """
    print(
        "🧠 Optimized sequential OpenCV frame processing (with smooth fades & optional zoom)..."
    )

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video file {video_file}")
        return

    # Seek to requested start frame (if provided). This avoids reading from the very beginning
    try:
        start_frame_to_seek = int(video_start_frame or 0)
        if start_frame_to_seek > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_to_seek)
    except Exception:
        # ignore if backend doesn't support seeking
        start_frame_to_seek = 0

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(
        f"📹 Input video: {input_width}x{input_height}, {input_fps:.2f} fps, {total_frames} frames"
    )

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(silent_video_file, fourcc, fps, (output_width, output_height))
    if not out.isOpened():
        print(f"❌ Error: Could not create output video {silent_video_file}")
        cap.release()
        return

    if visualize:
        cv2.namedWindow("Extraction Visualizer", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Extraction Visualizer", 1200, 800)
        print("Visualizer started. Press 'q' to quit, 'p' to pause.")

    processed_frames = 0
    # Initialize current_frame_idx to the actual capture position (video_start_frame) so comparisons
    # with absolute frame indices in frame_list work correctly.
    current_frame_idx = int(video_start_frame or 0)
    frame_buffer = []
    fadeout_frames = []

    # Track last center and zoom across frames (for smooth movement + zoom)
    last_center = None

    last_zoom = 1.0
    for seg in tqdm(frame_list):

        # If the center would jump more than 75 pixels at the start of this segment,
        # reset center smoothing so we don't pan across a large distance.
        if seg and last_center is not None:
            first_frame = seg[0]
            fb_box = first_frame.get("speaker_box")
            if fb_box is not None:
                target_cx = (fb_box[0] + fb_box[2]) / 2
                target_cy = (fb_box[1] + fb_box[3]) / 2 + 100
                dist = math.hypot(
                    target_cx - last_center[0], target_cy - last_center[1]
                )
                if dist > 75:
                    last_center = None

        # Process all frames in this segment
        fadeout_len = min(fade_frames, len(seg))
        possible_fadein_frames = len(seg[:-fadeout_len])
        if get_frame:
            fadeout_len = -1
            fadeout_frames = []

        for frame_info in seg[:-fadeout_len]:
            # Skip ahead to the next frame index
            target_idx = frame_info["frame_idx"] +read_shift_frames
            
            # Use OpenCV's frame seek to jump directly to the target frame
            if abs(target_idx - current_frame_idx) > 10:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                current_frame_idx = target_idx
            
            # Read the target frame
            ret, frame = cap.read()
            if not ret:
                print(f"❌ Failed to read frame {target_idx}")
                break
            current_frame_idx += 1


            # Prefer speaker_box (for zoom) if available; otherwise fall back to provided crop_bounds
            speaker_box = frame_info.get("speaker_box")
            speaker_id = frame_info.get("speaker_id")
            if speaker_box is not None and speaker_id is not None:
                # Use extract_cropped_segment to compute a (possibly zoomed) crop and updated center/zoom
                cropped, new_center, new_zoom = extract_cropped_segment(
                    frame,
                    speaker_box,
                    output_width,
                    output_height,
                    last_center=last_center,
                    last_zoom=last_zoom,
                    y_offset=100,
                    zoom_speed=zoom_speed,
                )
                # Resize to final output size for consistent blending
                frame = cv2.resize(cropped, (output_width, output_height))
                last_center = new_center
                last_zoom = new_zoom
            else:
                # No speaker -> do not apply zoom. Use precomputed crop boundaries if present.
                crop_bounds = frame_info.get("crop_boundaries")
                if crop_bounds:
                    x1, y1, x2, y2 = (
                        crop_bounds["crop_x1"],
                        crop_bounds["crop_y1"],
                        crop_bounds["crop_x2"],
                        crop_bounds["crop_y2"],
                    )
                    if x2 > x1 and y2 > y1:
                        frame = frame[y1:y2, x1:x2]
                # Reset zoom immediately on non-speaker frames
                last_zoom = 1.0

            # Ensure frame is final output size before buffering/overlay
            if (
                frame is not None
                and frame.size != 0
                and (
                    frame.shape[0] != output_height
                    or frame.shape[1] != output_width
                )
            ):
                frame = cv2.resize(frame, (output_width, output_height))
            # Overlay any active injection images for this time BEFORE buffering (so crossfades include overlays)
                frame = overlay_injections_on_frame(
                    frame, frame_info["time"], injections, fps
                )
            frame_buffer.append(frame)
            if len(frame_buffer) > fade_frames + 1:
                frame_buffer.pop(0)

            # if there are leftover frames, blend and write
            while len(fadeout_frames) > possible_fadein_frames:
                out.write(fadeout_frames.pop(0)[0])

            while len(fadeout_frames) > fade_frames:
                out.write(fadeout_frames.pop(0)[0])

            fade_len = len(fadeout_frames)

            for i, frame_and_word in enumerate(fadeout_frames):
                fadeout_frame, word = frame_and_word
                alpha = i / float(fade_len)
                # Blend with the corresponding buffered frame
                frame = cv2.addWeighted(
                    fadeout_frame, 1 - alpha, frame_buffer[-fade_len + i], alpha, 0
                )
                # Overlay any injections for this frame/time before adding text
                frame = overlay_injections_on_frame(
                    frame, frame_info["time"], injections, fps
                )
                frame = add_text_with_shadow(
                    frame,
                    word,
                    font_scale=font_scale,
                    text_height_from_bottom=text_height,
                )
                out.write(frame)
                
                # Visualization
                if visualize:
                    original = frame.copy()
                    current_time = target_idx / input_fps
                    speaker_info = {
                        "id": frame_info.get("speaker_id", "unknown"),
                        "box": frame_info.get("speaker_box"),
                    }
                    continue_processing = visualize_extraction_process(
                        frame,
                        current_time,
                        word,
                        speaker_info,
                        original,
                        show_original=True,
                    )
                    if not continue_processing:
                        print("Visualization stopped by user.")
                        break

            fadeout_frames = []
            # Resize and annotate (ensure final output size)
            if frame is None or frame.size == 0:
                break

            frame = cv2.resize(frame, (output_width, output_height))
            # Overlay injection (if any) before drawing text
            frame = overlay_injections_on_frame(
                frame, frame_info["time"], injections, fps
            )
            word = frame_info.get("word", "")
            frame = add_text_with_shadow(
                frame, word, font_scale=font_scale, text_height_from_bottom=text_height
            )

            # Add to current segment buffer
            out.write(frame)
            processed_frames += 1

            # Visualization
            if visualize:
                original = frame.copy()
                current_time = target_idx / input_fps
                speaker_info = {
                    "id": frame_info.get("speaker_id", "unknown"),
                    "box": frame_info.get("speaker_box"),
                }
                continue_processing = visualize_extraction_process(
                    frame,
                    current_time,
                    word,
                    speaker_info,
                    original,
                    show_original=True,
                    get_frame=get_frame
                )
                if not continue_processing:
                    print("Visualization stopped by user.")
                    break

        for frame_info in seg[-fadeout_len:]:

            ret, frame = cap.read()
            current_frame_idx += 1
            if not ret:
                break

            # Same logic for trailing frames (apply zoom only for speaker)
            speaker_box = frame_info.get("speaker_box")
            speaker_id = frame_info.get("speaker_id")
            if speaker_box is not None and speaker_id is not None:
                cropped, new_center, new_zoom = extract_cropped_segment(
                    frame,
                    speaker_box,
                    output_width,
                    output_height,
                    last_center=last_center,
                    last_zoom=last_zoom,
                    y_offset=100,
                    zoom_speed=zoom_speed,
                )
                frame = cv2.resize(cropped, (output_width, output_height))
                last_center = new_center
                last_zoom = new_zoom
            else:
                crop_bounds = frame_info.get("crop_boundaries")
                if crop_bounds:
                    x1, y1, x2, y2 = (
                        crop_bounds["crop_x1"],
                        crop_bounds["crop_y1"],
                        crop_bounds["crop_x2"],
                        crop_bounds["crop_y2"],
                    )
                    if x2 > x1 and y2 > y1:
                        frame = frame[y1:y2, x1:x2]
                last_zoom = 1.0

            # Resize and overlay/annotate BEFORE adding to the frame buffer so sizes match during blending
            frame = cv2.resize(frame, (output_width, output_height))
            # Apply injection overlay for fadeout frames so crossfades include the overlay
            frame = overlay_injections_on_frame(
                frame, frame_info["time"], injections, fps
            )
            word = frame_info.get("word", "")

            # Append resized frame to buffer (ensures consistent sizes for blending)
            frame_buffer.append(frame)
            if len(frame_buffer) > fade_frames:
                frame_buffer.pop(0)

            fadeout_frames.append([frame, word])
            frame = add_text_with_shadow(
                frame, word, font_scale=font_scale, text_height_from_bottom=text_height
            )

            processed_frames += 1

    for frame_and_word in fadeout_frames:
        frame, word = frame_and_word
        frame = add_text_with_shadow(
            frame, word, font_scale=font_scale, text_height_from_bottom=text_height
        )
        out.write(frame)
        processed_frames += 1

    cap.release()
    out.release()
    if visualize:
        cv2.destroyAllWindows()

    print(f"✅ Processing complete. Processed {processed_frames} frames total.")
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
        "ffmpeg",
        "-i",
        video_file,
        "-i",
        audio_file,
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-strict",
        "experimental",
        "-y",  # Overwrite output file if it exists
        final_output,
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Video and audio combined to: {final_output}")
    except subprocess.CalledProcessError as e:
        print(f"Error combining video and audio: {e}")
        return False

    return True


def panel2tiktok(args):

    # File paths
    csv_file = args.csv
    video_file = args.video
    audio_file = args.audio
    final_output = video_file + "_final.mp4"
    output_audio_file = audio_file + "_extracted.wav"


    # Preprocess injections (if provided) into images sized to the output dimensions.
    # This creates a local `injections` variable (or None) which will be passed around.
    injectionsfilename = getattr(args, "injections", None)
    injectionslist = []
    if injectionsfilename:
        injections_df = pd.read_csv(injectionsfilename)
        injectionslist = injections_df.to_dict('records')
        # Preload injection images for performance
        injectionslist = preprocess_injections(injectionslist, args.width, args.height)

    print("Starting video/audio extraction...")

    audio = AudioSegment.from_file(audio_file)
    sr = audio.frame_rate

    # --- Load CSV + JSON ---
    word_segment_times_df = pd.read_csv(csv_file)
    # align all times to audio sample boundaries
    word_segment_times_df["start"] = (word_segment_times_df["start"] * sr).astype(
        int
    ) / sr
    word_segment_times_df["end"] = (word_segment_times_df["end"] * sr).astype(int) / sr
    # Convert DataFrame to list of dictionaries for easier processing
    word_segment_times = word_segment_times_df.to_dict("records")
    word_segment_times = fix_zero_length_words(
        word_segment_times, sr, shift_seconds=args.shift_seconds
    )

    # Handle optional start time: drop segments that end before start_time and trim segments that overlap
    start_time = max(0.0, float(getattr(args, "start_time", 0.0) or 0.0))
    if start_time > 0.0:
        trimmed_segments = []
        for seg in word_segment_times:
            # skip segments that end before the requested start
            if seg.get("end", 0.0) <= start_time:
                continue
            new_seg = dict(seg)
            # trim segment start if it begins before the requested start_time
            if new_seg.get("start", 0.0) < start_time:
                new_seg["start"] = start_time
            trimmed_segments.append(new_seg)
        word_segment_times = trimmed_segments

    # Load speaking_times JSON if provided and valid. If missing or invalid, disable speaker cropping.
    speaking_times = None
    camera_keyframes = None

    # Check for camera keyframes first (takes precedence over speaking_times)
    if getattr(args, "camera_keyframes", None):
        camera_path = args.camera_keyframes
        if os.path.exists(camera_path):
            # Get video duration for interpolation
            cap = cv2.VideoCapture(video_file)
            fps = probe_fps(video_file)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            total_duration = total_frames / fps
            cap.release()

            camera_keyframes = load_and_interpolate_camera_keyframes(
                camera_path, total_duration, fps
            )
            print(f"✅ Loaded {len(camera_keyframes)} camera keyframes")
        else:
            print(f"Warning: camera keyframes file '{camera_path}' not found.")

    # Only load speaking_times if camera_keyframes not provided
    elif getattr(args, "speaking_times", None):
        st_path = args.speaking_times
        with open(st_path, "r") as f:
            speaking_times = json.load(f)

    if not args.no_audio:
        print("\n=== Extracting Audio Segments ===")
        extract_audio_segments(
            word_segment_times,
            audio_file,
            output_audio_file,
            fade_samples=args.fade_samples,
        )

    # Check video file only if not in audio-only mode
    if not os.path.exists(video_file):
        print(f"Error: Video file '{video_file}' not found")
        return

    silent_video_file = video_file + "_cropped.mp4"

    if not args.no_video:
        total_frames = int(cv2.VideoCapture(video_file).get(cv2.CAP_PROP_FRAME_COUNT))
        fps = probe_fps(video_file)
        precomputed_segments = precompute_segments(word_segment_times, fps)

        # If user requested a single frame preview, generate the frame list (using camera keyframes
        # or speaker positioning if available), seek to the requested frame, display it, and exit.
        if getattr(args, "get_frame", None) is not None:
            get_time = float(args.get_frame)
            get_frame_idx = int(round(get_time * fps))
            precomputed_segments = [{'word': 'test', 'start_frame': get_frame_idx, 'end_frame': get_frame_idx+1, 'start_sec': get_time, 'end_sec': get_time + 1.0/fps}]

        # Get input video dimensions
        cap = cv2.VideoCapture(video_file)
        input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        # Choose frame list generation method
        if camera_keyframes:
            frame_number_and_bounds = generate_frame_list_from_camera_keyframes(
                camera_keyframes,
                precomputed_segments,
                input_width=input_width,
                input_height=input_height,
                output_width=args.width,
                output_height=args.height,
                fps=fps,
            )
            print("✅ Using camera keyframes for crop positioning")

        elif speaking_times:
            speaking_times_averaged, json_times = preprocess_speaker_data(
                speaking_times
            )
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
                output_height=args.height,
            )
            print("✅ Using speaker detection for crop positioning")

        else:
            # No positioning data: create simple frame list
            frame_number_and_bounds = []
            for seg in precomputed_segments:
                seg_list = []
                start_frame = seg["start_frame"]
                end_frame = seg["end_frame"]
                word = seg.get("word", "")
                for frame_idx in range(start_frame, end_frame):
                    seg_list.append(
                        {
                            "frame_idx": frame_idx,
                            "time": frame_idx / fps,
                            "speaker_id": None,
                            "speaker_box": None,
                            "crop_boundaries": None,
                            "word": word,
                        }
                    )
                frame_number_and_bounds.append(seg_list)
            print("✅ Using simple frame list (no positioning)")

        # Save frame list for inspection
        with open("frame_number_and_bounds.json", "w") as f:
            # Convert to serializable format
            serializable_list = []
            for seg in frame_number_and_bounds:
                serializable_seg = []
                for frame_info in seg:
                    serializable_seg.append(frame_info)
                serializable_list.append(serializable_seg)
            json.dump(serializable_list, f, indent=2)
        print("✅ Frame list written to frame_number_and_bounds.json")

        # compute video_start_frame from the provided start_time (if any)
        video_start_frame = 0
        try:
            video_start_frame = int(max(0.0, float(getattr(args, "start_time", 0.0) or 0.0)) * fps)
        except Exception:
            video_start_frame = 0

        extract_video_segments(
            frame_number_and_bounds,
            video_file,
            silent_video_file,
            output_width=args.width,
            output_height=args.height,
            fps=fps,
            visualize=args.visualize,
            fade_frames=args.fade_frames,
            text_height=args.text_height,
            zoom_speed=args.zoom_speed,
            font_scale=args.font_scale,
            injections=injectionslist,
            get_frame=args.get_frame,
            video_start_frame=video_start_frame,
        )

    if not args.no_combine:
        print("\n=== Combining Video and Audio ===")
        combine_video_audio(silent_video_file, output_audio_file, final_output)


def main():
    """Main function to extract and combine video/audio segments."""

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Extract video segments based on CSV with optional JSON cropping"
    )
    parser.add_argument(
        "--speaking_times",
        type=str,
        help="JSON file with bounding box data for speaker cropping",
    )
    parser.add_argument(
        "--camera-keyframes",
        type=str,
        help="CSV file with camera keyframes (center_x, center_y, height, width, zoom)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="words.csv",
        help="CSV file with time segments (default: words.csv)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default="deblurred.mp4",
        help="Input video file (default: deblurred.mp4)",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default="GBRvid4_audio.wav",
        help="Input audio file (default: GBRvid4_audio.wav)",
    )
    parser.add_argument(
        "--width", type=int, default=640, help="Output window width (default: 640)"
    )
    parser.add_argument(
        "--height", type=int, default=480, help="Output window height (default: 480)"
    )
    parser.add_argument(
        "--cycle-after",
        type=int,
        default=8,
        help="How long to cycle off of an active speaker",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Show real-time visualization of extraction process",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=1.8,
        help="Font scale for text overlay (default: 4.0)",
    )
    parser.add_argument(
        "--min-stay-duration",
        type=float,
        default=5,
        help="Minimum time in seconds to stay on a person after switching target (default: 1.5)",
    )
    parser.add_argument("--no-video", action="store_true", help="Skip video processing")
    parser.add_argument("--no-audio", action="store_true", help="Skip audio processing")
    parser.add_argument("--no-combine", action="store_true", help="Skip combine")
    parser.add_argument(
        "--save-lengths", action="store_true", help="Save segment lengths summary file"
    )
    parser.add_argument(
        "--fade-samples",
        type=int,
        default=3000,
        help="Crossfade length in samples for audio stitching (default: 256)",
    )
    parser.add_argument(
        "--fade-frames",
        type=int,
        default=6,
        help="Crossfade length in Crossfade for video stitching (default: 6)",
    )
    parser.add_argument(
        "--text-height", type=int, default=300, help="Height of test above bottom"
    )
    parser.add_argument(
        "--shift-seconds",
        type=int,
        default=0.15,
        help="sometimes A/V is aligned but the segment times are shifted (thanks, whisper)",
    )
    parser.add_argument(
        "--zoom-speed",
        type=float,
        default=0.02,
        help="Zoom interpolation speed per frame (0 = no zoom, higher = faster; default: 0.02)",
    )

    parser.add_argument(
        "--start-time",
        type=float,
        default=0.0,
        help="Start time (in seconds) to begin extraction; segments before this time will be skipped or trimmed",
    )

    parser.add_argument(
        "--get-frame",
        type=float,
        default=None,
        help="If provided, show a single frame at this time (seconds) and exit. Works with a single-segment CSV.",
    )

    parser.add_argument(
        "--injections",
        type=str,
        default=None,
        help="Images injected into the video",
    )
    args = parser.parse_args()
    panel2tiktok(args)


if __name__ == "__main__":
    main()
