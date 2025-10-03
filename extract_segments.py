#!/usr/bin/env python3
"""
Script to extract video and audio segments based on time periods from a CSV file
and combine them into a final output video using OpenCV.

Modified to also accept JSON input with bounding box data for speaker-focused extraction.
"""

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

def preprocess_speaker_data(json_data, window_size=3.0):
    """
    Preprocess speaker data with a centered mode filter to smooth speaker transitions.
    Optimized with NumPy for better performance.
    
    Args:
        json_data: List of JSON entries with time and people data
        window_size: Size of the moving window in seconds
    
    Returns:
        List of JSON entries with smoothed speaker data
    """
    if not json_data:
        return json_data
    
    # Sort by time
    json_data = sorted(json_data, key=lambda x: x['time'])
    
    # Extract times for faster window calculations
    times = np.array([entry['time'] for entry in json_data])
    
    # Precompute all speaker data for faster processing
    all_speaker_data = []
    for entry in json_data:
        speakers = {}
        for person in entry['people']:
            if person.get('speaking', False):
                speakers[person['id']] = 1
        all_speaker_data.append(speakers)
    
    # Create a new list for processed data
    processed_data = []
    
    for i, entry in enumerate(json_data):
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
    
    return processed_data

def extract_video_segments_with_cropping(csv_file, json_file, video_file, output_video_file, output_width, output_height, visualize=False, font_scale=4.0, min_stay_duration=1.5):
    """
    Extract video segments based on CSV time periods and crop each frame to focus on the speaker using JSON data.
    Stays on each person for at least min_stay_duration seconds after switching target.
    After 8 seconds of continuous speaking, cycles through all people randomly including whole scene view.
    
    Args:
        csv_file (str): Path to CSV file with start, end, word columns
        json_file (str): Path to JSON file with time, people, and speaking data
        video_file (str): Path to video file
        output_video_file (str): Path for output video file
        output_width (int): Width of output window
        output_height (int): Height of output window
        visualize (bool): Whether to show real-time visualization
        font_scale (float): Font scale for text overlay
        min_stay_duration (float): Minimum time in seconds to stay on a person after switching target
    """
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Read JSON file
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    
    # Preprocess speaker data with 3-second mode filter
    print("Preprocessing speaker data with 3-second mode filter...")
    json_data = preprocess_speaker_data(json_data, window_size=3.0)
    print("Speaker data preprocessing complete.")
    
    # Open video file
    cap = cv2.VideoCapture(video_file)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
    print(f"Output window size: {output_width}x{output_height}")
    print(f"Minimum stay duration: {min_stay_duration}s")
    
    # Create video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (output_width, output_height))
    
    # Create a mapping from time to JSON entry for quick lookup
    json_time_map = {}
    for entry in json_data:
        time = entry['time']
        json_time_map[time] = entry
    
    # Initialize speaker tracking variables
    current_speaker = None
    last_speaker_change_time = 0
    last_speaker_box = None
    current_speaker_start_time = 0
    last_camera_center = None
    # Track last visit time for each person
    last_visit_times = {}
    
    # Initialize visualization window if needed
    if visualize:
        cv2.namedWindow('Extraction Visualizer', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Extraction Visualizer', 1200, 800)
        print("Visualizer started. Press 'q' to quit, 'p' to pause.")

    # Precompute list of times once, outside the frame loop
    json_times = [entry['time'] for entry in json_data]

    # Extract segments based on CSV
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing segments"):
        start_time = row['start']
        end_time = row['end']
        word = row['word']
        
        # Convert time to frame numbers
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # Set starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Extract frames for this segment
        for frame_num in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate current time for this frame
            current_time = frame_num / fps
            
            # Then inside your frame loop:
            idx = bisect.bisect_left(json_times, current_time)

            # Pick the closest index (either idx or idx-1)
            if idx == 0:
                closest_entry = json_data[0]
            elif idx == len(json_times):
                closest_entry = json_data[-1]
            else:
                before = json_data[idx - 1]
                after = json_data[idx]
                if abs(before['time'] - current_time) <= abs(after['time'] - current_time):
                    closest_entry = before
                else:
                    closest_entry = after
            
            if closest_entry:
                # Find the speaking person
                current_speaker_candidate = None
                for person in closest_entry['people']:
                    if person.get('speaking', False):
                        current_speaker_candidate = person
                        break
                
                # Check if we should switch to a new speaker
                should_switch_speaker = False
                if current_speaker_candidate:
                    if current_speaker is None:
                        # No current speaker, switch immediately
                        should_switch_speaker = True
                    elif current_speaker_candidate['id'] != current_speaker['id']:
                        # Different speaker, check if we've stayed long enough
                        time_since_last_change = current_time - last_speaker_change_time
                        if time_since_last_change >= min_stay_duration:
                            should_switch_speaker = True
                
                # Update speaker if needed
                if should_switch_speaker:
                    # Update visit time for current speaker before switching
                    if current_speaker:
                        last_visit_times[current_speaker['id']] = current_time
                    
                    current_speaker = current_speaker_candidate
                    last_speaker_change_time = current_time
                    current_speaker_start_time = current_time
                    if current_speaker:
                        last_speaker_box = current_speaker['box']
                        # Update visit time for new speaker
                        last_visit_times[current_speaker['id']] = current_time
                        #print(f"Switched to speaker {current_speaker['id']} at {current_time:.2f}s")
                
                # Check if we should switch to least recently visited person or zoomed-out mode (after 8 seconds of continuous speaking)
                if current_speaker:
                    time_speaking = current_time - current_speaker_start_time
                    if time_speaking >= 8.0:
                        # Check if we've stayed long enough on current speaker
                        time_since_last_change = current_time - last_speaker_change_time
                        if time_since_last_change >= min_stay_duration:
                            # Find available people (excluding current speaker)
                            available_people = [p for p in closest_entry['people'] if p['id'] != current_speaker['id']]
                            
                            if available_people:
                                # Find the least recently visited person
                                least_recent_person = None
                                least_recent_time = float('inf')
                                
                                for person in available_people:
                                    person_id = person['id']
                                    last_visit = last_visit_times.get(person_id, 0)  # Default to 0 if never visited
                                    if last_visit < least_recent_time:
                                        least_recent_time = last_visit
                                        least_recent_person = person
                                
                                if least_recent_person:
                                    # Update visit time for current speaker before switching
                                    last_visit_times[current_speaker['id']] = current_time
                                    
                                    # Switch to least recently visited person
                                    current_speaker = least_recent_person
                                    last_speaker_box = current_speaker['box']
                                    current_speaker_start_time = current_time
                                    last_speaker_change_time = current_time
                                    # Update visit time for new speaker
                                    last_visit_times[current_speaker['id']] = current_time
                                    print(f"Switched to least recently visited speaker {current_speaker['id']} at {current_time:.2f}s after {time_speaking:.2f}s of speaking")
                                            # Randomly decide whether to show zoomed-out view or switch to another person
                            else: 
                                # Switch to zoomed-out mode with blue background
                                current_speaker = None
                                last_speaker_box = None
                                current_speaker_start_time = current_time
                                last_speaker_change_time = current_time
                                print(f"Switched to zoomed-out mode at {current_time:.2f}s after {time_speaking:.2f}s of speaking")

                # Use current speaker for cropping
                target_person = current_speaker
                
                # Process frame based on target
                if target_person and target_person.get('box'):
                    # Extract cropped segment centered on person with camera movement smoothing
                    cropped_frame, new_center = extract_cropped_segment(frame, target_person['box'], output_width, output_height, last_camera_center)
                    last_camera_center = new_center
                    
                    # Add word overlay with drop shadow
                    frame_with_text = add_text_with_shadow(cropped_frame, word, font_scale=font_scale)
                    
                    out.write(frame_with_text)
                    
                    # Show visualization if enabled
                    if visualize:
                        continue_processing = visualize_extraction_process(
                            frame_with_text, current_time, word, target_person, frame, show_original=True
                        )
                        if not continue_processing:
                            print("Visualizer stopped by user.")
                            cap.release()
                            out.release()
                            cv2.destroyAllWindows()
                            return
                    
                    #print(f"Extracted frame {frame_num} at {current_time:.2f}s for {target_person['id']}")
                
                # Use current speaker for cropping (original logic)
                elif current_speaker and last_speaker_box:
                    # Extract cropped segment centered on speaker with camera movement smoothing
                    cropped_frame, new_center = extract_cropped_segment(frame, last_speaker_box, output_width, output_height, last_camera_center)
                    last_camera_center = new_center
                    
                    # Add word overlay with drop shadow
                    frame_with_text = add_text_with_shadow(cropped_frame, word, font_scale=font_scale)
                    
                    out.write(frame_with_text)
                    
                    # Show visualization if enabled
                    if visualize:
                        continue_processing = visualize_extraction_process(
                            frame_with_text, current_time, word, current_speaker, frame, show_original=True
                        )
                        if not continue_processing:
                            print("Visualizer stopped by user.")
                            cap.release()
                            out.release()
                            cv2.destroyAllWindows()
                            return
                    
                    print(f"Extracted frame {frame_num} at {current_time:.2f}s for speaker {current_speaker['id']}")
                else:
                    # Zoomed-out mode with blue background and scaled video in center
                    blue_background = np.full((output_height, output_width, 3), [255, 0, 0], dtype=np.uint8)  # Blue background
                    
                    # Scale the original frame to match output width while maintaining aspect ratio
                    frame_height, frame_width = frame.shape[:2]
                    scale_factor = output_width / frame_width
                    scaled_height = int(frame_height * scale_factor)
                    
                    # Resize frame to match output width
                    scaled_frame = cv2.resize(frame, (output_width, scaled_height))
                    
                    # Calculate vertical position to center the scaled frame
                    y_offset = (output_height - scaled_height) // 2
                    
                    # Place scaled frame in center of blue background
                    if y_offset >= 0:
                        blue_background[y_offset:y_offset+scaled_height, :] = scaled_frame
                    else:
                        # If scaled frame is taller than output, crop it to fit
                        crop_start = -y_offset // 2
                        crop_end = crop_start + output_height
                        blue_background = scaled_frame[crop_start:crop_end, :]
                    
                    frame_with_text = add_text_with_shadow(blue_background, word, font_scale=font_scale)
                    out.write(frame_with_text)
                    
                    # Show visualization if enabled
                    if visualize:
                        continue_processing = visualize_extraction_process(
                            frame_with_text, current_time, word, None, frame, show_original=True
                        )
                        if not continue_processing:
                            print("Visualizer stopped by user.")
                            cap.release()
                            out.release()
                            cv2.destroyAllWindows()
                            return
            else:
                print(f"No JSON data found for time {current_time:.2f}s, using full frame")
                # If no JSON data found, use full frame (resized to output dimensions)
                resized_frame = cv2.resize(frame, (output_width, output_height))
                frame_with_text = add_text_with_shadow(resized_frame, word, font_scale=font_scale)
                out.write(frame_with_text)
                
                # Show visualization if enabled
                if visualize:
                    continue_processing = visualize_extraction_process(
                        frame_with_text, current_time, word, None, frame, show_original=True
                    )
                    if not continue_processing:
                        print("Visualizer stopped by user.")
                        cap.release()
                        out.release()
                        cv2.destroyAllWindows()
                        return
    
    # Release resources
    cap.release()
    out.release()
    
    # Close visualization window if it was opened
    if visualize:
        cv2.destroyAllWindows()
    
    print(f"Video segments extracted to: {output_video_file}")

def extract_video_segments(csv_file, video_file, output_video_file, visualize=False, font_scale=4.0):
    """
    Extract video segments using OpenCV based on time periods in CSV.
    Each segment will have the corresponding word superimposed with drop shadow.
    
    Args:
        csv_file (str): Path to CSV file with start, end, word columns
        video_file (str): Path to video file
        output_video_file (str): Path for output video file
        visualize (bool): Whether to show real-time visualization
        font_scale (float): Font scale for text overlay
    """
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Open video file
    cap = cv2.VideoCapture(video_file)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # Create video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (width, height))
    
    # Initialize visualization window if needed
    if visualize:
        cv2.namedWindow('Extraction Visualizer', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Extraction Visualizer', 1200, 800)
        print("Visualizer started. Press 'q' to quit, 'p' to pause.")
    
    # Extract segments
    for _, row in df.iterrows():
        start_time = row['start']
        end_time = row['end']
        word = row['word']
        
        # Convert time to frame numbers
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        print(f"Extracting segment: '{word}' from {start_time:.2f}s to {end_time:.2f}s "
              f"(frames {start_frame} to {end_frame})")
        
        # Set starting position
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Extract frames for this segment
        for frame_num in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                print(f"Warning: Could not read frame {frame_num}")
                break
            
            # Calculate current time for this frame
            current_time = frame_num / fps
            
            # Add word overlay with drop shadow
            frame_with_text = add_text_with_shadow(frame, word, font_scale=font_scale)
            
            out.write(frame_with_text)
            
            # Show visualization if enabled
            if visualize:
                continue_processing = visualize_extraction_process(
                    frame_with_text, current_time, word, None, frame, show_original=True
                )
                if not continue_processing:
                    print("Visualizer stopped by user.")
                    cap.release()
                    out.release()
                    cv2.destroyAllWindows()
                    return
    
    # Release resources
    cap.release()
    out.release()
    
    # Close visualization window if it was opened
    if visualize:
        cv2.destroyAllWindows()
    
    print(f"Video segments extracted to: {output_video_file}")

def extract_audio_segments(csv_file, audio_file, output_audio_file, fade_samples=2000):
    """
    Extract audio segments using NumPy for processing and pydub only for I/O.
    Applies crossfade on every segment boundary for smooth transitions.
    
    Args:
        csv_file (str): Path to CSV file with start, end, word columns
        audio_file (str): Path to audio file
        output_audio_file (str): Path for output audio file
        fade_samples (int): Number of samples for crossfade (default: 128)
    """
    # Read CSV file
    df = pd.read_csv(csv_file)

    # Load audio file once and convert to NumPy array
    audio = AudioSegment.from_file(audio_file)
    sample_rate = audio.frame_rate
    channels = audio.channels
    dtype = np.int16 if audio.sample_width == 2 else np.int8

    # Convert entire audio to NumPy array for efficient slicing
    audio_samples = np.array(audio.get_array_of_samples(), dtype=dtype)

    # Reshape to (num_samples, channels) if stereo or more
    if channels > 1:
        audio_samples = audio_samples.reshape((-1, channels))

    # Initialize output as NumPy array
    output_samples = np.empty((0, channels), dtype=dtype) if channels > 1 else np.empty(0, dtype=dtype)

    # Extract segments with crossfade on every segment
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting audio"):
        start_time = row['start'] * 1000  # Convert to ms
        end_time = row['end'] * 1000

        # Calculate sample indices
        start_sample = int(start_time * sample_rate / 1000)
        end_sample = int(end_time * sample_rate / 1000)

        # Extract segment using NumPy slicing
        segment_samples = audio_samples[start_sample:end_sample].copy()

        # Apply crossfade on every segment (not just when pops are detected)
        if output_samples.size > 0:
            # Use overlap region for crossfade
            fade_len = min(fade_samples, len(output_samples), len(segment_samples))
            
            if fade_len > 0:
                # Split tail of output
                output_tail = output_samples[-fade_len:].astype(np.float32)
                output_keep = output_samples[:-fade_len]

                # Split head of segment
                seg_head = segment_samples[:fade_len].astype(np.float32)
                seg_rest = segment_samples[fade_len:]

                # Create complementary envelopes
                fade_out_env = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                fade_in_env  = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)

                if output_tail.ndim == 2:  # stereo/multi
                    fade_out_env = fade_out_env[:, None]
                    fade_in_env = fade_in_env[:, None]

                # Blend overlap region
                blended_overlap = (output_tail * fade_out_env + seg_head * fade_in_env).astype(output_samples.dtype)

                # New output = keep + blended overlap + rest
                output_samples = np.concatenate([output_keep, blended_overlap, seg_rest])
            else:
                # If no overlap possible, just concatenate
                output_samples = np.concatenate([output_samples, segment_samples])
        else:
            # First segment, no crossfade needed
            output_samples = segment_samples

    # Convert back to AudioSegment only at the end for export
    output_audio = AudioSegment(
        output_samples.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio.sample_width,
        channels=channels
    )

    # Export combined audio
    output_audio.export(output_audio_file, format="wav")
    print(f"✅ Audio segments extracted to: {output_audio_file}")


def combine_video_audio(video_file, audio_file, output_file):
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
        output_file
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"Video and audio combined to: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error combining video and audio: {e}")
        return False
    
    return True

def main():
    """Main function to extract and combine video/audio segments."""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract video segments based on CSV with optional JSON cropping')
    parser.add_argument('--json', type=str, help='JSON file with bounding box data for speaker cropping')
    parser.add_argument('--csv', type=str, default='words.csv', help='CSV file with time segments (default: words.csv)')
    parser.add_argument('--video', type=str, default='deblurred.mp4', help='Input video file (default: deblurred.mp4)')
    parser.add_argument('--audio', type=str, default='GBRvid4_audio.wav', help='Input audio file (default: GBRvid4_audio.wav)')
    parser.add_argument('--width', type=int, default=640, help='Output window width (default: 640)')
    parser.add_argument('--height', type=int, default=480, help='Output window height (default: 480)')
    parser.add_argument('--output', type=str, default='final_output.mp4', help='Output video file (default: final_output.mp4)')
    parser.add_argument('--visualize', action='store_true', help='Show real-time visualization of extraction process')
    parser.add_argument('--font-scale', type=float, default=4.0, help='Font scale for text overlay (default: 4.0)')
    parser.add_argument('--min-stay-duration', type=float, default=5, help='Minimum time in seconds to stay on a person after switching target (default: 1.5)')
    parser.add_argument('--audio-only', action='store_true', help='Extract audio only without video processing')
    
    args = parser.parse_args()
    
    # File paths
    csv_file = args.csv
    video_file = args.video
    audio_file = args.audio
    final_output = args.output
    
    # Check if files exist
    if not os.path.exists(csv_file):
        print(f"Error: CSV file '{csv_file}' not found")
        return
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found")
        return
    
    print("Starting video/audio extraction...")
    
    temp_output_audio = "temp_output_audio.wav"
    temp_output_video = "temp_output_video_cropped.mp4"

    # Audio-only mode
    if args.audio_only:
        print("\n=== Extracting Audio Segments (Audio Only Mode) ===")
        extract_audio_segments(csv_file, audio_file, temp_output_audio)
        print(f"\n✅ Audio output created: {temp_output_audio}")
        # Combine video and audio
        print("\n=== Combining Video and Audio ===")
        if combine_video_audio(temp_output_video, temp_output_audio, final_output):
            print(f"\n✅ Final output created: {final_output}")
            print("Note: Temporary files (output_video_cropped.mp4, output_audio.wav) were not cleaned up")
        else:
            print("\n❌ Failed to create final output")
        return
    
    # Check video file only if not in audio-only mode
    if not os.path.exists(video_file):
        print(f"Error: Video file '{video_file}' not found")
        return
    
    if args.json:
        # Combined mode - use CSV for time segments and JSON for speaker cropping
        if not os.path.exists(args.json):
            print(f"Error: JSON file '{args.json}' not found")
            return
        
        print("\n=== Extracting Video Segments with Speaker Cropping (Combined Mode) ===")
        extract_video_segments_with_cropping(csv_file, args.json, video_file, temp_output_video, args.width, args.height, visualize=args.visualize, font_scale=args.font_scale, min_stay_duration=args.min_stay_duration)
        
        print("\n=== Extracting Audio Segments ===")
        extract_audio_segments(csv_file, audio_file, temp_output_audio)
        
        # Combine video and audio
        print("\n=== Combining Video and Audio ===")
        if combine_video_audio(temp_output_video, temp_output_audio, final_output):
            print(f"\n✅ Final output created: {final_output}")
            print("Note: Temporary files (output_video_cropped.mp4, output_audio.wav) were not cleaned up")
        else:
            print("\n❌ Failed to create final output")
    
    else:
        # CSV mode - original functionality (no cropping)
        print("\n=== Extracting Video Segments (CSV Mode - No Cropping) ===")
        temp_output_video = "output_video.mp4"
        extract_video_segments(csv_file, video_file, temp_output_video, visualize=args.visualize, font_scale=args.font_scale)
        
        print("\n=== Extracting Audio Segments ===")
        temp_output_audio = "output_audio.wav"
        extract_audio_segments(csv_file, audio_file, temp_output_audio)
        
        # Combine video and audio
        print("\n=== Combining Video and Audio ===")
        if combine_video_audio(temp_output_video, temp_output_audio, final_output):
            print(f"\n✅ Final output created: {final_output}")
            print("Note: Temporary files (output_video.mp4, output_audio.wav) were not cleaned up")
        else:
            print("\n❌ Failed to create final output")

if __name__ == "__main__":
    main()
