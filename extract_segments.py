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
    thickness = int(font_scale * 3/2.0)
    
    # Use a readable font
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Calculate text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    # Position text at bottom center (like closed captions)
    x = (width - text_width) // 2
    y = height - 100  # Position from bottom
    
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

def extract_cropped_segment(frame, speaker_box, output_width, output_height):
    """
    Extract a cropped segment from the frame centered on the speaker's bounding box.
    
    Args:
        frame: The video frame
        speaker_box: Bounding box [x1, y1, x2, y2] of the speaker
        output_width: Width of the output window
        output_height: Height of the output window
    
    Returns:
        Cropped frame centered on speaker
    """
    # Calculate center of bounding box
    center_x = (speaker_box[0] + speaker_box[2]) / 2
    center_y = (speaker_box[1] + speaker_box[3]) / 2
    
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
    
    return cropped_frame

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

def extract_video_segments_with_cropping(csv_file, json_file, video_file, output_video_file, output_width, output_height, visualize=False, font_scale=4.0):
    """
    Extract video segments based on CSV time periods and crop each frame to focus on the speaker using JSON data.
    
    Args:
        csv_file (str): Path to CSV file with start, end, word columns
        json_file (str): Path to JSON file with time, people, and speaking data
        video_file (str): Path to video file
        output_video_file (str): Path for output video file
        output_width (int): Width of output window
        output_height (int): Height of output window
        visualize (bool): Whether to show real-time visualization
        font_scale (float): Font scale for text overlay
    """
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Read JSON file
    with open(json_file, 'r') as f:
        json_data = json.load(f)
    
    # Open video file
    cap = cv2.VideoCapture(video_file)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
    print(f"Output window size: {output_width}x{output_height}")
    
    # Create video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_file, fourcc, fps, (output_width, output_height))
    
    # Create a mapping from time to JSON entry for quick lookup
    json_time_map = {}
    for entry in json_data:
        time = entry['time']
        json_time_map[time] = entry
    
    # Initialize visualization window if needed
    if visualize:
        cv2.namedWindow('Extraction Visualizer', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Extraction Visualizer', 1200, 800)
        print("Visualizer started. Press 'q' to quit, 'p' to pause.")
    
    # Extract segments based on CSV
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
            
            # Find the closest JSON entry for this time
            closest_entry = None
            min_time_diff = float('inf')
            
            for entry in json_data:
                time_diff = abs(entry['time'] - current_time)
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_entry = entry
            
            if closest_entry:
                # Find the speaking person
                speaker = None
                for person in closest_entry['people']:
                    if person.get('speaking', False):
                        speaker = person
                        break
                
                if speaker:
                    # Extract cropped segment centered on speaker
                    speaker_box = speaker['box']
                    cropped_frame = extract_cropped_segment(frame, speaker_box, output_width, output_height)
                    
                    # Add word overlay with drop shadow
                    frame_with_text = add_text_with_shadow(cropped_frame, word, font_scale=font_scale)
                    
                    out.write(frame_with_text)
                    
                    # Show visualization if enabled
                    if visualize:
                        continue_processing = visualize_extraction_process(
                            frame_with_text, current_time, word, speaker, frame, show_original=True
                        )
                        if not continue_processing:
                            print("Visualizer stopped by user.")
                            cap.release()
                            out.release()
                            cv2.destroyAllWindows()
                            return
                    
                    print(f"Extracted frame {frame_num} at {current_time:.2f}s for speaker {speaker['id']}")
                else:
                    print(f"No speaker found at time {current_time:.2f}s, using full frame")
                    # If no speaker found, use full frame (resized to output dimensions)
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

def extract_audio_segments(csv_file, audio_file, output_audio_file):
    """
    Extract audio segments using pydub based on time periods in CSV.
    
    Args:
        csv_file (str): Path to CSV file with start, end, word columns
        audio_file (str): Path to audio file
        output_audio_file (str): Path for output audio file
    """
    
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Load audio file
    audio = AudioSegment.from_file(audio_file)
    
    # Create empty audio segment for output
    output_audio = AudioSegment.empty()
    
    # Extract segments
    for _, row in df.iterrows():
        start_time = row['start'] * 1000  # Convert to milliseconds
        end_time = row['end'] * 1000      # Convert to milliseconds
        word = row['word']
        
        print(f"Extracting audio segment: '{word}' from {start_time/1000:.2f}s to {end_time/1000:.2f}s")
        
        # Extract segment
        segment = audio[start_time:end_time]
        output_audio += segment
    
    # Export combined audio
    output_audio.export(output_audio_file, format="wav")
    print(f"Audio segments extracted to: {output_audio_file}")

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
    
    if not os.path.exists(video_file):
        print(f"Error: Video file '{video_file}' not found")
        return
    
    if not os.path.exists(audio_file):
        print(f"Error: Audio file '{audio_file}' not found")
        return
    
    print("Starting video/audio extraction...")
    
    if args.json:
        # Combined mode - use CSV for time segments and JSON for speaker cropping
        if not os.path.exists(args.json):
            print(f"Error: JSON file '{args.json}' not found")
            return
        
        print("\n=== Extracting Video Segments with Speaker Cropping (Combined Mode) ===")
        output_video = "output_video_cropped.mp4"
        extract_video_segments_with_cropping(csv_file, args.json, video_file, output_video, args.width, args.height, visualize=args.visualize, font_scale=args.font_scale)
        
        print("\n=== Extracting Audio Segments ===")
        output_audio = "output_audio.wav"
        extract_audio_segments(csv_file, audio_file, output_audio)
        
        # Combine video and audio
        print("\n=== Combining Video and Audio ===")
        if combine_video_audio(output_video, output_audio, final_output):
            print(f"\n✅ Final output created: {final_output}")
            
            # Clean up temporary files
            if os.path.exists(output_video):
                os.remove(output_video)
            if os.path.exists(output_audio):
                os.remove(output_audio)
            print("Temporary files cleaned up.")
        else:
            print("\n❌ Failed to create final output")
    
    else:
        # CSV mode - original functionality (no cropping)
        print("\n=== Extracting Video Segments (CSV Mode - No Cropping) ===")
        output_video = "output_video.mp4"
        extract_video_segments(csv_file, video_file, output_video, visualize=args.visualize, font_scale=args.font_scale)
        
        print("\n=== Extracting Audio Segments ===")
        output_audio = "output_audio.wav"
        extract_audio_segments(csv_file, audio_file, output_audio)
        
        # Combine video and audio
        print("\n=== Combining Video and Audio ===")
        if combine_video_audio(output_video, output_audio, final_output):
            print(f"\n✅ Final output created: {final_output}")
            
            # Clean up temporary files
            if os.path.exists(output_video):
                os.remove(output_video)
            if os.path.exists(output_audio):
                os.remove(output_audio)
            print("Temporary files cleaned up.")
        else:
            print("\n❌ Failed to create final output")

if __name__ == "__main__":
    main()
