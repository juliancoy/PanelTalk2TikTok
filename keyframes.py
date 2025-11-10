import os
import bisect
import pandas as pd
import numpy as np
from tqdm import tqdm
from io import StringIO


def load_and_interpolate_camera_keyframes(camera_keyframes_file, total_duration, fps):
    """
    Load camera keyframes from CSV and interpolate for every frame.
    
    Expected CSV columns: time, center_x, center_y, height, width, zoom
    - time: in seconds
    - center_x, center_y: coordinates in original video space
    - height, width: dimensions of crop area in original video space
    - zoom: zoom factor (optional, defaults to 1.0)
    
    Returns: list of dicts with keys: frame_idx, center_x, center_y, crop_width, crop_height, zoom
    """
    if not os.path.exists(camera_keyframes_file):
        raise FileNotFoundError(f"Camera keyframes file not found: {camera_keyframes_file}")
    

    uncommented = ""
    with open(camera_keyframes_file, 'r') as f:
        for line in f.read().split("\n"):
            uncommented += line.split("#")[0] + "\n"

    df = pd.read_csv(StringIO(uncommented))
    required_cols = ['time', 'center_x', 'center_y', 'height', 'width']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in camera keyframes CSV")
    
    # Add zoom column if not present, default to 1.0
    if 'zoom' not in df.columns:
        df['zoom'] = 1.0
    
    # Sort by time
    df = df.sort_values('time').reset_index(drop=True)
    
    # Create time points for all frames
    total_frames = int(total_duration * fps)
    frame_times = [i / fps for i in range(total_frames)]
    
    # Interpolate each parameter
    interpolated_data = []
    for frame_idx, current_time in enumerate(frame_times):
        # Find bounding keyframes
        if current_time <= df['time'].iloc[0]:
            # Before first keyframe, use first keyframe
            before_idx = 0
            after_idx = 0
        elif current_time >= df['time'].iloc[-1]:
            # After last keyframe, use last keyframe
            before_idx = len(df) - 1
            after_idx = len(df) - 1
        else:
            # Find keyframes that bracket current time
            after_idx = bisect.bisect_left(df['time'].values, current_time)
            before_idx = after_idx - 1
        
        if before_idx == after_idx:
            # Exact match or at boundaries
            frame_data = {
                'frame_idx': frame_idx,
                'center_x': float(df['center_x'].iloc[before_idx]),
                'center_y': float(df['center_y'].iloc[before_idx]),
                'crop_width': float(df['width'].iloc[before_idx]),
                'crop_height': float(df['height'].iloc[before_idx]),
                'zoom': float(df['zoom'].iloc[before_idx])
            }
        else:
            # Linear interpolation between keyframes
            t_before = df['time'].iloc[before_idx]
            t_after = df['time'].iloc[after_idx]
            alpha = (current_time - t_before) / (t_after - t_before)
            
            frame_data = {
                'frame_idx': frame_idx,
                'center_x': float(np.interp(alpha, [0, 1], 
                                    [df['center_x'].iloc[before_idx], df['center_x'].iloc[after_idx]])),
                'center_y': float(np.interp(alpha, [0, 1], 
                                    [df['center_y'].iloc[before_idx], df['center_y'].iloc[after_idx]])),
                'crop_width': float(np.interp(alpha, [0, 1], 
                                      [df['width'].iloc[before_idx], df['width'].iloc[after_idx]])),
                'crop_height': float(np.interp(alpha, [0, 1], 
                                       [df['height'].iloc[before_idx], df['height'].iloc[after_idx]])),
                'zoom': float(np.interp(alpha, [0, 1], 
                                [df['zoom'].iloc[before_idx], df['zoom'].iloc[after_idx]]))
            }
        
        interpolated_data.append(frame_data)
    
    return interpolated_data

def generate_frame_list_from_camera_keyframes(camera_data, precomputed_segments, 
                                           input_width, input_height, 
                                           output_width, output_height, fps):
    """
    Generate frame list using precomputed camera keyframe data.
    
    Args:
        camera_data: List of camera parameters from load_and_interpolate_camera_keyframes
        precomputed_segments: Word timing segments
        input_width, input_height: Original video dimensions
        output_width, output_height: Output video dimensions
        fps: frames per second of the source video
    
    Returns:
        Frame list compatible with extract_video_segments
    """
    print("Generating frame list from camera keyframes...")
    
    # Create a lookup dictionary for camera data by frame index
    camera_by_frame = {item['frame_idx']: item for item in camera_data}
    
    frame_list = []
    
    for seg in tqdm(precomputed_segments, desc="Generating frame list from keyframes"):
        seg_list = []
        start_frame = seg["start_frame"]
        end_frame = seg["end_frame"]
        word = seg["word"]
        
        for frame_idx in range(start_frame, end_frame):
            camera_info = camera_by_frame.get(frame_idx)
            
            if camera_info is None:
                # Fallback: center of frame, no zoom
                crop_boundaries = {
                    "crop_x1": 0,
                    "crop_y1": 0,
                    "crop_x2": output_width,
                    "crop_y2": output_height,
                    "center_x": input_width / 2,
                    "center_y": input_height / 2
                }
            else:
                # Calculate actual crop boundaries from camera parameters
                center_x = camera_info['center_x']
                center_y = camera_info['center_y']
                crop_width = camera_info['crop_width'] / camera_info.get('zoom', 1.0)
                crop_height = camera_info['crop_height'] / camera_info.get('zoom', 1.0)
                
                # Ensure crop dimensions don't exceed input dimensions
                crop_width = min(crop_width, input_width)
                crop_height = min(crop_height, input_height)
                
                # Calculate crop boundaries
                crop_x1 = int(center_x - crop_width / 2)
                crop_y1 = int(center_y - crop_height / 2)
                crop_x2 = int(center_x + crop_width / 2)
                crop_y2 = int(center_y + crop_height / 2)
                
                # Adjust boundaries to stay within frame
                crop_x1 = max(0, crop_x1)
                crop_y1 = max(0, crop_y1)
                crop_x2 = min(input_width, crop_x2)
                crop_y2 = min(input_height, crop_y2)
                
                # Ensure positive dimensions
                if crop_x2 <= crop_x1:
                    crop_x2 = crop_x1 + 1
                if crop_y2 <= crop_y1:
                    crop_y2 = crop_y1 + 1
                
                crop_boundaries = {
                    "crop_x1": crop_x1,
                    "crop_y1": crop_y1,
                    "crop_x2": crop_x2,
                    "crop_y2": crop_y2,
                    "center_x": center_x,
                    "center_y": center_y
                }
            
            seg_list.append({
                "frame_idx": frame_idx,
                "time": frame_idx / fps,
                "speaker_id": None,
                "speaker_box": None,
                "crop_boundaries": crop_boundaries,
                "word": word
            })
        
        frame_list.append(seg_list)
    
    print(f"✅ Camera keyframe frame list generation complete.")
    return frame_list
