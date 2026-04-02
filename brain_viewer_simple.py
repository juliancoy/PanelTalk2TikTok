#!/usr/bin/env python3
"""
brain_viewer_simple.py - Simple Python-based brain viewer for TribeV2 outputs

Usage:
    # View combined segments with video
    python brain_viewer_simple.py --video video.mp4 --npy combined.npy
    
    # View segments directory
    python brain_viewer_simple.py --segments /tmp/tribe_segments_xxx/
    
    # Interactive playback
    python brain_viewer_simple.py --video video.mp4 --npy combined.npy --play
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import subprocess
import tempfile
import os


class SimpleBrainViewer:
    """Simple matplotlib-based brain viewer."""
    
    def __init__(self):
        self.brain_data = None
        self.video_path = None
        self.n_timesteps = 0
        self.n_vertices = 0
        self.current_frame = 0
        
    def load_npy(self, npy_path: str):
        """Load brain prediction data."""
        self.brain_data = np.load(npy_path)
        self.n_timesteps, self.n_vertices = self.brain_data.shape
        print(f"[BrainViewer] Loaded: {npy_path}")
        print(f"[BrainViewer] Shape: {self.brain_data.shape}")
        print(f"[BrainViewer] Range: [{self.brain_data.min():.3f}, {self.brain_data.max():.3f}]")
        return self.brain_data
    
    def load_video(self, video_path: str):
        """Set video path."""
        self.video_path = video_path
        print(f"[BrainViewer] Video: {video_path}")
    
    def _brain_data_to_colors(self, values: np.ndarray) -> np.ndarray:
        """Convert BOLD values to RGB colors (fire colormap)."""
        vmin, vmax = np.percentile(self.brain_data, [1, 99])
        normalized = np.clip((values - vmin) / (vmax - vmin), 0, 1)
        
        colors = np.zeros((len(values), 3))
        colors[:, 0] = np.clip(normalized * 2, 0, 1)  # Red
        colors[:, 1] = np.clip((normalized - 0.5) * 2, 0, 1)  # Green
        colors[:, 2] = np.clip((normalized - 0.75) * 4, 0, 1)  # Blue
        
        return colors
    
    def _generate_brain_surface(self, n_vertices: int = 1000):
        """Generate a simple brain surface visualization."""
        # Create two hemispheres as point clouds
        np.random.seed(42)
        
        # Left hemisphere
        n_left = n_vertices // 2
        theta = np.random.uniform(0, np.pi, n_left)
        phi = np.random.uniform(0, 2*np.pi, n_left)
        r = 1.0 + np.random.normal(0, 0.1, n_left)
        
        x_left = r * np.sin(theta) * np.cos(phi) - 1.5
        y_left = r * np.sin(theta) * np.sin(phi)
        z_left = r * np.cos(theta)
        
        # Right hemisphere
        n_right = n_vertices - n_left
        theta = np.random.uniform(0, np.pi, n_right)
        phi = np.random.uniform(0, 2*np.pi, n_right)
        r = 1.0 + np.random.normal(0, 0.1, n_right)
        
        x_right = r * np.sin(theta) * np.cos(phi) + 1.5
        y_right = r * np.sin(theta) * np.sin(phi)
        z_right = r * np.cos(theta)
        
        x = np.concatenate([x_left, x_right])
        y = np.concatenate([y_left, y_right])
        z = np.concatenate([z_left, z_right])
        
        return x, y, z
    
    def view_static(self, timestep: int = 0):
        """View a single static frame."""
        fig = plt.figure(figsize=(16, 6))
        
        # Left: Brain visualization
        ax1 = fig.add_subplot(121, projection='3d')
        x, y, z = self._generate_brain_surface(min(self.n_vertices, 2000))
        
        # Sample brain data for visualization
        step = max(1, self.n_vertices // len(x))
        values = self.brain_data[timestep, ::step][:len(x)]
        colors = self._brain_data_to_colors(values)
        
        scatter = ax1.scatter(x, y, z, c=colors, s=20, alpha=0.6)
        ax1.set_title(f'Brain Activity (t={timestep})', fontsize=14)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Right: Timeseries
        ax2 = fig.add_subplot(122)
        mean_activity = np.mean(self.brain_data, axis=1)
        ax2.plot(mean_activity, linewidth=2, color='blue')
        ax2.axvline(x=timestep, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Timestep (2Hz)', fontsize=12)
        ax2.set_ylabel('Mean BOLD Response', fontsize=12)
        ax2.set_title('Activity Timeline', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def view_animation(self, interval: int = 500):
        """View animated brain activity."""
        fig = plt.figure(figsize=(16, 6))
        
        # Left: Brain visualization
        ax1 = fig.add_subplot(121, projection='3d')
        x, y, z = self._generate_brain_surface(min(self.n_vertices, 2000))
        
        # Initial scatter
        step = max(1, self.n_vertices // len(x))
        values = self.brain_data[0, ::step][:len(x)]
        colors = self._brain_data_to_colors(values)
        scatter = ax1.scatter(x, y, z, c=colors, s=20, alpha=0.6)
        ax1.set_title('Brain Activity (t=0)', fontsize=14)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Keep axis limits fixed
        ax1.set_xlim(-3, 3)
        ax1.set_ylim(-2, 2)
        ax1.set_zlim(-2, 2)
        
        # Right: Timeseries
        ax2 = fig.add_subplot(122)
        mean_activity = np.mean(self.brain_data, axis=1)
        line, = ax2.plot(mean_activity, linewidth=2, color='blue')
        vline = ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax2.set_xlabel('Timestep (2Hz)', fontsize=12)
        ax2.set_ylabel('Mean BOLD Response', fontsize=12)
        ax2.set_title('Activity Timeline', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        def update(frame):
            # Update brain colors
            values = self.brain_data[frame, ::step][:len(x)]
            colors = self._brain_data_to_colors(values)
            scatter._facecolor3d = colors
            scatter._edgecolor3d = colors
            ax1.set_title(f'Brain Activity (t={frame})', fontsize=14)
            
            # Update timeline
            vline.set_xdata([frame, frame])
            
            return scatter, vline
        
        anim = FuncAnimation(fig, update, frames=self.n_timesteps, 
                            interval=interval, blit=False, repeat=True)
        
        plt.show()
    
    def view_with_video(self, video_path: str = None, interval: int = 500):
        """View side-by-side with video frames (if available)."""
        try:
            import cv2
        except ImportError:
            print("[BrainViewer] OpenCV not available, showing brain only")
            self.view_animation(interval)
            return
        
        video = video_path or self.video_path
        if not video or not Path(video).exists():
            print("[BrainViewer] Video not found, showing brain only")
            self.view_animation(interval)
            return
        
        cap = cv2.VideoCapture(str(video))
        if not cap.isOpened():
            print("[BrainViewer] Could not open video")
            self.view_animation(interval)
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame mapping
        frame_map = np.linspace(0, total_frames-1, self.n_timesteps, dtype=int)
        
        fig = plt.figure(figsize=(18, 6))
        
        # Left: Video frame
        ax1 = fig.add_subplot(131)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
        im1 = ax1.imshow(frame)
        ax1.set_title('Video', fontsize=14)
        ax1.axis('off')
        
        # Middle: Brain
        ax2 = fig.add_subplot(132, projection='3d')
        x, y, z = self._generate_brain_surface(min(self.n_vertices, 2000))
        step = max(1, self.n_vertices // len(x))
        values = self.brain_data[0, ::step][:len(x)]
        colors = self._brain_data_to_colors(values)
        scatter = ax2.scatter(x, y, z, c=colors, s=20, alpha=0.6)
        ax2.set_title('Brain Activity (t=0)', fontsize=14)
        ax2.set_xlim(-3, 3)
        ax2.set_ylim(-2, 2)
        ax2.set_zlim(-2, 2)
        
        # Right: Timeseries
        ax3 = fig.add_subplot(133)
        mean_activity = np.mean(self.brain_data, axis=1)
        ax3.plot(mean_activity, linewidth=2, color='blue')
        vline = ax3.axvline(x=0, color='red', linestyle='--', linewidth=2)
        ax3.set_xlabel('Timestep (2Hz)', fontsize=12)
        ax3.set_ylabel('Mean BOLD Response', fontsize=12)
        ax3.set_title('Activity Timeline', fontsize=14)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        def update(frame):
            # Update video
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_map[frame])
            ret, video_frame = cap.read()
            if ret:
                video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
                im1.set_array(video_frame)
            
            # Update brain
            values = self.brain_data[frame, ::step][:len(x)]
            colors = self._brain_data_to_colors(values)
            scatter._facecolor3d = colors
            scatter._edgecolor3d = colors
            ax2.set_title(f'Brain Activity (t={frame})', fontsize=14)
            
            # Update timeline
            vline.set_xdata([frame, frame])
            
            return im1, scatter, vline
        
        anim = FuncAnimation(fig, update, frames=self.n_timesteps,
                            interval=interval, blit=False, repeat=True)
        
        plt.show()
        cap.release()


def load_segments(segments_dir: str, model: str = "vitg"):
    """Load and combine all segment files."""
    seg_dir = Path(segments_dir)
    files = sorted(seg_dir.glob(f"*_tribe_{model}.npy"))
    
    if not files:
        print(f"No segment files found in {segments_dir}")
        return None
    
    print(f"Loading {len(files)} segments...")
    all_data = [np.load(f) for f in files]
    combined = np.vstack(all_data)
    print(f"Combined shape: {combined.shape}")
    
    return combined


def main():
    parser = argparse.ArgumentParser(description='Simple Brain Viewer for TribeV2')
    parser.add_argument('--npy', '-n', help='Path to .npy file')
    parser.add_argument('--video', '-v', help='Path to video file')
    parser.add_argument('--segments', '-s', help='Directory containing segments')
    parser.add_argument('--model', default='vitg', help='Model suffix (default: vitg)')
    parser.add_argument('--animate', '-a', action='store_true', help='Show animation')
    parser.add_argument('--interval', type=int, default=500, help='Animation interval (ms)')
    parser.add_argument('--timestep', type=int, default=0, help='Static timestep to view')
    
    args = parser.parse_args()
    
    viewer = SimpleBrainViewer()
    
    # Load data
    if args.npy:
        viewer.load_npy(args.npy)
    elif args.segments:
        data = load_segments(args.segments, args.model)
        if data is None:
            return
        viewer.brain_data = data
        viewer.n_timesteps, viewer.n_vertices = data.shape
    else:
        print("Error: Specify --npy or --segments")
        return
    
    # Load video if specified
    if args.video:
        viewer.load_video(args.video)
    
    # Show visualization
    if args.animate:
        if args.video:
            viewer.view_with_video(interval=args.interval)
        else:
            viewer.view_animation(interval=args.interval)
    else:
        viewer.view_static(args.timestep)


if __name__ == "__main__":
    main()
