#!/usr/bin/env python3
"""
visualize_brain.py - Simple visualization for TribeV2 brain predictions

Usage:
    python visualize_brain.py <npy_file> [--video <video_file>]

Example:
    python visualize_brain.py CodeCollective/swc_preliminary_tribe.npy
    python visualize_brain.py CodeCollective/swc_preliminary_tribe.npy --video CodeCollective/swc_preliminary.mp4
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path


def load_brain_data(npy_path: str):
    """Load brain prediction data from .npy file."""
    data = np.load(npy_path)
    print(f"[BrainViz] Loaded: {npy_path}")
    print(f"[BrainViz] Shape: {data.shape} (timesteps × vertices)")
    print(f"[BrainViz] Data range: [{data.min():.3f}, {data.max():.3f}]")
    return data


def plot_timeseries(data: np.ndarray, num_vertices: int = 10):
    """Plot BOLD response timeseries for selected vertices."""
    n_timesteps, n_vertices = data.shape
    
    # Select vertices with highest variance (most active)
    variances = np.var(data, axis=0)
    top_vertices = np.argsort(variances)[-num_vertices:]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, v in enumerate(top_vertices):
        ax.plot(data[:, v], label=f'Vertex {v} (var={variances[v]:.3f})', alpha=0.7)
    
    ax.set_xlabel('Timestep')
    ax.set_ylabel('BOLD Response')
    ax.set_title(f'BOLD Response Timeseries (Top {num_vertices} Most Active Vertices)')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('brain_timeseries.png', dpi=150)
    print(f"[BrainViz] Saved: brain_timeseries.png")
    plt.show()


def plot_heatmap(data: np.ndarray):
    """Plot heatmap of all vertices over time."""
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Downsample if too large
    if data.shape[1] > 1000:
        step = data.shape[1] // 1000
        data_plot = data[:, ::step]
        print(f"[BrainViz] Downsampled vertices {data.shape[1]} → {data_plot.shape[1]} for display")
    else:
        data_plot = data
    
    im = ax.imshow(data_plot.T, aspect='auto', cmap='hot', interpolation='bilinear')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Vertex (downsampled)')
    ax.set_title('Brain Activity Heatmap')
    plt.colorbar(im, ax=ax, label='BOLD Response')
    
    plt.tight_layout()
    plt.savefig('brain_heatmap.png', dpi=150)
    print(f"[BrainViz] Saved: brain_heatmap.png")
    plt.show()


def plot_distribution(data: np.ndarray):
    """Plot distribution of BOLD values."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Histogram of all values
    axes[0].hist(data.flatten(), bins=100, color='blue', alpha=0.7)
    axes[0].set_xlabel('BOLD Response')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of All Values')
    axes[0].grid(True, alpha=0.3)
    
    # Mean activity per vertex
    mean_per_vertex = np.mean(data, axis=0)
    axes[1].hist(mean_per_vertex, bins=50, color='green', alpha=0.7)
    axes[1].set_xlabel('Mean BOLD Response')
    axes[1].set_ylabel('Number of Vertices')
    axes[1].set_title('Mean Activity per Vertex')
    axes[1].grid(True, alpha=0.3)
    
    # Variance per vertex
    var_per_vertex = np.var(data, axis=0)
    axes[2].hist(var_per_vertex, bins=50, color='red', alpha=0.7)
    axes[2].set_xlabel('Variance')
    axes[2].set_ylabel('Number of Vertices')
    axes[2].set_title('Activity Variance per Vertex')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('brain_distribution.png', dpi=150)
    print(f"[BrainViz] Saved: brain_distribution.png")
    plt.show()


def animate_with_video(data: np.ndarray, video_path: str, output_path: str = None):
    """Create side-by-side video and brain visualization."""
    try:
        import cv2
    except ImportError:
        print("[BrainViz] OpenCV (cv2) not installed. Install with: pip install opencv-python")
        return
    
    # Load video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[BrainViz] Error: Could not open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[BrainViz] Video: {frame_count} frames @ {fps:.1f} fps, {width}x{height}")
    
    # Create figure for brain activity
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Left: Video frame placeholder
    ax1.set_title('Video')
    ax1.axis('off')
    im1 = ax1.imshow(np.zeros((height, width, 3), dtype=np.uint8))
    
    # Right: Brain activity plot
    n_timesteps = data.shape[0]
    time_per_frame = frame_count / n_timesteps
    
    # Show mean activity across all vertices
    mean_activity = np.mean(data, axis=1)
    ax2.plot(mean_activity, color='blue', linewidth=2)
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Mean BOLD Response')
    ax2.set_title('Brain Activity (Mean across all vertices)')
    ax2.grid(True, alpha=0.3)
    
    # Add vertical line for current position
    vline = ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    
    plt.tight_layout()
    
    # Animation function
    def update(frame_idx):
        ret, frame = cap.read()
        if not ret:
            return im1, vline
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im1.set_array(frame_rgb)
        
        # Update brain activity indicator
        brain_idx = int((frame_idx / frame_count) * n_timesteps)
        brain_idx = min(brain_idx, n_timesteps - 1)
        vline.set_xdata([brain_idx, brain_idx])
        
        return im1, vline
    
    anim = FuncAnimation(fig, update, frames=frame_count, interval=1000/fps, blit=True)
    
    if output_path:
        anim.save(output_path, fps=fps, dpi=100)
        print(f"[BrainViz] Saved animation: {output_path}")
    else:
        plt.show()
    
    cap.release()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize TribeV2 brain predictions')
    parser.add_argument('npy', help='Path to _tribe.npy file')
    parser.add_argument('--video', '-v', help='Optional: path to video file for side-by-side')
    parser.add_argument('--output', '-o', help='Output path for animation (e.g., output.mp4)')
    parser.add_argument('--vertices', type=int, default=10, help='Number of vertices to plot (default: 10)')
    parser.add_argument('--heatmap', action='store_true', help='Show heatmap')
    parser.add_argument('--distribution', action='store_true', help='Show distribution plots')
    parser.add_argument('--all', action='store_true', help='Show all plots')
    
    args = parser.parse_args()
    
    # Load data
    data = load_brain_data(args.npy)
    
    # Show plots
    if args.all or (not args.heatmap and not args.distribution and not args.video):
        plot_timeseries(data, num_vertices=args.vertices)
    
    if args.all or args.heatmap:
        plot_heatmap(data)
    
    if args.all or args.distribution:
        plot_distribution(data)
    
    # Animate with video if provided
    if args.video:
        animate_with_video(data, args.video, args.output)


if __name__ == "__main__":
    main()
