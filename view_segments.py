#!/usr/bin/env python3
"""
view_segments.py - View and combine segmented TribeV2 outputs

Usage:
    python view_segments.py <segments_directory>

Example:
    python view_segments.py /tmp/tribe_segments_P1090551_panthers_2832842
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_segments(segments_dir: str, model: str = "vitg"):
    """Load all segment .npy files and combine them."""
    seg_dir = Path(segments_dir)
    
    # Find all tribe npy files
    pattern = f"*_tribe_{model}.npy"
    files = sorted(seg_dir.glob(pattern))
    
    if not files:
        print(f"No files found matching {pattern} in {segments_dir}")
        return None, []
    
    print(f"Found {len(files)} segments:")
    all_data = []
    for f in files:
        data = np.load(f)
        print(f"  {f.name}: {data.shape}")
        all_data.append(data)
    
    # Concatenate along time dimension
    combined = np.vstack(all_data)
    print(f"\nCombined shape: {combined.shape}")
    print(f"Data range: [{combined.min():.3f}, {combined.max():.3f}]")
    
    return combined, files


def plot_timeseries(data: np.ndarray, num_vertices: int = 5, output_path: str = None):
    """Plot BOLD response timeseries for selected vertices."""
    # Select vertices with highest variance
    variances = np.var(data, axis=0)
    top_vertices = np.argsort(variances)[-num_vertices:][::-1]
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    for v in top_vertices:
        ax.plot(data[:, v], label=f'Vertex {v}', alpha=0.8, linewidth=1.5)
    
    ax.set_xlabel('Timestep (2Hz)', fontsize=12)
    ax.set_ylabel('BOLD Response', fontsize=12)
    ax.set_title(f'BOLD Response Timeseries (Top {num_vertices} Most Active Vertices)', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_heatmap(data: np.ndarray, output_path: str = None):
    """Plot heatmap of all vertices over time."""
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Downsample if too many vertices
    if data.shape[1] > 500:
        step = data.shape[1] // 500
        data_plot = data[:, ::step]
        print(f"Downsampled vertices {data.shape[1]} → {data_plot.shape[1]} for display")
    else:
        data_plot = data
    
    im = ax.imshow(data_plot.T, aspect='auto', cmap='hot', interpolation='nearest')
    ax.set_xlabel('Timestep (2Hz)', fontsize=12)
    ax.set_ylabel('Vertex (downsampled)', fontsize=12)
    ax.set_title('Brain Activity Heatmap', fontsize=14)
    
    cbar = plt.colorbar(im, ax=ax, label='BOLD Response')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_segment_boundaries(data_list: list, output_path: str = None):
    """Plot all segments with boundaries marked."""
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Use mean across all vertices for each timestep
    offset = 0
    colors = plt.cm.tab10(np.linspace(0, 1, len(data_list)))
    
    for i, data in enumerate(data_list):
        mean_activity = np.mean(data, axis=1)
        times = np.arange(len(mean_activity)) + offset
        
        ax.plot(times, mean_activity, color=colors[i], linewidth=1.5, 
                label=f'Segment {i+1} ({len(mean_activity)} frames)')
        
        # Mark boundary
        if i < len(data_list) - 1:
            ax.axvline(x=offset + len(mean_activity), color='white', 
                      linestyle='--', linewidth=2, alpha=0.7)
        
        offset += len(mean_activity)
    
    ax.set_xlabel('Total Timesteps (2Hz)', fontsize=12)
    ax.set_ylabel('Mean BOLD Response', fontsize=12)
    ax.set_title('Brain Activity Across All Segments', fontsize=14)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='View TribeV2 segmented outputs')
    parser.add_argument('directory', help='Directory containing segment .npy files')
    parser.add_argument('--model', default='vitg', help='Model suffix (default: vitg)')
    parser.add_argument('--output-dir', '-o', help='Directory to save plots')
    parser.add_argument('--vertices', type=int, default=5, help='Number of vertices to plot')
    parser.add_argument('--show', action='store_true', help='Show plots interactively')
    
    args = parser.parse_args()
    
    # Load data
    combined, files = load_segments(args.directory, args.model)
    if combined is None:
        return
    
    # Setup output directory
    out_dir = Path(args.output_dir) if args.output_dir else Path(args.directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\nGenerating plots...")
    
    # 1. Combined timeseries
    plot_timeseries(combined, num_vertices=args.vertices, 
                   output_path=out_dir / f'combined_timeseries_{args.model}.png')
    
    # 2. Heatmap
    plot_heatmap(combined, 
                output_path=out_dir / f'combined_heatmap_{args.model}.png')
    
    # 3. Segment boundaries (load individual files)
    data_list = [np.load(f) for f in files]
    plot_segment_boundaries(data_list,
                           output_path=out_dir / f'segment_boundaries_{args.model}.png')
    
    # Save combined data
    combined_path = out_dir / f'combined_{args.model}.npy'
    np.save(combined_path, combined)
    print(f"\nSaved combined data: {combined_path}")
    print(f"Final shape: {combined.shape}")
    
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
