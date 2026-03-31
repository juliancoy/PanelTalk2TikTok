#!/usr/bin/env python3
"""
Visualize TribeV2 brain predictions on cortical surface.
This is the intended visualization method.
"""

import argparse
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'tribev2'))

from tribev2.plotting import PlotBrain


def visualize_surface(npy_file, timesteps=10, output='brain_viz.png'):
    """
    Create the standard TribeV2 visualization:
    - Multiple timesteps side by side
    - Cortical surface on fsaverage5 mesh
    - Fire colormap (black=low, red=med, yellow/white=high)
    """
    # Load predictions
    preds = np.load(npy_file)
    print(f"Loaded: {preds.shape} - {preds.shape[0]}s video, {preds.shape[1]} vertices")
    
    # Create plotter on fsaverage5 surface
    plotter = PlotBrain(
        mesh="fsaverage5",      # Standard brain mesh (20k vertices)
        inflate="half",         # Half-inflated surface (good visibility)
        bg_map="sulcal",        # Show sulcal/gyral patterns
    )
    
    # Plot first N timesteps
    n = min(timesteps, preds.shape[0])
    print(f"Creating visualization for timesteps 0-{n-1}...")
    
    fig = plotter.plot_timesteps(
        preds[:n],
        segments=None,          # No segment info available from .npy alone
        views="left",           # Left hemisphere view (best for language/visual)
        cmap="fire",            # Standard TribeV2 colormap
        norm_percentile=99,     # Normalize to 99th percentile
        show_stimuli=False,     # Set True if you have segments with video/audio
    )
    
    fig.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output}")
    return fig


def visualize_single(npy_file, timestep=0, view="left", output='brain_single.png'):
    """
    Plot a single timestep with multiple views (left, right, dorsal, etc.)
    """
    preds = np.load(npy_file)
    
    plotter = PlotBrain(mesh="fsaverage5", inflate="half")
    
    import matplotlib.pyplot as plt
    
    # Create figure with 4 views
    views = ["left", "right", "dorsal", "ventral"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for ax, view_name in zip(axes, views):
        plotter.plot_surf(
            preds[timestep],
            axes=[ax],
            views=view_name,
            cmap="fire",
            norm_percentile=99,
        )
        ax.set_title(f"{view_name.capitalize()} - t={timestep}s")
    
    plt.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches='tight')
    print(f"Saved to: {output}")
    return fig


def create_brain_video(npy_file, output='brain_video.mp4', fps=2):
    """
    Create video animation of brain activity over time
    """
    preds = np.load(npy_file)
    plotter = PlotBrain(mesh="fsaverage5", inflate="half")
    
    print(f"Creating {preds.shape[0]} frame video at {fps} fps...")
    
    plotter.plot_timesteps_mp4(
        preds,
        filepath=output,
        fps=fps,
        cmap="fire",
        norm_percentile=99,
        views="left",
    )
    
    print(f"Saved video to: {output}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize TribeV2 brain predictions on cortical surface'
    )
    parser.add_argument('npy_file', help='Path to _tribe.npy file')
    parser.add_argument('--mode', '-m', choices=['multi', 'single', 'video'],
                        default='multi',
                        help='Visualization mode (default: multi)')
    parser.add_argument('--timesteps', '-t', type=int, default=10,
                        help='Number of timesteps for multi mode (default: 10)')
    parser.add_argument('--timestep', '-s', type=int, default=0,
                        help='Single timestep to plot (default: 0)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output filename')
    parser.add_argument('--fps', type=int, default=2,
                        help='Video frame rate (default: 2)')
    
    args = parser.parse_args()
    
    if args.mode == 'multi':
        output = args.output or args.npy_file.replace('.npy', '_multi.png')
        visualize_surface(args.npy_file, args.timesteps, output)
    
    elif args.mode == 'single':
        output = args.output or args.npy_file.replace('.npy', f'_t{args.timestep}.png')
        visualize_single(args.npy_file, args.timestep, output=output)
    
    elif args.mode == 'video':
        output = args.output or args.npy_file.replace('.npy', '.mp4')
        create_brain_video(args.npy_file, output, args.fps)


if __name__ == '__main__':
    main()
