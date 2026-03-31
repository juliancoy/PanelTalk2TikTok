#!/usr/bin/env python3
"""Visualize TribeV2 brain predictions"""

import argparse
import numpy as np
from pathlib import Path
import sys

# Add tribev2 to path if running from repo
sys.path.insert(0, str(Path(__file__).parent / 'tribev2'))

def visualize_predictions(npy_file, n_timesteps=10, output=None, show_video=False):
    """Create brain visualization from predictions"""
    from tribev2.plotting import PlotBrain
    
    # Load predictions
    preds = np.load(npy_file)
    print(f"Loaded predictions: {preds.shape}")
    
    # Create plotter
    plotter = PlotBrain(mesh="fsaverage5")
    
    # Limit timesteps if needed
    n_timesteps = min(n_timesteps, preds.shape[0])
    
    print(f"Creating visualization for first {n_timesteps} timesteps...")
    
    # Create figure
    fig = plotter.plot_timesteps(
        preds[:n_timesteps],
        segments=None,  # No segment info available from .npy alone
        cmap="fire",
        norm_percentile=99,
        show_stimuli=show_video,
    )
    
    # Save or show
    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to: {output}")
    else:
        import matplotlib.pyplot as plt
        plt.show()
    
    return fig

def plot_single_timestep(npy_file, timestep=0, view="lateral", output=None):
    """Plot a single timestep"""
    from tribev2.plotting import PlotBrain
    import matplotlib.pyplot as plt
    
    preds = np.load(npy_file)
    
    if timestep >= preds.shape[0]:
        print(f"Error: Timestep {timestep} exceeds available {preds.shape[0]} timesteps")
        return
    
    plotter = PlotBrain(mesh="fsaverage5")
    
    # Get data for this timestep
    data = preds[timestep]
    
    # Create figure with 2 views (left and right hemisphere)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    plotter.plot(data, view="left", axes=axes[0])
    axes[0].set_title(f"Left - Timestep {timestep}")
    
    plotter.plot(data, view="right", axes=axes[1])
    axes[1].set_title(f"Right - Timestep {timestep}")
    
    plt.tight_layout()
    
    if output:
        fig.savefig(output, dpi=150, bbox_inches='tight')
        print(f"Saved to: {output}")
    else:
        plt.show()
    
    return fig

def create_video(npy_file, output_mp4, fps=2):
    """Create video animation of brain activity"""
    from tribev2.plotting import PlotBrain
    
    preds = np.load(npy_file)
    plotter = PlotBrain(mesh="fsaverage5")
    
    print(f"Creating video with {preds.shape[0]} frames at {fps} fps...")
    
    fig = plotter.plot_timesteps_mp4(
        preds,
        output_path=output_mp4,
        fps=fps,
        cmap="fire",
        norm_percentile=99,
    )
    
    print(f"Saved video to: {output_mp4}")
    return fig

def main():
    parser = argparse.ArgumentParser(description='Visualize TribeV2 brain predictions')
    parser.add_argument('npy_file', help='Path to _tribe.npy file')
    parser.add_argument('--timesteps', '-t', type=int, default=10,
                        help='Number of timesteps to visualize (default: 10)')
    parser.add_argument('--single', '-s', type=int, default=None,
                        help='Plot single timestep (0-indexed)')
    parser.add_argument('--output', '-o', default=None,
                        help='Output filename (PNG or MP4)')
    parser.add_argument('--video', action='store_true',
                        help='Create video animation (requires .mp4 output)')
    parser.add_argument('--fps', type=int, default=2,
                        help='Video frame rate (default: 2)')
    
    args = parser.parse_args()
    
    if args.video:
        output = args.output or args.npy_file.replace('.npy', '_brain.mp4')
        create_video(args.npy_file, output, args.fps)
    elif args.single is not None:
        output = args.output or args.npy_file.replace('.npy', f'_t{args.single}.png')
        plot_single_timestep(args.npy_file, args.single, output=output)
    else:
        output = args.output or args.npy_file.replace('.npy', '.png')
        visualize_predictions(args.npy_file, args.timesteps, output)

if __name__ == '__main__':
    main()
