#!/usr/bin/env python3
"""Inspect TribeV2 prediction results"""

import argparse
import numpy as np
from pathlib import Path

def inspect_npy(file_path):
    """Load and display info about a .npy file"""
    data = np.load(file_path)
    
    print(f"\n{'='*50}")
    print(f"File: {file_path}")
    print(f"{'='*50}")
    print(f"Shape: {data.shape}")
    print(f"Dtype: {data.dtype}")
    print(f"Size: {data.nbytes / 1024 / 1024:.2f} MB")
    print(f"\nData statistics:")
    print(f"  Mean: {data.mean():.4f}")
    print(f"  Std:  {data.std():.4f}")
    print(f"  Min:  {data.min():.4f}")
    print(f"  Max:  {data.max():.4f}")
    print(f"\nInterpretation:")
    print(f"  - {data.shape[0]} time steps (1 TR = 1 second)")
    print(f"  - {data.shape[1]} vertices on fsaverage5 cortical mesh")
    print(f"  - Total duration: ~{data.shape[0]} seconds")
    print(f"{'='*50}\n")
    
    return data

def main():
    parser = argparse.ArgumentParser(description='Inspect TribeV2 results')
    parser.add_argument('npy_file', help='Path to .npy file')
    parser.add_argument('--save-csv', action='store_true', 
                        help='Save as CSV for external analysis')
    parser.add_argument('--output', '-o', default=None,
                        help='Output CSV filename')
    args = parser.parse_args()
    
    data = inspect_npy(args.npy_file)
    
    if args.save_csv:
        output = args.output or args.npy_file.replace('.npy', '.csv')
        # Save as vertex x time for easier analysis
        np.savetxt(output, data.T, delimiter=',')
        print(f"Saved to: {output}")

if __name__ == '__main__':
    main()
