#!/usr/bin/env python3
"""
Panel2TikTok - Video processing wrapper for istalking.py

Takes a single video file as input and processes it using the
istalking.py lip movement analysis functionality.
"""

import argparse
import sys
import os

# Import the analyze_video function from istalking.py
from istalking import analyze_video


def main():
    """Main function for panel2tiktok.py"""
    parser = argparse.ArgumentParser(
        description="Panel2TikTok - Process video for speaking detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python panel2tiktok.py input_video.mp4
  python panel2tiktok.py input_video.mp4 --output results.json
  python panel2tiktok.py input_video.mp4 --show --viz-inner-lip
        """
    )
    
    parser.add_argument(
        "video", 
        help="Input video file path"
    )
    
    parser.add_argument(
        "--output", "-o", 
        help="Output JSON file path (default: <video>.json)"
    )
    
    parser.add_argument(
        "--show", 
        action="store_true",
        help="Show visualization during processing"
    )
    
    parser.add_argument(
        "--viz-inner-lip", 
        action="store_true",
        help="Visualize inner lip landmarks"
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        sys.exit(1)
    
    print(f"Processing video: {args.video}")
    if args.output:
        print(f"Output will be saved to: {args.output}")
    else:
        print(f"Output will be saved to: {args.video}.json")
    
    # Call analyze_video directly with the parsed arguments
    result_path = analyze_video(
        video_path=args.video,
        output_path=args.output,
        show=args.show,
        viz_inner_lip=args.viz_inner_lip,
        detection_confidence=0.09,
        tracking_confidence=0.09,
        max_faces=3
    )


if __name__ == "__main__":
    main()
