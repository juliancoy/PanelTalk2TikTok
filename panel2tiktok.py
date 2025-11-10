#!/usr/bin/env python3
"""
Panel2TikTok - Video processing wrapper for istalking.py

Takes a single video file as input and processes it using the
istalking.py lip movement analysis functionality.
"""

import argparse
import sys
import os
import subprocess
from types import SimpleNamespace


# Import the analyze_video function from istalking.py
from istalking import analyze_video
from transcriptJson2csv import export_words_to_csv
from extract_segments import direct_video

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
        """,
    )

    parser.add_argument("video", help="Input video file path")

    parser.add_argument(
        "--output", "-o", help="Output JSON file path (default: <video>.json)"
    )

    parser.add_argument(
        "--show", action="store_true", help="Show visualization during processing"
    )


    parser.add_argument(
        "--deblur", action="store_true", help="If the image needs to be refocused"
    )

    parser.add_argument(
        "--viz-inner-lip", action="store_true", help="Visualize inner lip landmarks"
    )
    parser.add_argument("--splits", type=int, default=1,
                    help="Number of vertical splits to divide each frame into for separate processing (default: 1)")
    args = parser.parse_args()

    deblurredvideo = args.video[:-4]+"_deblur.mp4"
    # Validate input file exists
    if not os.path.exists(deblurredvideo):
        print(f"Error: Video file not found: {deblurredvideo}")
        sys.exit(1)

    print(f"Processing video: {args.video}")
    if args.output:
        speakingtimesfilename = args.output
    else:
        speakingtimesfilename = f"{deblurredvideo}.json"

    if not os.path.exists(speakingtimesfilename):
        # Call analyze_video directly with the parsed arguments
        result_path = analyze_video(
            video_path=deblurredvideo,
            output_path=speakingtimesfilename,
            show=args.show,
            viz_inner_lip=args.viz_inner_lip,
            detection_confidence=0.18,
            tracking_confidence=0.18,
            max_faces=3,
            crop_percent=0.20,
            splits=args.splits
        )
    else:
        print(f"{speakingtimesfilename} already exists")

    audiofile = args.video+".wav"
    audiotranscript = args.video+ ".json"

    if os.path.exists(audiofile) and not os.path.exists(audiotranscript):
        print(f"{audiotranscript} does not exist. Computing transcript")
        # Run whisperx.sh and stream its stdout/stderr so the output is visible in real time
        # Uses subprocess.Popen to capture and forward lines as they arrive.
        proc = subprocess.Popen(
            ["bash", "./whisperx.sh", audiofile],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        # Stream output lines to this process's stdout
        if proc.stdout is not None:
            for line in proc.stdout:
                print(line, end="")
        proc.wait()
        if proc.returncode != 0:
            print(f"whisperx.sh exited with code {proc.returncode}")

    csv_file = audiotranscript + ".csv"
    if not os.path.exists(csv_file):
        export_words_to_csv(audiotranscript)
    finalvideo = deblurredvideo + "_final.mp4"
    if not os.path.exists(finalvideo):
        args = SimpleNamespace(
            csv=csv_file,
            video=deblurredvideo,
            audio=audiofile,
            visualize = args.show,
            width=480,
            height=853,
            cycle_after=8,
            font_scale=4.0,
            min_stay_duration=4.0,
            no_video=False,
            no_audio=False,
            no_combine=False,
            save_lengths=False,
            fade_samples=300,
            fade_frames=6,
            text_height=300,
            shift_seconds=0.15,
            zoom_speed=0.005,
            injections=[{"start":77.927, "end":91.647, "filename":"./victoria_nobg.png", "image":None}],
            speaking_times=speakingtimesfilename
        )

        direct_video(args)
if __name__ == "__main__":
    main()
