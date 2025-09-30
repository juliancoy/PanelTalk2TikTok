#!/usr/bin/env python3
import sys
import subprocess
import os

def trim_wav(input_file, start_time, length):
    if not input_file.lower().endswith(".wav"):
        print("Error: input must be a .wav file")
        sys.exit(1)

    base, ext = os.path.splitext(input_file)
    output_file = f"{base}_trunc{ext}"

    # ffmpeg command with start and duration
    cmd = [
        "ffmpeg",
        "-i", input_file,
        "-ss", start_time,   # Start time (HH:MM:SS.xxx)
        "-t", length,        # Duration (HH:MM:SS.xxx)
        output_file
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"Trimmed file written to: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error running ffmpeg: {e}")
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} input.wav start length")
        print("Example: ./trim_wav.py song.wav 0:00:03 0:00:10.5")
        sys.exit(1)

    input_file = sys.argv[1]
    start_time = sys.argv[2]   # format: H:M:S[.fraction]
    length = sys.argv[3]       # format: H:M:S[.fraction]

    trim_wav(input_file, start_time, length)
