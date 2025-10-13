#!/bin/bash

# Check if a filename was provided
if [ -z "$1" ]; then
  echo "Usage: $0 <video_file>"
  exit 1
fi

input="$1"
output="${input}.wav"

# Extract audio as WAV
ffmpeg -i "$input" -vn -acodec pcm_s16le -ar 44100 -ac 2 "$output"
