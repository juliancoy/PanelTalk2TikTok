#!/bin/bash

# First argument is the input folder
INPUT_FOLDER=$1

# Extract the folder name only (basename)
FOLDER_NAME=$(basename "$INPUT_FOLDER")

# Run ffmpeg with dynamic input/output
ffmpeg -framerate 30 -i "$INPUT_FOLDER"/%05d.jpg \
  -c:v h264_nvenc -preset fast -b:v 4M -pix_fmt yuv420p "${FOLDER_NAME}.mp4"
