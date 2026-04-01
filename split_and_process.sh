#!/bin/bash
# Split video into segments and process each with TribeV2
# Usage: ./split_and_process.sh <video_file> [segment_seconds]

set -e

INPUT_FILE="$1"
SEGMENT_SEC="${2:-30}"

if [ -z "$INPUT_FILE" ]; then
    echo "Usage: $0 <video_file> [segment_seconds]"
    echo "Example: $0 video.mp4 30"
    exit 1
fi

if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: File not found: $INPUT_FILE"
    exit 1
fi

INPUT_DIR=$(dirname "$INPUT_FILE")
INPUT_NAME=$(basename "$INPUT_FILE")
NAME_NO_EXT="${INPUT_NAME%.*}"

# Get video duration
DURATION=$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$INPUT_FILE" | cut -d. -f1)
echo "Video duration: ${DURATION}s"

if [ "$DURATION" -lt 60 ]; then
    echo "Video is short enough, processing directly..."
    VIDEOMODEL=vitl ./tribe.sh "$INPUT_FILE"
    exit 0
fi

# Create segments directory (use /tmp to avoid permission issues with existing cache)
SEGMENTS_DIR="/tmp/tribe_segments_${NAME_NO_EXT}"
rm -rf "$SEGMENTS_DIR"
mkdir -p "$SEGMENTS_DIR"

echo "Splitting into ${SEGMENT_SEC}s segments..."

# Split video (re-encode to fix codec issues at segment boundaries)
echo "Splitting and re-encoding into ${SEGMENT_SEC}s segments..."
ffmpeg -y -i "$INPUT_FILE" -c:v libx264 -preset fast -crf 23 \
    -c:a aac -b:a 128k -ar 48000 -ac 2 \
    -force_key_frames "expr:gte(t,n_forced*$SEGMENT_SEC)" \
    -f segment -segment_time "$SEGMENT_SEC" -reset_timestamps 1 \
    "$SEGMENTS_DIR/${NAME_NO_EXT}_%03d.mp4"

# Count segments
SEGMENT_COUNT=$(ls -1 "$SEGMENTS_DIR"/*.* 2>/dev/null | wc -l)
echo "Created $SEGMENT_COUNT segments"

# Process each segment (now .mp4 after re-encoding)
SEGMENT_NUM=0
for segment in "$SEGMENTS_DIR"/*.mp4; do
    SEGMENT_NUM=$((SEGMENT_NUM + 1))
    echo "=========================================="
    echo "Processing segment $SEGMENT_NUM of $SEGMENT_COUNT"
    echo "File: $segment"
    echo "=========================================="
    
    VIDEOMODEL=vitl ./tribe.sh "$segment"
done

echo "=========================================="
echo "All segments processed!"
echo "Segments directory: $SEGMENTS_DIR"
echo "=========================================="
