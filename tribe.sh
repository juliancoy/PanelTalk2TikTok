#!/bin/bash
# TribeV2 Docker wrapper script
# Usage: ./tribe.sh <input_file>
# Output: <input_basename>_tribe.npy

set -e  # Exit on error

# Check if input file is provided
if [ $# -eq 0 ]; then
    echo "Error: No input file specified"
    echo "Usage: $0 <input_file>"
    echo "Supported formats: .mp4, .avi, .mkv, .mov, .webm (video)"
    echo "                   .wav, .mp3, .flac, .ogg (audio)"
    echo "                   .txt (text)"
    exit 1
fi

INPUT_FILE="$1"

# Check if file exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "Error: Input file '$INPUT_FILE' not found"
    exit 1
fi

# Get absolute path of input file
INPUT_FILE_ABS=$(realpath "$INPUT_FILE")
INPUT_DIR=$(dirname "$INPUT_FILE_ABS")
INPUT_BASENAME=$(basename "$INPUT_FILE_ABS")
INPUT_NAME="${INPUT_BASENAME%.*}"
INPUT_EXT="${INPUT_BASENAME##*.}"

# Determine output filename
OUTPUT_BASENAME="${INPUT_NAME}_tribe.npy"
OUTPUT_FILE="$INPUT_DIR/$OUTPUT_BASENAME"

# Check if output file already exists
if [ -f "$OUTPUT_FILE" ]; then
    echo "Warning: Output file '$OUTPUT_FILE' already exists"
    read -p "Overwrite? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted"
        exit 1
    fi
fi

# Create cache directory if it doesn't exist
CACHE_DIR="$INPUT_DIR/.tribe_cache"
mkdir -p "$CACHE_DIR"

echo "=========================================="
echo "TribeV2 Processing"
echo "=========================================="
echo "Input file:  $INPUT_BASENAME"
echo "Input type:  $INPUT_EXT"
echo "Output file: $OUTPUT_BASENAME"
echo "Cache dir:   $CACHE_DIR"
echo "=========================================="

# Check if Docker image exists, build if not
if ! docker image inspect tribev2:latest >/dev/null 2>&1; then
    echo "Docker image 'tribev2:latest' not found. Building..."
    if [ -d "tribev2" ]; then
        cd tribev2
        docker build -t tribev2 .
        cd ..
    else
        echo "Error: 'tribev2' directory not found. Cannot build Docker image."
        exit 1
    fi
fi

# Run TribeV2 in Docker
echo "Running TribeV2 inference..."
docker run --rm --gpus all \
  -v "$CACHE_DIR:/root/.cache" \
  -v "$INPUT_DIR:/data" \
  tribev2 python -c "
from tribev2 import TribeModel
import numpy as np
import os

# Determine input type and process
input_file = '/data/$INPUT_BASENAME'
output_file = '/data/$OUTPUT_BASENAME'

print(f'Processing: {input_file}')
print(f'Output: {output_file}')

# Load model
model = TribeModel.from_pretrained(
    'facebook/tribev2',
    cache_folder='/root/.cache'
)

# Process based on file extension
ext = os.path.splitext(input_file)[1].lower()
if ext in ['.mp4', '.avi', '.mkv', '.mov', '.webm']:
    df = model.get_events_dataframe(video_path=input_file)
elif ext in ['.wav', '.mp3', '.flac', '.ogg']:
    df = model.get_events_dataframe(audio_path=input_file)
elif ext == '.txt':
    df = model.get_events_dataframe(text_path=input_file)
else:
    raise ValueError(f'Unsupported file type: {ext}')

# Run prediction
preds, segments = model.predict(df)

# Save results
np.save(output_file, preds)
print(f'Success! Predictions saved to {output_file}')
print(f'Shape: {preds.shape} (n_timesteps, n_vertices)')
print(f'Number of segments: {len(segments)}')
"

# Check if output was created
if [ -f "$OUTPUT_FILE" ]; then
    echo "=========================================="
    echo "Processing complete!"
    echo "Output saved to: $OUTPUT_FILE"
    
    # Show file info
    echo -n "Output size: "
    ls -lh "$OUTPUT_FILE" | awk '{print $5}'
    
    # Try to show array shape if it's a numpy file
    if command -v python3 >/dev/null 2>&1; then
        echo -n "Array shape: "
        python3 -c "import numpy as np; data = np.load('$OUTPUT_FILE'); print(data.shape)" 2>/dev/null || echo "Could not read numpy file"
    fi
else
    echo "Error: Output file was not created"
    exit 1
fi