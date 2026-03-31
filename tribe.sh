#!/bin/bash
# TribeV2 Docker wrapper script
# Usage: ./tribe.sh <input_file> [batch_size]
# Output: <input_basename>_tribe.npy

set -e  # Exit on error

# Check if input file is provided
if [ $# -eq 0 ]; then
    echo "Error: No input file specified"
    echo "Usage: $0 <input_file> [batch_size]"
    echo "Supported formats: .mp4, .avi, .mkv, .mov, .webm (video)"
    echo "                   .wav, .mp3, .flac, .ogg (audio)"
    echo "                   .txt (text)"
    echo ""
    echo "Optional:"
    echo "  batch_size: Batch size for inference (default: 16)"
    exit 1
fi

INPUT_FILE="$1"
BATCH_SIZE="${2:-16}"  # Default batch size is 16

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
# Also check if we should force rebuild (if Dockerfile is newer than image)
FORCE_REBUILD=false
# Disabled force rebuild check to avoid unnecessary rebuilds
# if docker image inspect tribev2:latest >/dev/null 2>&1; then
#     # Check if Dockerfile is newer than image
#     if [ -f "tribev2/Dockerfile" ]; then
#         IMAGE_CREATED=$(docker image inspect tribev2:latest --format '{{.Created}}' | cut -d'T' -f1)
#         DOCKERFILE_MODIFIED=$(stat -c %Y tribev2/Dockerfile)
#         # Convert image created time to timestamp (approximate)
#         IMAGE_TIMESTAMP=$(date -d "$IMAGE_CREATED" +%s 2>/dev/null || echo 0)
#         if [ $DOCKERFILE_MODIFIED -gt $IMAGE_TIMESTAMP ]; then
#             echo "Dockerfile has been modified since image was built. Forcing rebuild..."
#             FORCE_REBUILD=true
#         fi
#     fi
# fi

if ! docker image inspect tribev2:latest >/dev/null 2>&1 || [ "$FORCE_REBUILD" = true ]; then
    echo "Building Docker image 'tribev2:latest'..."
    if [ -d "tribev2" ]; then
        cd tribev2
        if [ "$FORCE_REBUILD" = true ]; then
            docker build --no-cache -t tribev2 .
        else
            docker build -t tribev2 .
        fi
        cd ..
    else
        echo "Error: 'tribev2' directory not found. Cannot build Docker image."
        exit 1
    fi
fi

# Run TribeV2 in Docker
echo "Running TribeV2 inference..."
echo "Using batch size: $BATCH_SIZE"
docker run --rm --gpus all \
  -e HF_TOKEN="$(cat hftoken.txt)" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$CACHE_DIR:/root/.cache" \
  -v "$INPUT_DIR:/data" \
  -v "$(pwd)/hftoken.txt:/root/hftoken.txt" \
  tribev2 python -c "
from tribev2 import TribeModel
import numpy as np
import os

# Determine input type and process
input_file = '/data/$INPUT_BASENAME'
output_file = '/data/$OUTPUT_BASENAME'

print(f'Processing: {input_file}')
print(f'Output: {output_file}')

# Load model with memory-optimized batch size
import torch

def get_optimal_batch_size(user_batch_size=$BATCH_SIZE):
    \"\"\"Calculate optimal batch size based on available GPU memory.\"\"\"
    if not torch.cuda.is_available():
        return user_batch_size
    
    # Clear cache first
    torch.cuda.empty_cache()
    
    # Get GPU memory info
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    free_memory = total_memory - allocated - reserved
    
    print(f'GPU Memory - Total: {total_memory / 1e9:.2f} GB, '
          f'Allocated: {allocated / 1e9:.2f} GB, '
          f'Reserved: {reserved / 1e9:.2f} GB, '
          f'Free: {free_memory / 1e9:.2f} GB')
    
    # If user specified a batch size, use it (but warn if too high)
    if user_batch_size > 0:
        # Estimate memory needed per batch item (approximate)
        # Based on error message: 1.76 GiB for video processing
        estimated_per_item = 1.76 * 1024**3  # 1.76 GiB in bytes
        
        max_batch_by_memory = int(free_memory / estimated_per_item)
        if user_batch_size > max_batch_by_memory:
            print(f'Warning: Requested batch size {user_batch_size} may exceed available memory.')
            print(f'Maximum recommended batch size: {max(1, max_batch_by_memory)}')
            # Auto-reduce if user batch size is too high
            return max(1, max_batch_by_memory)
        return user_batch_size
    
    # Auto-calculate batch size based on free memory
    # Conservative estimate: leave 20% free for other operations
    safe_free_memory = free_memory * 0.8
    
    # Estimate memory per batch item (varies by input type)
    # Video processing uses more memory than audio/text
    ext = os.path.splitext(input_file)[1].lower()
    if ext in ['.mp4', '.avi', '.mkv', '.mov', '.webm']:
        # Video processing: ~1.76 GiB per item based on error
        per_item_memory = 1.76 * 1024**3
    elif ext in ['.wav', '.mp3', '.flac', '.ogg']:
        # Audio processing: less memory than video
        per_item_memory = 0.5 * 1024**3
    else:  # .txt or other
        # Text processing: least memory
        per_item_memory = 0.1 * 1024**3
    
    calculated_batch = int(safe_free_memory / per_item_memory)
    
    # Apply bounds: minimum 1, maximum 64 (original default)
    calculated_batch = max(1, min(calculated_batch, 64))
    
    print(f'Auto-calculated batch size: {calculated_batch} '
          f'(based on {safe_free_memory / 1e9:.2f} GB available memory)')
    return calculated_batch

# Get optimal batch size
optimal_batch_size = get_optimal_batch_size()
print(f'Using batch size: {optimal_batch_size}')

model = TribeModel.from_pretrained(
    'facebook/tribev2',
    cache_folder='/root/.cache',
    config_update={
        'data.batch_size': optimal_batch_size
    }
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