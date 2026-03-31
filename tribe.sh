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
    echo "  batch_size: Batch size for inference (default: auto)"
    exit 1
fi

INPUT_FILE="$1"
BATCH_SIZE="${2:-0}"  # Default 0 means auto-calculate

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

# Check if output file already exists and is up to date
if [ -f "$OUTPUT_FILE" ]; then
    if [ "$OUTPUT_FILE" -nt "$INPUT_FILE" ]; then
        echo "Output file '$OUTPUT_FILE' already exists and is up to date."
        echo "To reprocess, delete the output file and run again."
        echo "=========================================="
        echo "Output saved to: $OUTPUT_FILE"
        echo -n "Output size: "
        ls -lh "$OUTPUT_FILE" | awk '{print $5}'
        if command -v python3 >/dev/null 2>&1; then
            echo -n "Array shape: "
            python3 -c "import numpy as np; data = np.load('$OUTPUT_FILE'); print(data.shape)" 2>/dev/null || echo "Could not read numpy file"
        fi
        exit 0
    else
        echo "Warning: Output file '$OUTPUT_FILE' exists but is older than input."
        read -p "Overwrite? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Aborted"
            exit 1
        fi
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
echo "Intermediate files (cached alongside source):"
echo "  - Extracted audio: ${INPUT_NAME}.wav (if video)"
echo "  - Transcript:      ${INPUT_NAME}.tsv"
echo "  - Features:        $CACHE_DIR/"
echo "=========================================="

# Check if Docker image exists, build if not
if ! docker image inspect tribev2:latest >/dev/null 2>&1; then
    echo "Building Docker image 'tribev2:latest'..."
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
if [ "$BATCH_SIZE" -gt 0 ]; then
    echo "User-specified batch size: $BATCH_SIZE"
else
    echo "Batch size: auto (memory-based calculation)"
fi

docker run --rm --gpus all \
  -e HF_TOKEN="$(cat hftoken.txt)" \
  -e PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True,max_split_size_mb:512" \
  -e SPACY_CACHE_DIR="/root/.cache/spacy" \
  -v "$CACHE_DIR:/root/.cache" \
  -v "$INPUT_DIR:/data" \
  -v "$(pwd)/hftoken.txt:/root/hftoken.txt" \
  tribev2 python -c "
from tribev2 import TribeModel
import numpy as np
import os
import gc

# Setup spaCy to use cache directory
spacy_cache = '/root/.cache/spacy'
os.makedirs(spacy_cache, exist_ok=True)
os.environ['SPACY_CACHE_DIR'] = spacy_cache

# Check if spaCy model exists in cache, link it if needed
import spacy
spacy_model_name = 'en_core_web_lg'
try:
    nlp = spacy.load(spacy_model_name)
    print(f'[Cache] spaCy model {spacy_model_name} is available')
except OSError:
    # Model not found in standard location, check cache
    import subprocess
    cache_model_path = os.path.join(spacy_cache, spacy_model_name)
    if os.path.exists(cache_model_path):
        print(f'[Cache] Linking spaCy model from {cache_model_path}')
        # Link the cached model to spacy's expected location
        import spacy.util
        spacy.util.set_data_path(spacy_cache)
        nlp = spacy.load(spacy_model_name)
    else:
        print(f'[Cache] Downloading spaCy model {spacy_model_name} to cache...')
        subprocess.run([
            'python', '-m', 'spacy', 'download', 
            spacy_model_name
        ], check=True, capture_output=True)
        # Copy to cache for persistence
        import shutil
        model_path = spacy.util.get_package_path(spacy_model_name)
        if model_path and os.path.exists(model_path):
            shutil.copytree(model_path, cache_model_path, dirs_exist_ok=True)
        print(f'[Cache] spaCy model cached to {cache_model_path}')
        nlp = spacy.load(spacy_model_name)

# Determine input type and process
input_file = '/data/$INPUT_BASENAME'
output_file = '/data/$OUTPUT_BASENAME'
input_name = os.path.splitext(input_file)[0]

print(f'Processing: {input_file}')
print(f'Output: {output_file}')

# Check for existing intermediate files
ext = os.path.splitext(input_file)[1].lower()
if ext in ['.mp4', '.avi', '.mkv', '.mov', '.webm']:
    audio_file = input_name + '.wav'
    if os.path.exists(audio_file):
        print(f'[Cache] Found existing audio: {audio_file}')
if ext in ['.mp4', '.avi', '.mkv', '.mov', '.webm', '.wav', '.mp3', '.flac', '.ogg']:
    tsv_file = input_name + '.tsv'
    if os.path.exists(tsv_file):
        print(f'[Cache] Found existing transcript: {tsv_file}')

# Load model with memory-optimized batch size
import torch

def get_optimal_batch_size(user_batch_size=$BATCH_SIZE):
    \"\"\"Calculate optimal batch size based on available GPU memory.\"\"\"
    if not torch.cuda.is_available():
        return user_batch_size if user_batch_size > 0 else 4
    
    # Clear cache first
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    # Get GPU memory info
    total_memory = torch.cuda.get_device_properties(0).total_memory
    
    # Get more accurate free memory measurement
    allocated = torch.cuda.memory_allocated(0)
    reserved = torch.cuda.memory_reserved(0)
    
    # Use nvidia-smi for more accurate free memory (not cached by PyTorch)
    import subprocess
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free', '--format=csv,nounits,noheader'],
            capture_output=True, text=True
        )
        free_mb = int(result.stdout.strip().split('\n')[0])
        free_memory = free_mb * 1024 * 1024  # Convert MB to bytes
    except:
        # Fallback to PyTorch measurement
        free_memory = total_memory - allocated
    
    # Reserve memory for video feature extraction (happens before batch processing)
    # Video extraction needs ~10-11GB for 1080p video with vitg model
    video_extraction_reserve = 10.5 * 1024**3  # 10.5 GB reservation
    
    # Additional safety margin for fragmentation and other operations
    safety_margin = 1 * 1024**3  # 1 GB
    
    # Memory available for batch processing after extraction
    available_for_batches = max(0, free_memory - video_extraction_reserve - safety_margin)
    
    print(f'GPU Memory - Total: {total_memory / 1e9:.2f} GB')
    print(f'           Free: {free_memory / 1e9:.2f} GB')
    print(f'           After video extraction reserve: {available_for_batches / 1e9:.2f} GB')
    
    # Determine input type for memory estimation
    ext = os.path.splitext(input_file)[1].lower()
    is_video = ext in ['.mp4', '.avi', '.mkv', '.mov', '.webm']
    is_audio = ext in ['.wav', '.mp3', '.flac', '.ogg']
    
    # Memory per batch item during model inference
    # Based on model architecture and fsaverage5 output size (~20k vertices)
    if is_video:
        # Video features with vitg model (1408 dims)
        per_item_memory = 1.2 * 1024**3  # ~1.2 GB per item (vitg uses more memory)
    elif is_audio:
        # Audio + text features
        per_item_memory = 0.4 * 1024**3  # ~0.4 GB per item
    else:
        # Text only
        per_item_memory = 0.1 * 1024**3  # ~0.1 GB per item
    
    # Calculate max batch size
    max_batch = int(available_for_batches / per_item_memory)
    max_batch = max(1, min(max_batch, 32))  # Clamp between 1 and 32
    
    if user_batch_size > 0:
        # User specified a batch size, use it but warn if too high
        if user_batch_size > max_batch:
            print(f'WARNING: Requested batch size {user_batch_size} may exceed safe memory limits.')
            print(f'         Recommended max: {max_batch}')
            print(f'         Will use: {user_batch_size} (may OOM)')
        return user_batch_size
    
    print(f'Auto-selected batch size: {max_batch}')
    return max_batch

# Get optimal batch size
optimal_batch_size = get_optimal_batch_size()
print(f'Using batch size: {optimal_batch_size}')

# Determine input type for configuration
ext = os.path.splitext(input_file)[1].lower()
is_video = ext in ['.mp4', '.avi', '.mkv', '.mov', '.webm']

# Build config update based on input type
config_update = {
    'data.batch_size': optimal_batch_size,
    'data.num_workers': 0,  # Avoid multiprocessing memory overhead
}

# For video files, configure feature extractors for memory efficiency
# NOTE: Video model MUST match training config (vitg) - do not change model dimensions
if is_video:
    print('Configuring video feature extractor for memory efficiency...')
    config_update['data.video_feature'] = {
        'name': 'HuggingFaceVideo',
        'frequency': 2,
        'event_types': 'Video',
        'aggregation': 'sum',
        'clip_duration': 2,  # Reduced from 4 to 2 seconds (memory saving)
        'image': {
            'name': 'HuggingFaceImage',
            # KEEP original model - model was trained with vitg dimensions
            'model_name': 'facebook/vjepa2-vitg-fpc64-256',
            'layers': [0.75, 1.0],
            'batch_size': 1,  # Smaller batch for memory
            'infra': {
                'keep_in_ram': False,
                'folder': None,  # Must be None for nested image extractor
            },
        },
        'infra': {
            'keep_in_ram': False,
            'folder': '/root/.cache',
        },
    }
    # Also optimize audio feature extractor for video processing
    config_update['data.audio_feature'] = {
        'name': 'Wav2VecBert',
        'frequency': 2,
        'layers': [1.0],  # Use only final layer
        'event_types': 'Audio',
        'aggregation': 'sum',
        'infra': {
            'keep_in_ram': False,
            'folder': '/root/.cache',
        },
    }
    # Optimize text feature extractor
    config_update['data.text_feature'] = {
        'name': 'HuggingFaceText',
        'event_types': 'Word',
        'model_name': 'meta-llama/Llama-3.2-3B',
        'aggregation': 'sum',
        'frequency': 2,
        'contextualized': True,
        'layers': [1.0],
        'batch_size': 2,  # Reduced from 4
        'infra': {
            'keep_in_ram': False,
            'folder': '/root/.cache',
        },
    }
    print('Using memory-optimized config: vjepa2-vitg with 2s clips, batch=1')

# Ensure cache mode is set to 'cached' for all feature extractors to reuse existing features
for modality in ['text', 'audio', 'video']:
    if f'data.{modality}_feature' in config_update:
        config_update[f'data.{modality}_feature.infra.mode'] = 'cached'
    else:
        # If not already configured, set up with cache settings
        config_update[f'data.{modality}_feature.infra.folder'] = '/root/.cache'
        config_update[f'data.{modality}_feature.infra.mode'] = 'cached'
        config_update[f'data.{modality}_feature.infra.keep_in_ram'] = False

# Create model with conservative settings
try:
    model = TribeModel.from_pretrained(
        'facebook/tribev2',
        cache_folder='/root/.cache',
        cluster=None,  # Local execution, no slurm cluster
        config_update=config_update
    )
except torch.cuda.OutOfMemoryError as e:
    print(f'ERROR: Out of memory during model loading: {e}')
    print('Try reducing batch size or processing a shorter video clip.')
    raise

# Clear any cached models from feature extractors to free VRAM
# This happens automatically in get_loaders via _free_extractor_model
gc.collect()
torch.cuda.empty_cache()

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

# Clear memory after event preparation
gc.collect()
torch.cuda.empty_cache()

print(f'Events DataFrame shape: {df.shape}')
print(f'Event types: {df.type.unique().tolist()}')

# Run prediction with memory management
print('Starting prediction...')
try:
    preds, segments = model.predict(df)
    
    # Save results
    np.save(output_file, preds)
    print(f'Success! Predictions saved to {output_file}')
    print(f'Shape: {preds.shape} (n_timesteps, n_vertices)')
    print(f'Number of segments: {len(segments)}')
    
except torch.cuda.OutOfMemoryError as e:
    print(f'ERROR: Out of memory during prediction: {e}')
    print(f'Current batch size: {optimal_batch_size}')
    
    # Try to recover with even smaller settings
    if is_video and optimal_batch_size > 1:
        print('\\nAttempting recovery with batch_size=1...')
        torch.cuda.empty_cache()
        gc.collect()
        
        # Create new model with batch_size=1 (caching still enabled)
        config_update['data.batch_size'] = 1
        model = TribeModel.from_pretrained(
            'facebook/tribev2',
            cache_folder='/root/.cache',
            cluster=None,
            config_update=config_update
        )
        
        # Retry prediction
        preds, segments = model.predict(df)
        np.save(output_file, preds)
        print(f'Recovery successful! Predictions saved to {output_file}')
        print(f'Shape: {preds.shape} (n_timesteps, n_vertices)')
        print(f'Number of segments: {len(segments)}')
    else:
        print('Even batch size 1 is too large. The video may be too long or high-resolution.')
        print('Consider splitting the video into shorter clips (< 30 seconds).')
        raise
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
