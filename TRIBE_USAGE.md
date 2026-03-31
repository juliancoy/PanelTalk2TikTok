# TribeV2 Docker Wrapper Script

This directory contains a wrapper script `tribe.sh` for running TribeV2 inference on various media files using Docker with NVIDIA GPU support.

## Quick Start

### 1. Build the Docker Image (First time only)
```bash
cd tribev2
docker build -t tribev2 .
cd ..
```

### 2. Run Inference on a File
```bash
./tribe.sh <input_file>
```

Examples:
```bash
./tribe.sh output.mp4                    # Video file
./tribe.sh CCAN_Platform_audiofix.wav    # Audio file  
./tribe.sh sample.txt                    # Text file
```

## Script Features

### Input Support
- **Video**: `.mp4`, `.avi`, `.mkv`, `.mov`, `.webm`
- **Audio**: `.wav`, `.mp3`, `.flac`, `.ogg`
- **Text**: `.txt` (converted to speech via gTTS)

### Output
- Creates `<input_basename>_tribe.npy` in the same directory as input
- NumPy array format with shape `(n_timesteps, n_vertices)`
- `n_timesteps`: Number of time steps (1 TR = 1 second)
- `n_vertices`: ~20,000 vertices on fsaverage5 cortical mesh

### Caching
- Creates `.tribe_cache` directory in input file's directory
- Caches downloaded models and extracted features
- Subsequent runs on same file are faster

## How It Works

The script:
1. Validates input file exists and has supported extension
2. Checks if Docker image exists, builds if needed
3. Mounts input directory as `/data` in container
4. Mounts cache directory as `/root/.cache`
5. Runs TribeV2 inference in Docker with GPU support
6. Saves predictions as NumPy `.npy` file

## Docker Command Equivalent

The script runs this equivalent Docker command:
```bash
docker run --rm --gpus all \
  -v "$(dirname $INPUT_FILE)/.tribe_cache:/root/.cache" \
  -v "$(dirname $INPUT_FILE):/data" \
  tribev2 python -c "
from tribev2 import TribeModel
import numpy as np
# ... inference code
"
```

## Example Output

```
==========================================
TribeV2 Processing
==========================================
Input file:  output.mp4
Input type:  mp4
Output file: output_tribe.npy
Cache dir:   /path/to/.tribe_cache
==========================================
Running TribeV2 inference...
Processing: /data/output.mp4
Output: /data/output_tribe.npy
Success! Predictions saved to /data/output_tribe.npy
Shape: (120, 20484) (n_timesteps, n_vertices)
Number of segments: 120
==========================================
Processing complete!
Output saved to: /path/to/output_tribe.npy
Output size: 19M
Array shape: (120, 20484)
```

## Requirements

1. **Docker** with NVIDIA Container Toolkit
2. **NVIDIA GPU** with CUDA support
3. **Sufficient disk space** (~5GB for Docker image + cache)

## Testing NVIDIA Container Toolkit

Run the test script:
```bash
cd tribev2
./test_nvidia.sh
```

## Troubleshooting

### Docker Image Not Found
```bash
cd tribev2
docker build -t tribev2 .
```

### CUDA Not Available in Container
```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base nvidia-smi
```

### Permission Denied
```bash
chmod +x tribe.sh
```

### Output File Not Created
- Check Docker logs for errors
- Ensure input file is valid media file
- Check available disk space

## Advanced Usage

### Batch Processing
```bash
for file in *.mp4; do
    ./tribe.sh "$file"
done
```

### Custom Output Directory
```bash
INPUT="output.mp4"
OUTPUT_DIR="./results"
mkdir -p "$OUTPUT_DIR"
./tribe.sh "$INPUT"
mv "$(dirname "$INPUT")/$(basename "$INPUT" .mp4)_tribe.npy" "$OUTPUT_DIR/"
```

### Clean Cache
```bash
rm -rf .tribe_cache
```

## Notes

- First run downloads ~1GB model from HuggingFace
- Video/audio feature extraction can be slow on first run
- Results are cached for subsequent runs
- GPU memory usage depends on input length and batch size