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
The script creates several types of cached files to speed up subsequent runs:

1. **Extracted Audio** (`.wav`): Saved alongside video files
2. **Transcripts** (`.tsv`): Word-level timestamps from audio
3. **Features** (in `.tribe_cache/`): Video, audio, and text embeddings
4. **Models** (in `.tribe_cache/`): Downloaded HuggingFace models (~1GB)

All intermediate files are stored **alongside the source media** and are **not regenerated** if they exist. To force reprocessing, delete the relevant cached files.

```bash
# Force re-extract audio and regenerate transcript
rm video.wav video.tsv
./tribe.sh video.mp4

# Force recompute all features (keep downloaded models)
rm -rf .tribe_cache/features
./tribe.sh video.mp4

# Complete fresh start (delete everything including models)
rm -rf .tribe_cache video.wav video.tsv
./tribe.sh video.mp4
```

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

### CUDA Out of Memory
The script automatically handles memory scaling, but for very long/high-resolution videos:
```bash
# Try with explicit smaller batch size
./tribe.sh video.mp4 4

# For very long videos, split into shorter clips first
ffmpeg -i long_video.mp4 -t 30 -c copy segment1.mp4  # First 30 seconds
ffmpeg -i long_video.mp4 -ss 30 -t 30 -c copy segment2.mp4  # Next 30 seconds
```

**Memory Optimization Features:**
- Auto-calculates batch size based on available GPU memory
- Uses smaller video model (`vjepa2-vitl` vs `vjepa2-vitg`)
- Reduces clip duration from 4s to 2s for video processing
- Automatically retries with batch_size=1 if OOM occurs
- Frees extractor models from VRAM after feature extraction

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

## Inspecting Results

### Output Format
The output is a NumPy array with shape `(n_timesteps, n_vertices)`:
- **n_timesteps**: Number of 1-second time points
- **n_vertices**: ~20,000 vertices on fsaverage5 cortical surface

### Quick Inspection
```bash
# View basic statistics
./inspect_tribe.py CodeCollective/swc_preliminary_tribe.npy

# Export to CSV for external analysis
./inspect_tribe.py CodeCollective/swc_preliminary_tribe.npy --save-csv
```

### Visualization
```bash
# Plot first 10 timesteps as brain surface
./visualize_tribe.py CodeCollective/swc_preliminary_tribe.npy -o brain.png

# Plot specific timestep
./visualize_tribe.py CodeCollective/swc_preliminary_tribe.npy -s 5 -o t5.png

# Create video animation (requires Docker with display)
./visualize_tribe.py CodeCollective/swc_preliminary_tribe.npy --video -o brain.mp4
```

### Python API
```python
import numpy as np
from tribev2.plotting import PlotBrain

# Load predictions
preds = np.load('video_tribe.npy')  # Shape: (timesteps, vertices)

# Create plotter
plotter = PlotBrain(mesh="fsaverage5")

# Plot single timestep
fig = plotter.plot(preds[0], view="left")
fig.savefig('brain_t0.png')

# Plot multiple timesteps
fig = plotter.plot_timesteps(
    preds[:15],  # First 15 seconds
    cmap="fire",
    norm_percentile=99,
)
fig.savefig('brain_timesteps.png')
```

### Interpreting Results
- **Values**: Represent predicted fMRI BOLD response at each cortical vertex
- **Normalization**: Use `norm_percentile=99` for consistent visualization
- **Views**: "left", "right", "dorsal", "ventral", "anterior", "posterior"
- **Atlas**: Results are on fsaverage5 (20k vertices), full brain coverage

### Visualization Style
TribeV2 uses a **cortical surface projection** with:
- **Colormap**: "fire" (black → red → yellow/white for low → high activity)
- **Surface**: fsaverage5 standard brain mesh
- **Background**: Sulcal/gyral patterns for anatomical reference
- **Layout**: Multiple timesteps side-by-side for temporal comparison

The "fire" colormap is specifically chosen to highlight brain activation patterns similar to traditional fMRI visualization.

## Notes

- First run downloads ~1GB model from HuggingFace
- Video/audio feature extraction can be slow on first run
- Results are cached for subsequent runs
- GPU memory usage depends on input length and batch size