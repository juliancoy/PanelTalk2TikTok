#!/usr/bin/env bash
set -euo pipefail

# Input file (change if needed)
AUDIO="$1"

# Optional: sanity check before running
[ -f "$AUDIO" ] || { echo "File not found: $AUDIO"; exit 1; }

docker run --rm -it \
  --gpus all \
  --env AUDIO="$AUDIO" \
  --mount type=bind,src="$(pwd)",dst=/app \
  --mount type=volume,src=whisper-venv,dst=/venv \
  --mount type=volume,src=pip-cache,dst=/root/.cache/pip \
  --mount type=volume,src=whisper-model-cache,dst=/root/.cache/whisper \
  --workdir /app \
  pytorch/pytorch:latest \
  bash -lc '
    set -e
    apt-get update && apt-get install -y ffmpeg git && \
    python -m venv /venv && . /venv/bin/activate && \
    pip install --upgrade pip && \
    pip install git+https://github.com/openai/whisper.git && \
    mkdir -p /app/transcripts && \
    # Use the env var AUDIO inside the container and the bind mount path /app
    whisper "/app/$AUDIO" \
      --model large-v3 \
      --output_dir /app/transcripts \
      --output_format all \
      --word_timestamps True

  '

