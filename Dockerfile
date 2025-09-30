# Use NVIDIA CUDA devel image (includes toolkit + nvcc for Triton)
FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip ffmpeg git curl && \
    rm -rf /var/lib/apt/lists/*

# Create venv
RUN python3 -m venv /venv
ENV PATH="/venv/bin:$PATH"

# Upgrade pip and install requirements
RUN pip install --upgrade pip && \
    pip install \
      git+https://github.com/openai/whisper.git \
      whisperx "transformers<5" torch torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Set cache directories for persistence (if you mount them later)
ENV TRANSFORMERS_CACHE=/root/.cache/huggingface
ENV WHISPER_CACHE=/root/.cache/whisper
ENV TORCH_HOME=/root/.cache/torch

# Pre-download the model at build time so container has it cached
# We'll generate a 1-second silent wav and run whisper on it.
RUN mkdir -p /tmp/init && \
    ffmpeg -f lavfi -i anullsrc=r=16000:cl=mono -t 1 /tmp/init/silence.wav && \
    whisper /tmp/init/silence.wav --model large-v3 --output_dir /tmp/init/out --output_format json --word_timestamps True || true

WORKDIR /app

# Default entrypoint just opens a shell; override with your whisper command
ENTRYPOINT ["/bin/bash"]

