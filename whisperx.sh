#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <input-audio-or-video-file>" >&2
  exit 1
fi

if [[ ! -f hftoken.txt ]]; then
  echo "ERROR: hftoken.txt not found" >&2
  exit 1
fi

HF_TOKEN="$(tr -d '\r\n' < hftoken.txt)"
mkdir -p .cache

HOST_PWD="$(pwd)"
HOST_PWD_ABS="$(readlink -f "$HOST_PWD")"
INPUT_ABS="$(readlink -f "$1")"

if [[ ! -f "$INPUT_ABS" ]]; then
  echo "ERROR: input file not found: $1" >&2
  exit 1
fi

case "$INPUT_ABS" in
  "$HOST_PWD_ABS"/*) ;;
  *)
    echo "ERROR: input file must be inside the current project directory:" >&2
    echo "  project dir: $HOST_PWD_ABS" >&2
    echo "  input file : $INPUT_ABS" >&2
    exit 1
    ;;
esac

CONTAINER_INPUT="/app/${INPUT_ABS#$HOST_PWD_ABS/}"
CONTAINER_OUT_DIR="$(dirname "$CONTAINER_INPUT")"

docker run --rm --gpus all -it \
  --user "$(id -u):$(id -g)" \
  -v "$HOST_PWD_ABS/.cache":/.cache \
  -v "$HOST_PWD_ABS/.cache":/tmp/.cache \
  -v "$HOST_PWD_ABS":/app \
  -w /app \
  -e HOME="/tmp" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e MPLCONFIGDIR="/tmp/.cache/matplotlib" \
  -e HF_HOME="/tmp/.cache/huggingface" \
  -e HF_HUB_CACHE="/tmp/.cache/huggingface/hub" \
  -e TRANSFORMERS_CACHE="/tmp/.cache/huggingface/transformers" \
  -e TORCH_HOME="/tmp/.cache/torch" \
  ghcr.io/jim60105/whisperx:large-v3-tl-77e20c4 \
  whisperx "$CONTAINER_INPUT" \
    --output_dir "$CONTAINER_OUT_DIR" \
    --output_format json \
    --diarize \
    --language en \
    --hf_token "$HF_TOKEN"
