docker run --gpus all -it \
  -v $(pwd):/app -w /app \
  -e hf_token=$HF_TOKEN \
  ghcr.io/jim60105/whisperx:large-v3-tl-77e20c4 \
  whisperx $1 \
    --output_dir /app \
    --output_format json \
    --diarize \
    --language en \
    --hf_token $HF_TOKEN
