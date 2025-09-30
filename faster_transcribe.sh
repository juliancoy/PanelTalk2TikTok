docker run --rm --gpus all \
  -v "$PWD":/work -w /work \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface \
  -v "$HOME/.cache/ctranslate2":/root/.cache/ctranslate2 \
  -v "fasterwhisper-python-packages":"/usr/local/lib/python3.10/site-packages" \
  nvcr.io/nvidia/pytorch:24.06-py3 \
  bash -c 'python -m pip install --upgrade pip && pip install faster-whisper && python faster_transcribe.py "$@"' -- "$1"
