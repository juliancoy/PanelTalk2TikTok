import sys, os, json
import whisperx

infilename = sys.argv[1]
basename, _ = os.path.splitext(infilename)
outfilename = basename + ".json"

device = "cuda"  # or "cpu"
batch_size = 16
compute_type = "float16"  # same as faster-whisper

# 1. Load Whisper model
model = whisperx.load_model("medium", device, compute_type=compute_type)

# 2. Transcribe
result = model.transcribe(infilename, batch_size=batch_size)

# 3. Load alignment model for the detected language
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)

# 4. Align Whisper output
result_aligned = whisperx.align(result["segments"], model_a, metadata, infilename, device)

# result_aligned["segments"] now has improved word timings
with open(outfilename, "w", encoding="utf-8") as f:
    json.dump(result_aligned, f, indent=2, ensure_ascii=False)

print(f"\nJSON transcript saved to {outfilename}")
