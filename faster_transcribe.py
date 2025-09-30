from faster_whisper import WhisperModel
import os, sys, json

infilename = sys.argv[1]
basename, _ = os.path.splitext(infilename)
outfilename = basename + ".json"

model_size = "medium"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

segments, info = model.transcribe(
    infilename,
    beam_size=5,
    word_timestamps=True   # 👈 enable word boundaries
)

print("Detected language '%s' with probability %f" % (info.language, info.language_probability))

results = {
    "language": info.language,
    "language_probability": info.language_probability,
    "segments": []
}

for segment in segments:
    print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
    seg_data = {
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
        "words": []
    }
    for word in segment.words:
        #print("   [%.2fs -> %.2fs] %s" % (word.start, word.end, word.word))
        seg_data["words"].append({
            "start": word.start,
            "end": word.end,
            "word": word.word
        })
    results["segments"].append(seg_data)

# Save JSON
with open(outfilename, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"\nJSON transcript saved to {outfilename}")
