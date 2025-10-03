#!/usr/bin/env python3
import json
import csv
import sys

def export_words_to_csv(json_file, csv_file):
    # Load Whisper JSON
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Open CSV for writing
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["start", "end", "word"])  # header row

        # Iterate over segments and words
        for segment in data.get("segments", []):
            for word_info in segment.get("words", []):
                writer.writerow([
                    word_info.get("speaker"),
                    word_info.get("start"),
                    word_info.get("end"),
                    word_info.get("word").strip()
                ])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python export_words.py input.json")
        sys.exit(1)

    json_file = sys.argv[1]
    csv_file = json_file+".csv"
    export_words_to_csv(json_file, csv_file)
    print(f"Exported word timings to {csv_file}")

