#!/usr/bin/env python3
import json
import csv
import sys
import argparse

def export_words_to_csv(json_file, fill_gaps=False, min_gap=0.0):
    csv_file = json_file + ".csv"
    # Load Whisper JSON
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Collect words in chronological order
    words = []
    for segment in data.get("segments", []):
        for w in segment.get("words", []):
            try:
                start = float(w.get("start", 0.0))
            except Exception:
                start = 0.0
            try:
                end = float(w.get("end", start))
            except Exception:
                end = start
            words.append({
                "speaker": w.get("speaker", "") if w.get("speaker") is not None else "",
                "start": start,
                "end": end,
                "word": (w.get("word") or "").strip()
            })

    # Ensure chronological order by start time
    words.sort(key=lambda x: x["start"])

    # Open CSV for writing
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["speaker", "start", "end", "word"])  # header row

        prev = None
        for w in words:
            if prev is not None and fill_gaps:
                gap = w["start"] - prev["end"]
                if gap > float(min_gap):
                    # Insert a blank word row occupying the gap time.
                    # Use the previous speaker for the blank row (keeps context of who was speaking before the gap).
                    writer.writerow([
                        prev.get("speaker", ""),
                        "{:.3f}".format(prev["end"]),
                        "{:.3f}".format(w["start"]),
                        ""
                    ])
            # Write the current word row
            writer.writerow([
                w.get("speaker", ""),
                "{:.3f}".format(w["start"]),
                "{:.3f}".format(w["end"]),
                w.get("word", "")
            ])
            prev = w

    print(f"Exported word timings to {csv_file}")
    return csv_file

def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Export word-level timings from Whisper JSON to CSV."
    )
    parser.add_argument("json_file", help="Path to Whisper JSON file")
    parser.add_argument(
        "--fill-gaps",
        action="store_true",
        help="Insert blank rows for temporal gaps between consecutive words"
    )
    parser.add_argument(
        "--min-gap",
        type=float,
        default=0.0,
        help="Minimum gap (in seconds) required to insert a blank row (default: 0.0)"
    )
    return parser.parse_args(argv)

if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    export_words_to_csv(args.json_file, fill_gaps=args.fill_gaps, min_gap=args.min_gap)
