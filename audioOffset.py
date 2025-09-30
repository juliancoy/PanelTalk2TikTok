import sys
import numpy as np
from pydub import AudioSegment
from scipy.signal import correlate

def load_audio(filename, target_sr=16000):
    """Load audio file with pydub, convert to mono, resample, and return numpy array + sr."""
    print(f"[INFO] Loading {filename} ...")
    audio = AudioSegment.from_file(filename)
    audio = audio.set_channels(1).set_frame_rate(target_sr)  # mono + resample
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    samples /= np.iinfo(audio.array_type).max  # normalize to -1..1
    print(f"[INFO] Loaded {filename}: {len(samples)/target_sr:.2f}s at {target_sr} Hz")
    return samples, target_sr

def find_offset(y1, y2, sr):
    """Find offset between two signals in seconds using cross-correlation."""
    print("[INFO] Computing cross-correlation (this may take a while for long files)...")
    corr = correlate(y1, y2, mode="full")
    lag = np.argmax(corr) - (len(y2) - 1)
    offset = lag / sr
    return offset

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python audioOffset.py file1 file2")
        sys.exit(1)

    sound1, sound2 = sys.argv[1], sys.argv[2]

    # Load both audio files
    y1, sr1 = load_audio(sound1)
    y2, sr2 = load_audio(sound2)

    # Ensure both sample rates match (they should, since we enforce it)
    assert sr1 == sr2
    sr = sr1

    # Find offset
    offset = find_offset(y1, y2, sr)

    print(f"[RESULT] Offset between {sound1} and {sound2}: {offset:.3f} seconds")
    if offset > 0:
        print(f"[RESULT] {sound1} is ahead of {sound2} by {offset:.3f} seconds")
    else:
        print(f"[RESULT] {sound2} is ahead of {sound1} by {-offset:.3f} seconds")
