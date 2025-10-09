#!/usr/bin/env python3
import sys
import librosa
import numpy as np
import soundfile as sf
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def _compute_mfcc(y, sr, start, window_samples, n_mfcc=13):
    """Helper: compute flattened MFCC for a chunk starting at 'start'."""
    chunk = y[start:start + window_samples]
    if len(chunk) < window_samples:
        chunk = np.pad(chunk, (0, window_samples - len(chunk)))
    mfcc = librosa.feature.mfcc(y=chunk, sr=sr, n_mfcc=n_mfcc)
    return start, mfcc.flatten()

def _cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)

def find_best_local_matches(yq, yc, sr, window_sec=0.5, hop_sec=0.5, corr_threshold=0.6):
    """
    Find best frequency-based matches between quality and content audio.
    Uses MFCC cosine similarity on fixed-size chunks.
    Each content chunk maps to the best unused quality chunk.
    Returns list of (content_start, quality_start, duration, similarity).
    """
    window_samples = int(window_sec * sr)

    # --- Precompute MFCCs in parallel ---from tqdm import tqdm  # pip install tqdm
    def process_chunks(y, label):
        starts = range(0, len(y) - window_samples, int(hop_sec * sr))
        tasks = [(y, sr, s, window_samples) for s in starts]
        print(f"Extracting MFCCs for {label} ({len(tasks)} chunks)...")
        with Pool(processes=cpu_count()) as pool:
            results = list(tqdm(
                pool.starmap(_compute_mfcc, tasks),
                total=len(tasks)
            ))
        return dict(results)


    print("Extracting MFCC features (parallelized)...")
    quality_feats = process_chunks(yq, "quality")
    content_feats = process_chunks(yc, "content")


    quality_used = {}   # quality_start -> best similarity
    matches = []

    print("Matching content chunks to quality chunks...")
    for c_start, c_feat in tqdm(content_feats.items(), total=len(content_feats)):
        best_sim = -1
        best_q_start = None

        for q_start, q_feat in quality_feats.items():
            sim = _cosine_similarity(c_feat, q_feat)
            if sim > best_sim:
                best_sim = sim
                best_q_start = q_start

        if best_sim >= corr_threshold and best_q_start is not None:
            if best_q_start not in quality_used or best_sim > quality_used[best_q_start]:
                quality_used[best_q_start] = best_sim
                matches.append((c_start, best_q_start, window_samples, best_sim))

    # Apply matches in content order
    matches.sort(key=lambda x: x[0])
    return matches

def align_sequential(quality_audio, content_audio, output_file="quality_aligned.wav", 
                    window_sec=0.5, hop_sec=0.5, corr_threshold=0.6):
    """
    Align quality audio to content using frequency-based chunk similarity.
    Leaves unmatched regions as original content audio.
    """
    print("Loading audio...")
    yq, srq = librosa.load(quality_audio, sr=None, mono=True)
    yc, src = librosa.load(content_audio, sr=srq, mono=True)
    
    if srq != src:
        raise ValueError(f"Sample rate mismatch: quality={srq}, content={src}")
    
    print(f"Finding best frequency matches (chunk={window_sec}s, threshold={corr_threshold})...")
    matches = find_best_local_matches(yq, yc, srq, window_sec, hop_sec, corr_threshold)
    
    print(f"Found {len(matches)} matches")
    
    # Start with content audio (not silence)
    yq_aligned = yc.copy()
    fade_samples = int(0.01 * srq)  # 10ms crossfade
    
    for i, (c_pos, q_pos, duration, sim) in enumerate(matches):
        c_end = min(c_pos + duration, len(yc))
        q_end = min(q_pos + duration, len(yq))
        copy_len = min(c_end - c_pos, q_end - q_pos)
        
        chunk = yq[q_pos:q_pos + copy_len].copy()
        
        # Crossfade boundaries
        if copy_len > 2 * fade_samples:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            chunk[:fade_samples] = (chunk[:fade_samples] * fade_in +
                                    yq_aligned[c_pos:c_pos + fade_samples] * (1 - fade_in))
            chunk[-fade_samples:] = (chunk[-fade_samples:] * fade_out +
                                     yq_aligned[c_pos + copy_len - fade_samples:c_pos + copy_len] * (1 - fade_out))
        
        # Replace content with quality
        yq_aligned[c_pos:c_pos + copy_len] = chunk

        if i % 10 == 0:
            print(f"  Match {i+1}/{len(matches)}: content={c_pos/srq:.2f}s "
                  f"-> quality={q_pos/srq:.2f}s (sim={sim:.3f})")
    
    # Coverage info
    replaced_samples = sum(min(duration, len(yc) - c_pos, len(yq) - q_pos) 
                          for c_pos, q_pos, duration, _ in matches)
    coverage = replaced_samples / len(yq_aligned) * 100
    print(f"\nReplaced: {coverage:.1f}% of content audio with quality audio")
    print(f"Unmatched regions kept as original content audio")
    
    # Normalize
    if np.max(np.abs(yq_aligned)) > 0:
        yq_aligned = yq_aligned / np.max(np.abs(yq_aligned)) * 0.95
    
    print(f"\nExporting aligned audio to {output_file}...")
    sf.write(output_file, yq_aligned, src)
    print(f"✅ Done! Saved aligned track as {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python align_sequential.py <quality_audio.wav> <content_audio.wav> [threshold]")
        print("  threshold: similarity threshold 0-1 (default: 0.6, higher = stricter)")
        sys.exit(1)
    
    threshold = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
    
    align_sequential(
        sys.argv[1], 
        sys.argv[2], 
        sys.argv[2] + "_quality_aligned.wav",
        corr_threshold=threshold
    )
