#!/usr/bin/env python3
"""
Tune the boundary detection threshold.
"""

import re
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import openai
client = openai.OpenAI()


def parse_vtt(vtt_path: str):
    words = []
    with open(vtt_path, 'r') as f:
        content = f.read()
    
    for line in content.split('\n'):
        if '-->' not in line and not line.startswith('WEBVTT') and line.strip():
            matches = re.findall(r'<(\d{2}:\d{2}:\d{2}\.\d{3})><c>\s*([^<]+)</c>', line)
            for ts, word in matches:
                h, m, s = ts.split(':')
                ms = int(float(h) * 3600000 + float(m) * 60000 + float(s) * 1000)
                words.append({'text': word.strip(), 'start_ms': ms})
    
    return words


def words_to_sentences(words, pause_threshold_ms=800):
    if not words:
        return []
    
    sentences = []
    current_words = [words[0]]
    
    for i in range(1, len(words)):
        w = words[i]
        prev = words[i-1]
        gap = w['start_ms'] - prev['start_ms'] - 200
        
        if gap > pause_threshold_ms:
            text = ' '.join(cw['text'] for cw in current_words)
            sentences.append({
                'text': text,
                'start_ms': current_words[0]['start_ms'],
                'end_ms': current_words[-1]['start_ms'] + 200
            })
            current_words = [w]
        else:
            current_words.append(w)
    
    if current_words:
        text = ' '.join(cw['text'] for cw in current_words)
        sentences.append({
            'text': text,
            'start_ms': current_words[0]['start_ms'],
            'end_ms': current_words[-1]['start_ms'] + 200
        })
    
    return sentences


def get_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts[:200]
    )
    return np.array([e.embedding for e in response.data])


def analyze_similarities(embeddings):
    """Analyze the distribution of adjacent sentence similarities."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / norms
    
    # Calculate all adjacent similarities
    sims = []
    for i in range(1, len(embeddings_norm)):
        sim = np.dot(embeddings_norm[i-1], embeddings_norm[i])
        sims.append(sim)
    
    return np.array(sims)


def main():
    print("=" * 70)
    print("THRESHOLD TUNING ANALYSIS")
    print("=" * 70)
    
    # Use one VTT file
    vtt_file = Path('audio/gXY1kx7zlkk.en.vtt')
    
    words = parse_vtt(str(vtt_file))
    sentences = words_to_sentences(words)
    print(f"Sentences: {len(sentences)}")
    
    texts = [s['text'] for s in sentences[:200]]
    print(f"Getting embeddings for {len(texts)} sentences...")
    embeddings = get_embeddings(texts)
    
    sims = analyze_similarities(embeddings)
    
    print(f"\nAdjacent sentence similarity distribution:")
    print(f"  Min: {sims.min():.3f}")
    print(f"  Max: {sims.max():.3f}")
    print(f"  Mean: {sims.mean():.3f}")
    print(f"  Median: {np.median(sims):.3f}")
    print(f"  Std: {sims.std():.3f}")
    
    # Percentiles
    print(f"\nPercentiles:")
    for p in [5, 10, 25, 50, 75, 90, 95]:
        print(f"  {p}th: {np.percentile(sims, p):.3f}")
    
    # Count boundaries at different thresholds
    print(f"\nBoundaries at different thresholds:")
    for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        boundaries = sum(1 for s in sims if s < thresh)
        segments = boundaries + 1
        avg_len = len(sentences[:200]) / segments * 7  # rough seconds
        print(f"  threshold={thresh:.1f}: {boundaries:3d} boundaries, {segments:3d} segments, ~{avg_len:.0f}s avg")
    
    # What threshold gives us ~25 segments (similar to current)?
    target_segments = 25
    target_boundaries = target_segments - 1
    # Find threshold that gives approximately this many
    sorted_sims = np.sort(sims)
    if target_boundaries < len(sorted_sims):
        ideal_threshold = sorted_sims[target_boundaries]
        print(f"\nTo get ~{target_segments} segments, use threshold: {ideal_threshold:.3f}")
    
    # Show what the lowest similarities are (likely real topic changes)
    print(f"\nLowest similarity pairs (likely topic changes):")
    indices = np.argsort(sims)[:10]
    for idx in indices:
        print(f"  sim={sims[idx]:.3f} at sentence {idx}:")
        print(f"    BEFORE: ...{sentences[idx]['text'][-50:]}")
        print(f"    AFTER:  {sentences[idx+1]['text'][:50]}...")


if __name__ == "__main__":
    main()
