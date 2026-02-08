#!/usr/bin/env python3
"""
Final comparison: Current vs V3 with tuned threshold.
"""

import re
import numpy as np
import chromadb
from pathlib import Path
from dotenv import load_dotenv
import statistics
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
    all_emb = []
    for i in range(0, len(texts), 100):
        batch = texts[i:i+100]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        all_emb.extend([e.embedding for e in response.data])
    return np.array(all_emb)


def find_boundaries_tuned(embeddings, threshold=0.15):
    """Find boundaries with tuned threshold."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / norms
    
    boundaries = []
    for i in range(1, len(embeddings_norm)):
        sim = np.dot(embeddings_norm[i-1], embeddings_norm[i])
        if sim < threshold:
            boundaries.append(i)
    
    return boundaries


def main():
    print("=" * 70)
    print("FINAL COMPARISON: Current vs V3 (Tuned)")
    print("=" * 70)
    
    vtt_file = Path('audio/gXY1kx7zlkk.en.vtt')
    video_id = 'gXY1kx7zlkk'
    
    # Parse and prepare
    words = parse_vtt(str(vtt_file))
    sentences = words_to_sentences(words)
    total_duration_ms = words[-1]['start_ms'] + 200
    
    print(f"\nEpisode: ICE Chaos / All-In Podcast")
    print(f"Duration: {total_duration_ms/1000/60:.1f} minutes")
    print(f"Sentences: {len(sentences)}")
    
    # Get all embeddings
    print(f"\nGetting embeddings for all {len(sentences)} sentences...")
    texts = [s['text'] for s in sentences]
    embeddings = get_embeddings(texts)
    print(f"Done. Shape: {embeddings.shape}")
    
    # Find boundaries with tuned threshold
    # Use percentile-based threshold to get ~25-30 segments
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / norms
    
    sims = []
    for i in range(1, len(embeddings_norm)):
        sim = np.dot(embeddings_norm[i-1], embeddings_norm[i])
        sims.append(sim)
    sims = np.array(sims)
    
    # Get threshold for ~28 segments (matching current)
    target_segments = 28
    threshold = np.percentile(sims, (target_segments / len(sentences)) * 100)
    print(f"\nUsing adaptive threshold: {threshold:.3f} (to get ~{target_segments} segments)")
    
    boundaries = [i for i, s in enumerate(sims) if s < threshold]
    
    # Create v3 segments
    v3_segments = []
    boundary_indices = [0] + [b+1 for b in boundaries] + [len(sentences)]
    for i in range(len(boundary_indices) - 1):
        start_idx = boundary_indices[i]
        end_idx = boundary_indices[i+1] - 1
        if end_idx >= start_idx and start_idx < len(sentences) and end_idx < len(sentences):
            v3_segments.append({
                'start_ms': sentences[start_idx]['start_ms'],
                'end_ms': sentences[end_idx]['end_ms'],
                'first_sentence': sentences[start_idx]['text'][:80]
            })
    
    # Analyze v3
    v3_durations = [(s['end_ms'] - s['start_ms'])/1000 for s in v3_segments]
    v3_covered = sum(s['end_ms'] - s['start_ms'] for s in v3_segments)
    v3_coverage = v3_covered / total_duration_ms
    
    print(f"\n{'='*70}")
    print("V3 SEGMENTATION (Tuned)")
    print(f"{'='*70}")
    print(f"  Segments: {len(v3_segments)}")
    print(f"  Coverage: {v3_coverage*100:.1f}%")
    print(f"  Mean duration: {statistics.mean(v3_durations):.1f}s")
    print(f"  Median duration: {statistics.median(v3_durations):.1f}s")
    print(f"  Range: {min(v3_durations):.1f}s - {max(v3_durations):.1f}s")
    
    short = sum(1 for d in v3_durations if d < 30)
    long = sum(1 for d in v3_durations if d > 300)
    print(f"  Too short (<30s): {short} ({short/len(v3_durations)*100:.0f}%)")
    print(f"  Too long (>5min): {long}")
    
    print(f"\n  First 5 segments:")
    for i, s in enumerate(v3_segments[:5]):
        dur = (s['end_ms'] - s['start_ms'])/1000
        print(f"    {i+1}. {s['start_ms']/1000:.0f}s-{s['end_ms']/1000:.0f}s ({dur:.0f}s): {s['first_sentence'][:50]}...")
    
    # Get current segments
    db = chromadb.PersistentClient(path='chroma_data')
    col = db.get_collection('segments')
    results = col.get(include=['metadatas', 'documents'])
    
    current_segs = []
    for i, meta in enumerate(results['metadatas']):
        if 'ICE' in meta.get('episode_title', ''):
            current_segs.append({
                'start_ms': meta.get('start_ms', 0) or 0,
                'end_ms': meta.get('end_ms', 0) or 0,
                'topic': meta.get('primary_topic', '') or '[NO TOPIC]',
                'summary': results['documents'][i][:80] if results['documents'][i] else ''
            })
    
    current_segs.sort(key=lambda x: x['start_ms'])
    
    curr_durations = [(s['end_ms'] - s['start_ms'])/1000 for s in current_segs if s['end_ms'] > s['start_ms']]
    curr_covered = sum(s['end_ms'] - s['start_ms'] for s in current_segs)
    curr_coverage = curr_covered / total_duration_ms
    
    print(f"\n{'='*70}")
    print("CURRENT SEGMENTATION")
    print(f"{'='*70}")
    print(f"  Segments: {len(current_segs)}")
    print(f"  Coverage: {curr_coverage*100:.1f}%")
    if curr_durations:
        print(f"  Mean duration: {statistics.mean(curr_durations):.1f}s")
        print(f"  Median duration: {statistics.median(curr_durations):.1f}s")
        print(f"  Range: {min(curr_durations):.1f}s - {max(curr_durations):.1f}s")
    
    short_curr = sum(1 for d in curr_durations if d < 30)
    long_curr = sum(1 for d in curr_durations if d > 300)
    print(f"  Too short (<30s): {short_curr}")
    print(f"  Too long (>5min): {long_curr}")
    
    print(f"\n  First 5 segments:")
    for i, s in enumerate(current_segs[:5]):
        dur = (s['end_ms'] - s['start_ms'])/1000
        print(f"    {i+1}. {s['start_ms']/1000:.0f}s-{s['end_ms']/1000:.0f}s ({dur:.0f}s): {s['topic'][:50]}")
    
    # Final comparison
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"\n  {'Metric':<25} {'Current':>15} {'V3 (Tuned)':>15} {'Winner':>10}")
    print(f"  {'-'*65}")
    
    # Coverage
    cov_winner = "V3" if v3_coverage > curr_coverage else "Current"
    print(f"  {'Coverage':<25} {curr_coverage*100:>14.1f}% {v3_coverage*100:>14.1f}% {cov_winner:>10}")
    
    # Segment count
    print(f"  {'Segments':<25} {len(current_segs):>15} {len(v3_segments):>15} {'--':>10}")
    
    # Mean duration
    if curr_durations:
        mean_curr = statistics.mean(curr_durations)
        mean_v3 = statistics.mean(v3_durations)
        dur_winner = "Tie" if abs(mean_curr - mean_v3) < 20 else ("Current" if abs(mean_curr - 90) < abs(mean_v3 - 90) else "V3")
        print(f"  {'Mean duration (s)':<25} {mean_curr:>15.1f} {mean_v3:>15.1f} {dur_winner:>10}")
    
    # Short segments
    print(f"  {'Too short (<30s)':<25} {short_curr:>15} {short:>15} {'V3' if short < short_curr else 'Current':>10}")
    
    # Long segments
    print(f"  {'Too long (>5min)':<25} {long_curr:>15} {long:>15} {'V3' if long < long_curr else 'Current':>10}")
    
    print(f"\n  KEY INSIGHT:")
    print(f"  V3 achieves {v3_coverage*100:.0f}% coverage vs current {curr_coverage*100:.0f}%")
    print(f"  That's {(v3_coverage - curr_coverage)*100:.0f} percentage points more content indexed!")


if __name__ == "__main__":
    main()
