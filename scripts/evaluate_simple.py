#!/usr/bin/env python3
"""
Evaluate current segmentation quality without needing new embeddings.
"""

import re
import json
import chromadb
from pathlib import Path
from dataclasses import dataclass
from typing import List
import statistics


@dataclass
class Word:
    text: str
    start_ms: int
    end_ms: int


@dataclass
class Sentence:
    text: str
    start_ms: int
    end_ms: int


def parse_vtt(vtt_path: str) -> List[Word]:
    """Parse VTT file with word-level timestamps."""
    words = []
    with open(vtt_path, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    for line in lines:
        if '-->' not in line and not line.startswith('WEBVTT') and line.strip():
            # Extract: <timestamp><c> word</c> pattern
            matches = re.findall(r'<(\d{2}:\d{2}:\d{2}\.\d{3})><c>\s*([^<]+)</c>', line)
            for ts, word in matches:
                h, m, s = ts.split(':')
                ms = int(float(h) * 3600000 + float(m) * 60000 + float(s) * 1000)
                words.append(Word(text=word.strip(), start_ms=ms, end_ms=ms + 200))
    
    # Sort and dedupe
    words.sort(key=lambda w: w.start_ms)
    return words


def words_to_sentences(words: List[Word], pause_threshold_ms: int = 800) -> List[Sentence]:
    """Group words into sentences based on pauses."""
    if not words:
        return []
    
    sentences = []
    current_words = [words[0]]
    
    for i in range(1, len(words)):
        w = words[i]
        prev = words[i-1]
        gap = w.start_ms - prev.end_ms
        
        if gap > pause_threshold_ms:
            if current_words:
                text = ' '.join(cw.text for cw in current_words)
                sentences.append(Sentence(
                    text=text,
                    start_ms=current_words[0].start_ms,
                    end_ms=current_words[-1].end_ms
                ))
            current_words = [w]
        else:
            current_words.append(w)
    
    if current_words:
        text = ' '.join(cw.text for cw in current_words)
        sentences.append(Sentence(
            text=text,
            start_ms=current_words[0].start_ms,
            end_ms=current_words[-1].end_ms
        ))
    
    return sentences


def main():
    print("=" * 70)
    print("SEGMENTATION QUALITY ANALYSIS")
    print("=" * 70)
    
    # Get current segments
    client = chromadb.PersistentClient(path='chroma_data')
    col = client.get_collection('segments')
    results = col.get(include=['metadatas', 'documents'])
    
    # Group by episode
    episodes = {}
    for i, meta in enumerate(results['metadatas']):
        ep = meta.get('episode_title', 'unknown')
        if ep not in episodes:
            episodes[ep] = {'segments': [], 'docs': []}
        episodes[ep]['segments'].append(meta)
        episodes[ep]['docs'].append(results['documents'][i])
    
    print(f"\nTotal episodes: {len(episodes)}")
    print(f"Total segments: {len(results['metadatas'])}")
    
    # Analyze each episode
    all_durations = []
    all_empty_topics = 0
    all_short = 0
    all_long = 0
    
    for ep_name, data in episodes.items():
        segs = data['segments']
        docs = data['docs']
        
        print(f"\n{'='*70}")
        print(f"EPISODE: {ep_name[:60]}...")
        print(f"{'='*70}")
        
        # Calculate durations
        durations = []
        for s in segs:
            start = s.get('start_ms', 0) or 0
            end = s.get('end_ms', 0) or 0
            if end > start:
                durations.append((end - start) / 1000)
        
        if not durations:
            print("  No valid segments")
            continue
        
        all_durations.extend(durations)
        
        # Coverage
        total_duration = max(s.get('end_ms', 0) or 0 for s in segs)
        covered = sum((s.get('end_ms', 0) or 0) - (s.get('start_ms', 0) or 0) for s in segs)
        coverage = covered / total_duration if total_duration > 0 else 0
        
        # Count issues
        empty_topics = sum(1 for s in segs if not s.get('primary_topic'))
        short_segs = sum(1 for d in durations if d < 30)
        long_segs = sum(1 for d in durations if d > 300)
        
        all_empty_topics += empty_topics
        all_short += short_segs
        all_long += long_segs
        
        print(f"\n  Segments: {len(segs)}")
        print(f"  Duration: {total_duration/1000/60:.1f} minutes")
        print(f"  Coverage: {coverage*100:.1f}%")
        print(f"\n  Segment durations:")
        print(f"    Mean: {statistics.mean(durations):.1f}s")
        print(f"    Median: {statistics.median(durations):.1f}s")
        print(f"    Std Dev: {statistics.stdev(durations):.1f}s" if len(durations) > 1 else "")
        print(f"    Range: {min(durations):.1f}s - {max(durations):.1f}s")
        
        print(f"\n  Issues:")
        print(f"    Empty topics: {empty_topics} ({empty_topics/len(segs)*100:.0f}%)")
        print(f"    Too short (<30s): {short_segs} ({short_segs/len(segs)*100:.0f}%)")
        print(f"    Too long (>5min): {long_segs} ({long_segs/len(segs)*100:.0f}%)")
        
        # Sample segments
        print(f"\n  Sample segments:")
        sorted_segs = sorted(zip(segs, docs), key=lambda x: x[0].get('start_ms', 0) or 0)
        for s, doc in sorted_segs[:3]:
            start = (s.get('start_ms', 0) or 0) / 1000
            end = (s.get('end_ms', 0) or 0) / 1000
            topic = s.get('primary_topic', '') or '[NO TOPIC]'
            print(f"    {start:.0f}s-{end:.0f}s ({end-start:.0f}s): {topic[:50]}")
    
    # Overall summary
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    print(f"\n  Total segments: {len(all_durations)}")
    print(f"  Mean duration: {statistics.mean(all_durations):.1f}s")
    print(f"  Median duration: {statistics.median(all_durations):.1f}s")
    print(f"\n  Quality issues:")
    print(f"    Empty topics: {all_empty_topics} ({all_empty_topics/len(all_durations)*100:.0f}%)")
    print(f"    Too short (<30s): {all_short} ({all_short/len(all_durations)*100:.0f}%)")
    print(f"    Too long (>5min): {all_long} ({all_long/len(all_durations)*100:.0f}%)")
    
    # Duration distribution
    print(f"\n  Duration distribution:")
    buckets = {'<30s': 0, '30-60s': 0, '60-120s': 0, '120-180s': 0, '180-300s': 0, '>300s': 0}
    for d in all_durations:
        if d < 30: buckets['<30s'] += 1
        elif d < 60: buckets['30-60s'] += 1
        elif d < 120: buckets['60-120s'] += 1
        elif d < 180: buckets['120-180s'] += 1
        elif d < 300: buckets['180-300s'] += 1
        else: buckets['>300s'] += 1
    
    for k, v in buckets.items():
        bar = '█' * int(v / len(all_durations) * 40)
        print(f"    {k:>10}: {v:3} ({v/len(all_durations)*100:4.0f}%) {bar}")
    
    # Parse VTT files for sentence analysis
    print(f"\n{'='*70}")
    print("SENTENCE-LEVEL ANALYSIS (from VTT)")
    print(f"{'='*70}")
    
    audio_dir = Path('audio')
    for vtt_file in sorted(audio_dir.glob('*.vtt'))[:3]:  # Just first 3
        print(f"\n  {vtt_file.name}:")
        words = parse_vtt(str(vtt_file))
        if not words:
            print("    No words parsed")
            continue
        
        sentences = words_to_sentences(words)
        print(f"    Words: {len(words)}")
        print(f"    Sentences (pause-based): {len(sentences)}")
        
        if sentences:
            sent_durations = [(s.end_ms - s.start_ms)/1000 for s in sentences]
            print(f"    Avg sentence: {statistics.mean(sent_durations):.1f}s")
            print(f"    Sentence range: {min(sent_durations):.1f}s - {max(sent_durations):.1f}s")
            
            # How many sentences per current segment (rough estimate)
            ep_duration = words[-1].end_ms / 1000 / 60  # minutes
            print(f"    Episode duration: {ep_duration:.1f} min")
            
            # If we segmented at every pause, we'd have ~{len(sentences)} segments
            # Current approach gives us ~20-30 segments per episode
            # So current approach groups ~{len(sentences)/25} sentences per segment on average


if __name__ == "__main__":
    main()
