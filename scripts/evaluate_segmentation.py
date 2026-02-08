#!/usr/bin/env python3
"""
Evaluate current segmentation vs proposed v3 approach.
"""

import re
import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import chromadb

# Try to import sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    HAS_EMBEDDINGS = True
except ImportError:
    HAS_EMBEDDINGS = False
    print("Warning: sentence-transformers not installed, will skip embedding-based analysis")


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
    words: List[Word]


@dataclass
class Segment:
    start_ms: int
    end_ms: int
    topic: str
    source: str  # 'current' or 'proposed'


def parse_vtt(vtt_path: str) -> List[Word]:
    """Parse VTT file with word-level timestamps."""
    words = []
    with open(vtt_path, 'r') as f:
        content = f.read()
    
    # Pattern to match word timestamps like: word<00:00:01.234>
    # VTT format: <timestamp><c> word</c>
    lines = content.split('\n')
    
    for line in lines:
        if '-->' not in line and not line.startswith('WEBVTT') and line.strip():
            # Extract words with timestamps
            # Pattern: word<timestamp> or <timestamp><c> word</c>
            matches = re.findall(r'(\w+)<(\d{2}:\d{2}:\d{2}\.\d{3})>', line)
            for word, ts in matches:
                h, m, s = ts.split(':')
                ms = int(float(h) * 3600000 + float(m) * 60000 + float(s) * 1000)
                words.append(Word(text=word, start_ms=ms, end_ms=ms + 200))  # Estimate end
            
            # Also try: <timestamp><c> word</c> pattern
            matches2 = re.findall(r'<(\d{2}:\d{2}:\d{2}\.\d{3})><c>\s*(\w+)</c>', line)
            for ts, word in matches2:
                h, m, s = ts.split(':')
                ms = int(float(h) * 3600000 + float(m) * 60000 + float(s) * 1000)
                words.append(Word(text=word, start_ms=ms, end_ms=ms + 200))
    
    # Sort by timestamp and dedupe
    words.sort(key=lambda w: w.start_ms)
    deduped = []
    for w in words:
        if not deduped or w.start_ms != deduped[-1].start_ms:
            deduped.append(w)
    
    return deduped


def words_to_sentences(words: List[Word], pause_threshold_ms: int = 800) -> List[Sentence]:
    """Group words into sentences based on pauses and punctuation."""
    if not words:
        return []
    
    sentences = []
    current_words = [words[0]]
    
    for i in range(1, len(words)):
        w = words[i]
        prev = words[i-1]
        
        # Check for pause
        gap = w.start_ms - prev.end_ms
        
        # Check for sentence-ending punctuation in previous word
        ends_sentence = prev.text.rstrip().endswith(('.', '!', '?'))
        
        if gap > pause_threshold_ms or ends_sentence:
            # End current sentence
            if current_words:
                text = ' '.join(cw.text for cw in current_words)
                sentences.append(Sentence(
                    text=text,
                    start_ms=current_words[0].start_ms,
                    end_ms=current_words[-1].end_ms,
                    words=current_words
                ))
            current_words = [w]
        else:
            current_words.append(w)
    
    # Don't forget last sentence
    if current_words:
        text = ' '.join(cw.text for cw in current_words)
        sentences.append(Sentence(
            text=text,
            start_ms=current_words[0].start_ms,
            end_ms=current_words[-1].end_ms,
            words=current_words
        ))
    
    return sentences


def find_boundaries_v3(embeddings: np.ndarray, 
                       sharp_threshold: float = 0.65,
                       drift_window: int = 5,
                       drift_threshold: float = 0.5) -> List[int]:
    """
    Find topic boundaries using v3 algorithm:
    - Sharp transitions: adjacent sentence similarity < sharp_threshold
    - Drift detection: sentence differs from rolling context
    """
    boundaries = []
    
    for i in range(1, len(embeddings)):
        # Sharp boundary
        sim = np.dot(embeddings[i-1], embeddings[i])
        if sim < sharp_threshold:
            boundaries.append(i)
            continue
        
        # Drift boundary
        if i >= drift_window:
            context = embeddings[i-drift_window:i].mean(axis=0)
            context_norm = context / np.linalg.norm(context)
            drift_sim = np.dot(context_norm, embeddings[i])
            if drift_sim < drift_threshold:
                boundaries.append(i)
    
    return boundaries


def get_current_segments(episode_title: str) -> List[Segment]:
    """Get current segments from ChromaDB."""
    client = chromadb.PersistentClient(path='chroma_data')
    col = client.get_collection('segments')
    
    results = col.get(
        where={"episode_title": episode_title},
        include=['metadatas']
    )
    
    segments = []
    for meta in results['metadatas']:
        segments.append(Segment(
            start_ms=meta.get('start_ms', 0) or 0,
            end_ms=meta.get('end_ms', 0) or 0,
            topic=meta.get('primary_topic', '') or '',
            source='current'
        ))
    
    segments.sort(key=lambda s: s.start_ms)
    return segments


def analyze_segments(segments: List[Segment], total_duration_ms: int) -> dict:
    """Analyze segment quality metrics."""
    if not segments:
        return {'count': 0}
    
    durations = [(s.end_ms - s.start_ms) / 1000 for s in segments]
    
    # Coverage
    covered_ms = sum(s.end_ms - s.start_ms for s in segments)
    coverage = covered_ms / total_duration_ms if total_duration_ms > 0 else 0
    
    # Check for gaps
    gaps = []
    sorted_segs = sorted(segments, key=lambda s: s.start_ms)
    for i in range(1, len(sorted_segs)):
        gap = sorted_segs[i].start_ms - sorted_segs[i-1].end_ms
        if gap > 1000:  # >1s gap
            gaps.append(gap / 1000)
    
    # Check for overlaps
    overlaps = []
    for i in range(1, len(sorted_segs)):
        overlap = sorted_segs[i-1].end_ms - sorted_segs[i].start_ms
        if overlap > 0:
            overlaps.append(overlap / 1000)
    
    return {
        'count': len(segments),
        'mean_duration': np.mean(durations),
        'median_duration': np.median(durations),
        'std_duration': np.std(durations),
        'min_duration': min(durations),
        'max_duration': max(durations),
        'coverage': coverage,
        'num_gaps': len(gaps),
        'total_gap_seconds': sum(gaps) if gaps else 0,
        'num_overlaps': len(overlaps),
        'short_segments': sum(1 for d in durations if d < 30),
        'long_segments': sum(1 for d in durations if d > 300),
        'empty_topics': sum(1 for s in segments if not s.topic),
    }


def main():
    print("=" * 60)
    print("SEGMENTATION EVALUATION: Current vs Proposed v3")
    print("=" * 60)
    
    # Find episodes with VTT files
    audio_dir = Path('audio')
    vtt_files = list(audio_dir.glob('*.vtt'))
    
    print(f"\nFound {len(vtt_files)} VTT files")
    
    # Get current segments from ChromaDB
    client = chromadb.PersistentClient(path='chroma_data')
    col = client.get_collection('segments')
    all_results = col.get(include=['metadatas'])
    
    # Group by episode
    episodes = {}
    for meta in all_results['metadatas']:
        ep = meta.get('episode_title', 'unknown')
        if ep not in episodes:
            episodes[ep] = []
        episodes[ep].append(meta)
    
    print(f"Found {len(episodes)} episodes in ChromaDB\n")
    
    # Load embedding model if available
    model = None
    if HAS_EMBEDDINGS:
        print("Loading embedding model (BGE-base)...")
        try:
            model = SentenceTransformer('BAAI/bge-base-en-v1.5')
            print("Model loaded.\n")
        except Exception as e:
            print(f"Failed to load model: {e}\n")
    
    # Analyze each episode
    for vtt_file in vtt_files:
        video_id = vtt_file.stem.replace('.en', '')
        print(f"\n{'='*60}")
        print(f"VTT: {vtt_file.name}")
        
        # Parse VTT
        words = parse_vtt(str(vtt_file))
        print(f"  Words parsed: {len(words)}")
        
        if not words:
            print("  Skipping - no words parsed")
            continue
        
        total_duration_ms = words[-1].end_ms
        print(f"  Duration: {total_duration_ms/1000/60:.1f} minutes")
        
        # Convert to sentences
        sentences = words_to_sentences(words)
        print(f"  Sentences: {len(sentences)}")
        
        if sentences:
            durations = [(s.end_ms - s.start_ms)/1000 for s in sentences]
            print(f"  Avg sentence: {np.mean(durations):.1f}s")
        
        # Find matching episode in ChromaDB
        matching_ep = None
        for ep_name in episodes.keys():
            if video_id in ep_name or any(video_id in str(m.get('video_id', '')) for m in episodes[ep_name]):
                matching_ep = ep_name
                break
        
        if not matching_ep:
            # Try to match by partial name
            for ep_name in episodes.keys():
                if 'ICE' in ep_name and 'gXY1kx7zlkk' == video_id:
                    matching_ep = ep_name
                    break
                if 'Oz' in ep_name and 'b5p40OuTTW4' == video_id:
                    matching_ep = ep_name
                    break
        
        if matching_ep:
            print(f"\n  Matched episode: {matching_ep[:50]}...")
            current_segs = get_current_segments(matching_ep)
            current_stats = analyze_segments(current_segs, total_duration_ms)
            
            print(f"\n  CURRENT SEGMENTATION:")
            print(f"    Segments: {current_stats['count']}")
            print(f"    Mean duration: {current_stats['mean_duration']:.1f}s")
            print(f"    Coverage: {current_stats['coverage']*100:.1f}%")
            print(f"    Short (<30s): {current_stats['short_segments']}")
            print(f"    Long (>5min): {current_stats['long_segments']}")
            print(f"    Empty topics: {current_stats['empty_topics']}")
            print(f"    Gaps: {current_stats['num_gaps']} ({current_stats['total_gap_seconds']:.0f}s total)")
        else:
            print(f"  No matching episode in ChromaDB")
            current_stats = None
        
        # Run proposed v3 segmentation
        if model and len(sentences) > 10:
            print(f"\n  PROPOSED v3 SEGMENTATION:")
            
            # Embed sentences
            texts = [s.text for s in sentences]
            embeddings = model.encode(texts, normalize_embeddings=True)
            
            # Find boundaries
            boundaries = find_boundaries_v3(embeddings)
            print(f"    Boundaries found: {len(boundaries)}")
            
            # Create segments from boundaries
            proposed_segs = []
            boundary_indices = [0] + boundaries + [len(sentences)]
            for i in range(len(boundary_indices) - 1):
                start_idx = boundary_indices[i]
                end_idx = boundary_indices[i+1] - 1
                if end_idx >= start_idx:
                    proposed_segs.append(Segment(
                        start_ms=sentences[start_idx].start_ms,
                        end_ms=sentences[end_idx].end_ms,
                        topic='',  # Would need LLM for topics
                        source='proposed'
                    ))
            
            proposed_stats = analyze_segments(proposed_segs, total_duration_ms)
            
            print(f"    Segments: {proposed_stats['count']}")
            print(f"    Mean duration: {proposed_stats['mean_duration']:.1f}s")
            print(f"    Coverage: {proposed_stats['coverage']*100:.1f}%")
            print(f"    Short (<30s): {proposed_stats['short_segments']}")
            print(f"    Long (>5min): {proposed_stats['long_segments']}")
            
            # Compare
            if current_stats and current_stats['count'] > 0:
                print(f"\n  COMPARISON:")
                print(f"    Segments: {current_stats['count']} → {proposed_stats['count']} ({proposed_stats['count'] - current_stats['count']:+d})")
                print(f"    Coverage: {current_stats['coverage']*100:.1f}% → {proposed_stats['coverage']*100:.1f}%")
                print(f"    Mean dur: {current_stats['mean_duration']:.1f}s → {proposed_stats['mean_duration']:.1f}s")
                
                # Analyze boundary similarity
                current_boundaries = set()
                for s in current_segs:
                    current_boundaries.add(s.start_ms // 10000)  # 10s buckets
                    current_boundaries.add(s.end_ms // 10000)
                
                proposed_boundaries = set()
                for s in proposed_segs:
                    proposed_boundaries.add(s.start_ms // 10000)
                    proposed_boundaries.add(s.end_ms // 10000)
                
                overlap = len(current_boundaries & proposed_boundaries)
                union = len(current_boundaries | proposed_boundaries)
                jaccard = overlap / union if union > 0 else 0
                print(f"    Boundary agreement (Jaccard): {jaccard:.2f}")
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
