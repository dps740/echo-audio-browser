#!/usr/bin/env python3
"""
Evaluate v3 segmentation proposal using OpenAI embeddings.
"""

import re
import os
import json
import chromadb
from pathlib import Path
from dataclasses import dataclass
from typing import List
import statistics
import numpy as np

# Load OpenAI API key
from dotenv import load_dotenv
load_dotenv()

import openai
client = openai.OpenAI()


@dataclass
class Sentence:
    text: str
    start_ms: int
    end_ms: int


def parse_vtt(vtt_path: str) -> List[dict]:
    """Parse VTT file with word-level timestamps."""
    words = []
    with open(vtt_path, 'r') as f:
        content = f.read()
    
    lines = content.split('\n')
    
    for line in lines:
        if '-->' not in line and not line.startswith('WEBVTT') and line.strip():
            matches = re.findall(r'<(\d{2}:\d{2}:\d{2}\.\d{3})><c>\s*([^<]+)</c>', line)
            for ts, word in matches:
                h, m, s = ts.split(':')
                ms = int(float(h) * 3600000 + float(m) * 60000 + float(s) * 1000)
                words.append({'text': word.strip(), 'start_ms': ms, 'end_ms': ms + 200})
    
    words.sort(key=lambda w: w['start_ms'])
    return words


def words_to_sentences(words: List[dict], pause_threshold_ms: int = 800) -> List[Sentence]:
    """Group words into sentences based on pauses."""
    if not words:
        return []
    
    sentences = []
    current_words = [words[0]]
    
    for i in range(1, len(words)):
        w = words[i]
        prev = words[i-1]
        gap = w['start_ms'] - prev['end_ms']
        
        if gap > pause_threshold_ms:
            if current_words:
                text = ' '.join(cw['text'] for cw in current_words)
                sentences.append(Sentence(
                    text=text,
                    start_ms=current_words[0]['start_ms'],
                    end_ms=current_words[-1]['end_ms']
                ))
            current_words = [w]
        else:
            current_words.append(w)
    
    if current_words:
        text = ' '.join(cw['text'] for cw in current_words)
        sentences.append(Sentence(
            text=text,
            start_ms=current_words[0]['start_ms'],
            end_ms=current_words[-1]['end_ms']
        ))
    
    return sentences


def get_embeddings(texts: List[str], batch_size: int = 100) -> np.ndarray:
    """Get embeddings from OpenAI."""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        embeddings = [e.embedding for e in response.data]
        all_embeddings.extend(embeddings)
    
    return np.array(all_embeddings)


def find_boundaries_v3(embeddings: np.ndarray, 
                       sharp_threshold: float = 0.65,
                       drift_window: int = 5,
                       drift_threshold: float = 0.5) -> List[int]:
    """Find topic boundaries using v3 algorithm."""
    boundaries = []
    
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_norm = embeddings / norms
    
    for i in range(1, len(embeddings_norm)):
        # Sharp boundary - adjacent similarity
        sim = np.dot(embeddings_norm[i-1], embeddings_norm[i])
        if sim < sharp_threshold:
            boundaries.append(i)
            continue
        
        # Drift boundary - context similarity
        if i >= drift_window:
            context = embeddings_norm[i-drift_window:i].mean(axis=0)
            context = context / np.linalg.norm(context)
            drift_sim = np.dot(context, embeddings_norm[i])
            if drift_sim < drift_threshold:
                boundaries.append(i)
    
    return boundaries


def main():
    print("=" * 70)
    print("V3 SEGMENTATION EVALUATION")
    print("=" * 70)
    
    audio_dir = Path('audio')
    vtt_files = sorted(audio_dir.glob('*.vtt'))[:3]
    
    for vtt_file in vtt_files:
        print(f"\n{'='*70}")
        print(f"FILE: {vtt_file.name}")
        print(f"{'='*70}")
        
        # Parse VTT
        words = parse_vtt(str(vtt_file))
        if not words:
            print("  No words parsed")
            continue
        
        total_duration_ms = words[-1]['end_ms']
        print(f"  Duration: {total_duration_ms/1000/60:.1f} minutes")
        print(f"  Words: {len(words)}")
        
        # Convert to sentences
        sentences = words_to_sentences(words)
        print(f"  Sentences: {len(sentences)}")
        
        if len(sentences) < 10:
            print("  Too few sentences")
            continue
        
        # Get embeddings (sample for cost)
        print(f"  Getting embeddings...")
        texts = [s.text for s in sentences]
        
        # Limit to first 200 sentences for cost
        max_sentences = min(200, len(sentences))
        embeddings = get_embeddings(texts[:max_sentences])
        print(f"  Got {len(embeddings)} embeddings")
        
        # Find boundaries
        boundaries = find_boundaries_v3(embeddings)
        print(f"  Boundaries found: {len(boundaries)}")
        
        # Create segments
        v3_segments = []
        boundary_indices = [0] + boundaries + [max_sentences]
        for i in range(len(boundary_indices) - 1):
            start_idx = boundary_indices[i]
            end_idx = boundary_indices[i+1] - 1
            if end_idx >= start_idx:
                v3_segments.append({
                    'start_ms': sentences[start_idx].start_ms,
                    'end_ms': sentences[end_idx].end_ms,
                })
        
        # Analyze v3 segments
        if v3_segments:
            durations = [(s['end_ms'] - s['start_ms'])/1000 for s in v3_segments]
            covered_ms = sum(s['end_ms'] - s['start_ms'] for s in v3_segments)
            # Coverage of the portion we analyzed
            analyzed_duration = sentences[max_sentences-1].end_ms
            coverage = covered_ms / analyzed_duration if analyzed_duration > 0 else 0
            
            print(f"\n  V3 RESULTS (first {max_sentences} sentences):")
            print(f"    Segments: {len(v3_segments)}")
            print(f"    Coverage: {coverage*100:.1f}%")
            print(f"    Mean duration: {statistics.mean(durations):.1f}s")
            print(f"    Median duration: {statistics.median(durations):.1f}s")
            print(f"    Range: {min(durations):.1f}s - {max(durations):.1f}s")
            
            short = sum(1 for d in durations if d < 30)
            long = sum(1 for d in durations if d > 300)
            print(f"    Too short (<30s): {short} ({short/len(durations)*100:.0f}%)")
            print(f"    Too long (>5min): {long} ({long/len(durations)*100:.0f}%)")
            
            # Sample segments
            print(f"\n    Sample segments:")
            for s in v3_segments[:5]:
                dur = (s['end_ms'] - s['start_ms'])/1000
                print(f"      {s['start_ms']/1000:.0f}s - {s['end_ms']/1000:.0f}s ({dur:.0f}s)")
        
        # Compare to current
        video_id = vtt_file.stem.replace('.en', '')
        client_db = chromadb.PersistentClient(path='chroma_data')
        col = client_db.get_collection('segments')
        
        # Try to find matching episode
        all_results = col.get(include=['metadatas'])
        current_segs = []
        for meta in all_results['metadatas']:
            # Check if this segment is from an episode matching our video_id
            ep_title = meta.get('episode_title', '')
            if video_id in str(meta) or ('ICE' in ep_title and video_id == 'gXY1kx7zlkk') or \
               ('Oz' in ep_title and video_id == 'b5p40OuTTW4') or \
               ('Future' in ep_title and video_id == 'w2BqPnVKVo4'):
                current_segs.append(meta)
        
        if current_segs:
            curr_durations = []
            for s in current_segs:
                start = s.get('start_ms', 0) or 0
                end = s.get('end_ms', 0) or 0
                if end > start:
                    curr_durations.append((end - start)/1000)
            
            # Calculate current coverage
            curr_covered = sum((s.get('end_ms', 0) or 0) - (s.get('start_ms', 0) or 0) for s in current_segs)
            curr_coverage = curr_covered / total_duration_ms if total_duration_ms > 0 else 0
            
            print(f"\n  CURRENT APPROACH:")
            print(f"    Segments: {len(current_segs)}")
            print(f"    Coverage: {curr_coverage*100:.1f}%")
            if curr_durations:
                print(f"    Mean duration: {statistics.mean(curr_durations):.1f}s")
            
            print(f"\n  COMPARISON:")
            print(f"    Segments: {len(current_segs)} → {len(v3_segments)}")
            print(f"    Coverage: {curr_coverage*100:.1f}% → {coverage*100:.1f}%")
    
    print(f"\n{'='*70}")
    print("EVALUATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
