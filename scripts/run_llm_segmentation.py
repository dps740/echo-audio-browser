#!/usr/bin/env python3
"""Run LLM segmentation on episodes that only have raw chunks."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import re
from pathlib import Path

# Setup
INPUT_DIR = Path("/home/ubuntu/clawd/projects/echo-processing/input")
AUDIO_DIR = Path("/home/ubuntu/clawd/projects/echo-audio-browser/audio")

def parse_vtt_to_words(vtt_path: str) -> list:
    """Parse VTT to get word-level timestamps."""
    with open(vtt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    words = []
    # Match timestamp tags: <HH:MM:SS.mmm><c> word</c>
    pattern = r'<(\d{2}:\d{2}:\d{2}\.\d{3})><c>([^<]+)</c>'
    
    def ts_to_ms(ts: str) -> int:
        parts = ts.split(':')
        h, m = int(parts[0]), int(parts[1])
        s_ms = parts[2].split('.')
        s, ms = int(s_ms[0]), int(s_ms[1])
        return h * 3600000 + m * 60000 + s * 1000 + ms
    
    # Also get cue-level for first words
    cue_pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})'
    
    lines = content.split('\n')
    current_cue_start = 0
    
    for line in lines:
        # Check for cue timing
        cue_match = re.match(cue_pattern, line)
        if cue_match:
            current_cue_start = ts_to_ms(cue_match.group(1))
            continue
        
        # Find word timestamps in line
        matches = re.findall(pattern, line)
        
        # First word before any tag
        first_word_match = re.match(r'^([A-Za-z][^\s<]*)', line.strip())
        if first_word_match and matches:
            word = first_word_match.group(1).strip()
            if word:
                words.append({
                    'word': word,
                    'start_ms': current_cue_start,
                    'end_ms': ts_to_ms(matches[0][0]) if matches else current_cue_start + 500
                })
        
        for ts, word in matches:
            word = word.strip()
            if word:
                words.append({
                    'word': word,
                    'start_ms': ts_to_ms(ts),
                    'end_ms': 0  # Will be filled
                })
    
    # Fill end times
    for i in range(len(words) - 1):
        if words[i]['end_ms'] == 0:
            words[i]['end_ms'] = words[i + 1]['start_ms']
    if words and words[-1]['end_ms'] == 0:
        words[-1]['end_ms'] = words[-1]['start_ms'] + 500
    
    return words


def get_transcript_text(words: list) -> str:
    """Get full transcript text from words."""
    return ' '.join(w['word'] for w in words)


async def run_segmentation(episode_id: str, words: list, title: str, channel: str):
    """Run LLM segmentation and store in ChromaDB."""
    from app.services.segmentation import segment_transcript
    from app.services.transcription import TranscriptWord, TranscriptResult
    from app.services.vectordb import get_collection, store_segments
    
    # Convert to TranscriptWord objects
    transcript_words = [
        TranscriptWord(word=w['word'], start_ms=w['start_ms'], end_ms=w['end_ms'], confidence=1.0)
        for w in words
    ]
    
    full_text = get_transcript_text(words)
    total_duration = words[-1]['end_ms'] if words else 0
    
    transcript = TranscriptResult(
        text=full_text,
        words=transcript_words,
        duration_ms=total_duration,
        speakers=[]
    )
    
    print(f"  Running LLM segmentation ({len(words)} words, {total_duration/60000:.1f} min)...")
    
    segments = await segment_transcript(transcript)
    
    print(f"  Got {len(segments)} segments")
    
    if segments:
        # Delete old raw chunks first
        collection = get_collection()
        old_ids = [f"{episode_id}_{i}" for i in range(200)]  # Max 200 chunks
        try:
            collection.delete(ids=old_ids)
            print(f"  Deleted old raw chunks")
        except:
            pass
        
        # Store new segments
        audio_url = f"/audio/{episode_id}.wav"
        await store_segments(
            episode_id=episode_id,
            episode_title=title,
            podcast_title=channel,
            audio_url=audio_url,
            segments=segments
        )
        print(f"  ✅ Stored {len(segments)} LLM segments")
        
        # Print segment summary
        for i, seg in enumerate(segments[:5]):
            dur = (seg.end_ms - seg.start_ms) / 1000
            print(f"    [{i}] {dur:.0f}s: {seg.primary_topic or seg.topic_tags[0] if seg.topic_tags else 'unknown'}")
        if len(segments) > 5:
            print(f"    ... and {len(segments) - 5} more")
    else:
        print(f"  ❌ No segments returned")
    
    return segments


async def main():
    episodes_to_process = ['gXY1kx7zlkk', 'w2BqPnVKVo4', 'wTiHheA40nI']
    
    for ep_id in episodes_to_process:
        ep_dir = INPUT_DIR / ep_id
        if not ep_dir.exists():
            print(f"❌ {ep_id}: directory not found")
            continue
        
        print(f"\n📺 Processing {ep_id}")
        
        # Find VTT
        vtt_files = list(ep_dir.glob('*.vtt'))
        if not vtt_files:
            print(f"  ❌ No VTT file")
            continue
        
        # Load metadata
        metadata = {}
        meta_path = ep_dir / 'metadata.json'
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
        
        title = metadata.get('title', ep_id)
        channel = metadata.get('channel', 'All-In Podcast')
        
        print(f"  Title: {title}")
        print(f"  Parsing VTT...")
        
        words = parse_vtt_to_words(str(vtt_files[0]))
        print(f"  Words: {len(words)}")
        
        if not words:
            print(f"  ❌ No words parsed")
            continue
        
        await run_segmentation(ep_id, words, title, channel)
    
    print("\n=== Done ===")


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
