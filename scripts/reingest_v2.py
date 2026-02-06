#!/usr/bin/env python3
"""
Re-ingest all episodes with v2 segmentation:
- Shorter segments (1-3 min target)
- Primary + secondary topics
- Ad/intro/outro filtering
"""

import asyncio
import json
import shutil
import sys
from pathlib import Path
from glob import glob

sys.path.insert(0, str(Path(__file__).parent.parent))

import chromadb
from app.config import settings, get_embedding_function
from app.services.segmentation import segment_transcript, SegmentResult
from app.services.transcription import TranscriptResult, TranscriptWord
from app.services.vectordb import store_segments


def parse_json3_transcript(json3_path: Path) -> TranscriptResult:
    """Parse YouTube json3 caption file into TranscriptResult."""
    with open(json3_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    words = []
    events = data.get('events', [])
    
    for event in events:
        # Skip non-word events
        if 'segs' not in event:
            continue
        
        start_ms = event.get('tStartMs', 0)
        
        for seg in event.get('segs', []):
            text = seg.get('utf8', '').strip()
            if not text or text == '\n':
                continue
            
            # Calculate word timing
            offset = seg.get('tOffsetMs', 0)
            word_start = start_ms + offset
            # Estimate word end (100ms per word or next word start)
            word_end = word_start + 100
            
            words.append(TranscriptWord(
                word=text,
                start_ms=word_start,
                end_ms=word_end,
                confidence=1.0
            ))
    
    # Build full text
    full_text = ' '.join(w.word for w in words)
    duration_ms = words[-1].end_ms if words else 0
    
    return TranscriptResult(
        text=full_text,
        words=words,
        duration_ms=duration_ms,
        speakers=[]
    )


def find_transcript_for_video(video_id: str, audio_dir: Path) -> Path | None:
    """Find the json3 transcript file for a video ID."""
    # Look for files containing the video ID in brackets [video_id]
    for json_file in audio_dir.glob("*.json3"):
        if f"[{video_id}]" in json_file.name:
            return json_file
    
    # Fallback: exact match
    exact = audio_dir / f"{video_id}.json3"
    if exact.exists():
        return exact
    
    return None


async def reingest_episode(video_id: str, audio_dir: Path, dry_run: bool = False):
    """Re-ingest a single episode with new segmentation."""
    print(f"\n{'='*60}")
    print(f"Processing: {video_id}")
    print(f"{'='*60}")
    
    # Find transcript
    transcript_path = find_transcript_for_video(video_id, audio_dir)
    if not transcript_path:
        print(f"  ❌ No transcript found for {video_id}")
        return None
    
    print(f"  📄 Transcript: {transcript_path.name}")
    
    # Parse transcript
    transcript = parse_json3_transcript(transcript_path)
    print(f"  📝 Words: {len(transcript.words)}")
    
    if len(transcript.words) < 100:
        print(f"  ⚠️ Transcript too short, skipping")
        return None
    
    duration_min = transcript.words[-1].end_ms / 60000
    print(f"  ⏱️ Duration: {duration_min:.1f} min")
    
    # Get episode title from transcript filename
    episode_title = transcript_path.stem.split('[')[0].strip() or video_id
    
    if dry_run:
        print(f"  [DRY RUN] Would segment with new v2 prompt")
        return None
    
    # Segment with new prompt
    print(f"  🔄 Running v2 segmentation...")
    try:
        segments = await segment_transcript(transcript, episode_title=episode_title)
        print(f"  ✅ Generated {len(segments)} content segments")
        
        # Show segment stats
        durations = [(s.end_ms - s.start_ms) / 60000 for s in segments]
        if durations:
            print(f"  📊 Segment lengths: min={min(durations):.1f}m, max={max(durations):.1f}m, avg={sum(durations)/len(durations):.1f}m")
        
        # Show sample
        for i, seg in enumerate(segments[:3]):
            print(f"\n  [{i+1}] {seg.start_ms//60000}m-{seg.end_ms//60000}m")
            print(f"      Primary: {seg.primary_topic}")
            print(f"      Secondary: {seg.secondary_topics[:3]}...")
            print(f"      Summary: {seg.summary[:80]}...")
        
        return segments, episode_title
        
    except Exception as e:
        print(f"  ❌ Segmentation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Don't actually segment")
    parser.add_argument("--episode", help="Only process specific video ID")
    parser.add_argument("--limit", type=int, help="Limit number of episodes")
    args = parser.parse_args()
    
    audio_dir = Path(__file__).parent.parent / "audio"
    chroma_path = Path(__file__).parent.parent / "chroma_data"
    
    # Find all MP3 files
    mp3_files = list(audio_dir.glob("*.mp3"))
    video_ids = [f.stem for f in mp3_files]
    
    print(f"Found {len(video_ids)} episodes to process")
    
    if args.episode:
        video_ids = [args.episode]
    if args.limit:
        video_ids = video_ids[:args.limit]
    
    # Nuke existing ChromaDB
    if not args.dry_run:
        print(f"\n🗑️ Nuking existing ChromaDB at {chroma_path}")
        if chroma_path.exists():
            shutil.rmtree(chroma_path)
        chroma_path.mkdir(exist_ok=True)
    
    # Process each episode
    total_segments = 0
    success = 0
    
    for video_id in video_ids:
        result = await reingest_episode(video_id, audio_dir, dry_run=args.dry_run)
        
        if result and not args.dry_run:
            segments, episode_title = result
            
            # Store in ChromaDB
            count = await store_segments(
                segments=segments,
                episode_id=video_id,
                episode_title=episode_title,
                podcast_title="YouTube",
                audio_url=f"/audio/{video_id}.mp3"
            )
            print(f"  💾 Stored {count} segments")
            total_segments += count
            success += 1
    
    print(f"\n{'='*60}")
    print(f"COMPLETE: {success}/{len(video_ids)} episodes, {total_segments} total segments")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
