#!/usr/bin/env python3
"""
Nuke ChromaDB and re-ingest all episodes with improved LLM segmentation.

This script:
1. Deletes the chroma_data folder completely
2. Clears the segments from the database
3. Re-runs LLM segmentation on all episodes
4. Provides diagnostics showing segment quality

Usage:
    python nuke_and_reingest.py
    python nuke_and_reingest.py --dry-run  # See what would happen without doing it
"""

import os
import sys
import shutil
import asyncio
import argparse
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.services.vectordb import VectorDB
from app.services.segmentation import segment_transcript, SegmentResult
from app.services.transcription import TranscriptResult, TranscriptWord


def print_header(msg: str):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}\n")


def print_segment_diagnostics(segments: list[SegmentResult], episode_title: str):
    """Print detailed diagnostics about segment quality."""
    print(f"\n📊 SEGMENT DIAGNOSTICS: {episode_title[:50]}...")
    print("-" * 50)
    
    if not segments:
        print("  ❌ NO SEGMENTS GENERATED!")
        return
    
    # Collect stats
    summary_lengths = [len(s.summary) for s in segments]
    tag_counts = [len(s.topic_tags) for s in segments]
    all_tags = [tag for s in segments for tag in s.topic_tags]
    densities = [s.density_score for s in segments]
    durations = [(s.end_ms - s.start_ms) / 60000 for s in segments]  # minutes
    
    # Quality checks
    generic_tags = ["AI", "technology", "business", "science", "philosophy", 
                   "discussion", "conversation", "interview", "podcast"]
    bad_tags = [t for t in all_tags if t.lower() in [g.lower() for g in generic_tags]]
    specific_tags = [t for t in all_tags if t not in bad_tags and len(t.split()) >= 2]
    
    short_summaries = [s for s in segments if len(s.summary) < 100]
    
    print(f"  Segments: {len(segments)}")
    print(f"  Total duration covered: {sum(durations):.1f} min")
    print(f"  Avg segment duration: {sum(durations)/len(durations):.1f} min")
    print()
    
    # Summary quality
    avg_summary_len = sum(summary_lengths) / len(summary_lengths)
    print(f"  📝 SUMMARY QUALITY:")
    print(f"     Avg length: {avg_summary_len:.0f} chars")
    if avg_summary_len < 150:
        print(f"     ⚠️  WARNING: Summaries may be too short (target: 150+ chars)")
    else:
        print(f"     ✅ Good summary length")
    print(f"     Short summaries (<100 chars): {len(short_summaries)}")
    print()
    
    # Tag quality
    print(f"  🏷️  TAG QUALITY:")
    print(f"     Total tags: {len(all_tags)}")
    print(f"     Unique tags: {len(set(all_tags))}")
    print(f"     Avg tags/segment: {sum(tag_counts)/len(tag_counts):.1f}")
    if bad_tags:
        print(f"     ⚠️  Generic tags found: {bad_tags[:5]}{'...' if len(bad_tags) > 5 else ''}")
    if specific_tags:
        print(f"     ✅ Specific tags: {specific_tags[:5]}{'...' if len(specific_tags) > 5 else ''}")
    else:
        print(f"     ⚠️  WARNING: No specific tags found!")
    print()
    
    # Sample segments
    print(f"  📋 SAMPLE SEGMENTS:")
    for i, seg in enumerate(segments[:3]):
        print(f"\n     [{i+1}] {seg.start_ms//60000}m - {seg.end_ms//60000}m (density: {seg.density_score:.2f})")
        print(f"         Summary: {seg.summary[:120]}{'...' if len(seg.summary) > 120 else ''}")
        print(f"         Tags: {seg.topic_tags}")


async def reingest_episode(db: VectorDB, episode_id: str, dry_run: bool = False):
    """Re-ingest a single episode with LLM segmentation."""
    # Get episode data from DB
    episode = db.get_episode(episode_id)
    if not episode:
        print(f"  ⚠️  Episode {episode_id} not found in DB")
        return None
    
    title = episode.get("title", "Unknown")
    print(f"\n  Processing: {title[:60]}...")
    
    # Get full transcript
    transcript_data = db.get_transcript(episode_id)
    if not transcript_data or not transcript_data.get("words"):
        print(f"    ❌ No transcript found for episode {episode_id}")
        return None
    
    # Convert to TranscriptResult
    words = [
        TranscriptWord(
            word=w["word"],
            start_ms=w["start_ms"],
            end_ms=w["end_ms"],
            confidence=w.get("confidence", 1.0)
        )
        for w in transcript_data["words"]
    ]
    transcript = TranscriptResult(words=words, source=transcript_data.get("source", "unknown"))
    
    print(f"    Transcript: {len(words)} words, {words[-1].end_ms // 60000} minutes")
    
    if dry_run:
        print(f"    [DRY RUN] Would run LLM segmentation here")
        return None
    
    # Run LLM segmentation
    print(f"    Running LLM segmentation...")
    try:
        segments = await segment_transcript(transcript, episode_title=title)
        print(f"    ✅ Generated {len(segments)} segments")
        
        # Print diagnostics
        print_segment_diagnostics(segments, title)
        
        return segments
    except Exception as e:
        print(f"    ❌ Segmentation failed: {e}")
        return None


async def main():
    parser = argparse.ArgumentParser(description="Nuke ChromaDB and re-ingest with improved LLM segmentation")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without doing it")
    parser.add_argument("--keep-chroma", action="store_true", help="Don't delete ChromaDB (just re-segment)")
    parser.add_argument("--episode", help="Only re-ingest specific episode ID")
    args = parser.parse_args()
    
    print_header("ECHO AUDIO BROWSER - NUKE & RE-INGEST")
    
    chroma_path = Path(settings.chroma_path if hasattr(settings, 'chroma_path') else "./chroma_data")
    
    # Step 1: Nuke ChromaDB
    if not args.keep_chroma:
        print("STEP 1: Nuking ChromaDB")
        print(f"  Path: {chroma_path}")
        
        if chroma_path.exists():
            if args.dry_run:
                print(f"  [DRY RUN] Would delete {chroma_path}")
            else:
                shutil.rmtree(chroma_path)
                print(f"  ✅ Deleted {chroma_path}")
        else:
            print(f"  (ChromaDB folder doesn't exist)")
    else:
        print("STEP 1: Keeping existing ChromaDB (--keep-chroma)")
    
    # Step 2: Initialize fresh DB
    print("\nSTEP 2: Initializing database")
    db = VectorDB(str(chroma_path))
    
    # Step 3: Get all episodes
    print("\nSTEP 3: Getting episodes to re-ingest")
    if args.episode:
        episode_ids = [args.episode]
        print(f"  Single episode: {args.episode}")
    else:
        episodes = db.list_episodes()
        episode_ids = [e["id"] for e in episodes]
        print(f"  Found {len(episode_ids)} episodes")
    
    if not episode_ids:
        print("\n  ⚠️  No episodes found! Have you ingested any content yet?")
        print("     Use the web UI to add podcasts/videos first.")
        return
    
    # Step 4: Re-ingest each episode
    print_header(f"STEP 4: Re-ingesting {len(episode_ids)} episodes with LLM segmentation")
    
    success_count = 0
    fail_count = 0
    total_segments = 0
    
    for i, episode_id in enumerate(episode_ids, 1):
        print(f"\n[{i}/{len(episode_ids)}] Episode: {episode_id}")
        
        segments = await reingest_episode(db, episode_id, dry_run=args.dry_run)
        
        if segments:
            success_count += 1
            total_segments += len(segments)
            
            if not args.dry_run:
                # Store segments in ChromaDB
                episode = db.get_episode(episode_id)
                db.store_segments(
                    episode_id=episode_id,
                    episode_title=episode.get("title", "Unknown"),
                    podcast_title=episode.get("podcast", "Unknown"),
                    audio_url=episode.get("audio_url", ""),
                    segments=segments
                )
                print(f"    ✅ Stored {len(segments)} segments in ChromaDB")
        else:
            fail_count += 1
    
    # Final summary
    print_header("SUMMARY")
    print(f"  Episodes processed: {len(episode_ids)}")
    print(f"  Successful: {success_count}")
    print(f"  Failed: {fail_count}")
    print(f"  Total segments created: {total_segments}")
    
    if args.dry_run:
        print("\n  [DRY RUN] No changes were made. Run without --dry-run to apply.")
    else:
        print("\n  ✅ Re-ingestion complete!")
        print("     You can now search the improved segments in the UI.")


if __name__ == "__main__":
    asyncio.run(main())
