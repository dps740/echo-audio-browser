#!/usr/bin/env python3
"""
Ingest local audio files using Whisper transcription.
Bypasses YouTube entirely - works with any audio files.

Usage:
    python ingest_local_whisper.py           # Process all mp3 files in app/audio/
    python ingest_local_whisper.py file.mp3  # Process specific file
"""

import os
import sys
import json
import subprocess
import asyncio
from pathlib import Path
from datetime import datetime

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config import settings
from app.services.segmentation import segment_transcript, SegmentResult
from app.services.transcription import TranscriptResult, TranscriptWord

import chromadb
from chromadb.utils import embedding_functions


def get_collection():
    """Get or create ChromaDB collection with embedding function."""
    client = chromadb.PersistentClient(path="./chroma_data")
    
    # Use OpenAI embeddings
    embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
        api_key=settings.openai_api_key,
        model_name="text-embedding-3-small"
    )
    
    return client.get_or_create_collection(
        name="segments",
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"}
    )


def transcribe_with_whisper(audio_path: str, model: str = "base") -> list:
    """Transcribe audio file with Whisper and get word-level timestamps."""
    print(f"  Transcribing with Whisper ({model})...")
    
    # Use whisper CLI with JSON output
    output_dir = "/tmp/whisper_output"
    os.makedirs(output_dir, exist_ok=True)
    
    cmd = [
        "whisper", audio_path,
        "--model", model,
        "--output_format", "json",
        "--output_dir", output_dir,
        "--word_timestamps", "True",
        "--language", "en"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    
    if result.returncode != 0:
        print(f"  Whisper error: {result.stderr[:500]}")
        return []
    
    # Find output JSON
    base_name = Path(audio_path).stem
    json_path = Path(output_dir) / f"{base_name}.json"
    
    if not json_path.exists():
        print(f"  No JSON output found at {json_path}")
        return []
    
    with open(json_path) as f:
        data = json.load(f)
    
    # Extract word-level data
    words = []
    for segment in data.get("segments", []):
        for word_data in segment.get("words", []):
            words.append({
                "word": word_data.get("word", "").strip(),
                "start_ms": int(word_data.get("start", 0) * 1000),
                "end_ms": int(word_data.get("end", 0) * 1000),
                "confidence": word_data.get("probability", 1.0)
            })
    
    print(f"  Got {len(words)} words from transcript")
    return words


def store_segments(episode_id: str, title: str, podcast: str, audio_url: str, 
                   segments: list[SegmentResult]):
    """Store segments in ChromaDB."""
    collection = get_collection()
    
    ids = []
    documents = []
    metadatas = []
    
    for i, seg in enumerate(segments):
        seg_id = f"{episode_id}_seg_{i}"
        
        ids.append(seg_id)
        documents.append(seg.transcript_text)
        metadatas.append({
            "segment_id": seg_id,
            "episode_id": episode_id,
            "episode_title": title,
            "podcast_title": podcast,
            "audio_url": audio_url,
            "start_ms": seg.start_ms,
            "end_ms": seg.end_ms,
            "summary": seg.summary,
            "topic_tags": ",".join(seg.topic_tags),
            "density_score": seg.density_score,
            "source": "whisper"
        })
    
    if ids:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
    
    return len(ids)


async def process_file(audio_path: str, podcast_name: str = "Local Audio"):
    """Process a single audio file."""
    filename = os.path.basename(audio_path)
    episode_id = Path(filename).stem
    
    print(f"\n{'='*60}")
    print(f"Processing: {filename}")
    print(f"{'='*60}")
    
    # Check if already ingested
    collection = get_collection()
    existing = collection.get(where={"episode_id": episode_id}, include=["metadatas"])
    if existing and existing["ids"]:
        print(f"  Already ingested ({len(existing['ids'])} segments). Skipping.")
        return 0
    
    # Transcribe with Whisper
    words_data = transcribe_with_whisper(audio_path, model="base")
    
    if not words_data:
        print("  No transcript available. Skipping.")
        return 0
    
    # Convert to TranscriptResult
    words = [
        TranscriptWord(
            word=w["word"],
            start_ms=w["start_ms"],
            end_ms=w["end_ms"],
            confidence=w.get("confidence", 1.0)
        )
        for w in words_data
    ]
    transcript = TranscriptResult(words=words, source="whisper")
    
    # Get duration
    duration_min = words[-1].end_ms // 60000 if words else 0
    print(f"  Duration: {duration_min} minutes")
    
    # Run LLM segmentation
    print(f"  Running LLM segmentation...")
    try:
        segments = await segment_transcript(transcript, episode_title=filename)
        print(f"  Generated {len(segments)} segments")
        
        # Print sample
        if segments:
            print(f"\n  Sample segment:")
            print(f"    Summary: {segments[0].summary[:100]}...")
            print(f"    Tags: {segments[0].topic_tags}")
            print(f"    Density: {segments[0].density_score}")
    except Exception as e:
        print(f"  Segmentation failed: {e}")
        return 0
    
    # Store in ChromaDB
    print(f"  Storing in ChromaDB...")
    count = store_segments(
        episode_id=episode_id,
        title=filename,
        podcast=podcast_name,
        audio_url=f"/audio/{filename}",
        segments=segments
    )
    
    print(f"  ✅ Stored {count} segments")
    return count


async def main():
    """Main entry point."""
    audio_dir = Path(__file__).parent.parent / "app" / "audio"
    
    if len(sys.argv) > 1:
        # Process specific file
        files = [sys.argv[1]]
    else:
        # Process all MP3s in audio directory
        files = sorted(audio_dir.glob("*.mp3"))
    
    print(f"\n{'='*60}")
    print(f"LOCAL AUDIO INGESTION (Whisper + LLM Segmentation)")
    print(f"{'='*60}")
    print(f"Found {len(files)} files to process")
    print(f"Audio directory: {audio_dir}")
    
    total_segments = 0
    processed = 0
    failed = 0
    
    for f in files:
        file_path = str(f) if isinstance(f, Path) else f
        if not os.path.isabs(file_path):
            file_path = str(audio_dir / file_path)
        
        try:
            count = await process_file(file_path)
            if count > 0:
                processed += 1
                total_segments += count
            else:
                failed += 1
        except Exception as e:
            print(f"  ❌ Error: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Processed: {processed}")
    print(f"Failed: {failed}")
    print(f"Total segments: {total_segments}")


if __name__ == "__main__":
    asyncio.run(main())
