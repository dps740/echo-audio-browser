#!/usr/bin/env python3
"""
Test script: Clear and re-ingest 3 test episodes with LLM segmentation.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import re
import chromadb
from app.config import settings, get_embedding_function
from app.services.segmentation import segment_transcript, resolve_timestamps
from app.services.transcription import TranscriptResult, TranscriptWord

# Test episodes
TEST_EPISODES = [
    {"id": "gXY1kx7zlkk", "title": "ICE Chaos", "podcast": "All-In Podcast"},
    {"id": "b5p40OuTTW4", "title": "Dr Oz Episode", "podcast": "All-In Podcast"},
    {"id": "w2BqPnVKVo4", "title": "Future of Everything", "podcast": "All-In Podcast"},
]


def parse_vtt(vtt_path: str) -> list:
    """Parse VTT file to segments with start_ms, end_ms, text.
    
    Handles YouTube auto-caption format with:
    - position alignment info after timestamps
    - embedded <c> tags for word timing
    - duplicate lines (deduped by checking for substantive content)
    """
    with open(vtt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    segments = []
    
    # Pattern: timestamp line with optional alignment info, then text
    # YouTube format: 00:00:00.000 --> 00:00:01.590 align:start position:0%
    pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})[^\n]*\n([^\n]+(?:\n[^\n]+)*?)(?=\n\n|\n\d{2}:\d{2}|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    def to_ms(t):
        h, m, rest = t.split(':')
        s, ms = rest.split('.')
        return int(h)*3600000 + int(m)*60000 + int(s)*1000 + int(ms)
    
    seen_texts = set()
    
    for start, end, text in matches:
        # Remove HTML-like tags (<c>, timing tags like <00:00:00.000>)
        clean_text = re.sub(r'<[^>]+>', '', text)
        # Remove extra whitespace and newlines
        clean_text = ' '.join(clean_text.split())
        
        # Skip empty or whitespace-only
        if not clean_text or clean_text.isspace():
            continue
        
        # Skip duplicates (YouTube VTT has rolling captions)
        if clean_text in seen_texts:
            continue
        seen_texts.add(clean_text)
        
        segments.append({
            "start_ms": to_ms(start),
            "end_ms": to_ms(end),
            "text": clean_text
        })
    
    print(f"  [VTT Parser] Found {len(matches)} raw matches, {len(segments)} unique segments")
    return segments


def segments_to_words(segments: list) -> list:
    """Convert VTT segments to individual word tokens."""
    words = []
    for seg in segments:
        text = seg["text"]
        text = text.replace("&gt;", ">").replace("&lt;", "<")
        text = text.replace(">>", "").replace("> >", "")
        
        seg_words = text.split()
        if not seg_words:
            continue
        
        duration = seg["end_ms"] - seg["start_ms"]
        word_duration = duration / len(seg_words) if seg_words else duration
        
        for i, word in enumerate(seg_words):
            word_start = seg["start_ms"] + int(i * word_duration)
            word_end = seg["start_ms"] + int((i + 1) * word_duration)
            words.append(TranscriptWord(
                word=word,
                start_ms=word_start,
                end_ms=word_end,
                confidence=1.0
            ))
    return words


def extract_key_terms(transcript: str, summary: str) -> list:
    """Extract key terms using GPT-4o-mini."""
    import openai
    
    if not settings.openai_api_key:
        return []
    
    try:
        client = openai.OpenAI(api_key=settings.openai_api_key)
        
        prompt = f"""Extract 3-7 key terms, entities, and concepts from this podcast segment.
Focus on: named entities, technical terms, topics discussed, important concepts.
Return as comma-separated list.

Summary: {summary}

Transcript excerpt: {transcript[:1000]}

Key terms:"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100,
        )
        
        terms_text = response.choices[0].message.content.strip()
        terms = [t.strip() for t in terms_text.split(",") if t.strip()]
        return terms[:10]
    except Exception as e:
        print(f"  Key term extraction failed: {e}")
        return []


async def process_episode(episode: dict, collection):
    """Process a single episode: parse VTT, segment, store."""
    video_id = episode["id"]
    title = episode["title"]
    podcast = episode["podcast"]
    
    audio_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio")
    vtt_path = os.path.join(audio_dir, f"{video_id}.en.vtt")
    
    if not os.path.exists(vtt_path):
        print(f"  ERROR: VTT file not found: {vtt_path}")
        return {"id": video_id, "error": "VTT not found", "segments": 0}
    
    print(f"\n  Parsing VTT: {vtt_path}")
    vtt_segments = parse_vtt(vtt_path)
    print(f"  Found {len(vtt_segments)} VTT segments")
    
    # Convert to word-level tokens
    words = segments_to_words(vtt_segments)
    print(f"  Converted to {len(words)} word tokens")
    
    # Create transcript
    total_duration_ms = max([s["end_ms"] for s in vtt_segments]) if vtt_segments else 0
    transcript = TranscriptResult(
        text=" ".join([s["text"] for s in vtt_segments]),
        words=words,
        duration_ms=total_duration_ms,
        speakers=[]
    )
    
    # Run LLM segmentation
    print(f"  Running LLM segmentation...")
    try:
        llm_segments = await segment_transcript(transcript, title)
        print(f"  LLM returned {len(llm_segments)} segments")
    except Exception as e:
        print(f"  ERROR: LLM segmentation failed: {e}")
        return {"id": video_id, "error": str(e), "segments": 0}
    
    # Analyze timestamp resolution
    resolved = 0
    failed = 0
    sample_segments = []
    
    ids, docs, metas = [], [], []
    for i, seg in enumerate(llm_segments):
        if seg.start_ms is not None and seg.end_ms is not None and seg.end_ms > seg.start_ms:
            resolved += 1
            duration_sec = (seg.end_ms - seg.start_ms) / 1000
            
            # Collect sample segments
            if len(sample_segments) < 3:
                sample_segments.append({
                    "topic": seg.primary_topic,
                    "timestamp": f"{seg.start_ms // 60000}:{(seg.start_ms % 60000) // 1000:02d}",
                    "duration": f"{duration_sec:.1f}s",
                    "summary": seg.summary[:100] + "..." if len(seg.summary) > 100 else seg.summary
                })
            
            # Extract key terms
            key_terms = extract_key_terms(seg.transcript_text, seg.summary)
            
            # Build enriched doc
            enriched_doc = f"""TOPIC: {', '.join(seg.topic_tags)}
SUMMARY: {seg.summary}
KEY TERMS: {', '.join(key_terms)}
TRANSCRIPT: {seg.transcript_text[:2000]}"""
            
            ids.append(f"{video_id}_{i}")
            docs.append(enriched_doc)
            metas.append({
                "episode_id": video_id,
                "episode_title": title,
                "podcast_title": podcast,
                "audio_url": f"/audio/{video_id}.mp3",
                "start_ms": seg.start_ms,
                "end_ms": seg.end_ms,
                "summary": seg.summary,
                "topic_tags": ",".join(seg.topic_tags),
                "key_terms": ",".join(key_terms),
                "density_score": seg.density_score,
                "source": "youtube",
            })
        else:
            failed += 1
    
    # Store in ChromaDB
    if ids:
        print(f"  Storing {len(ids)} segments in ChromaDB...")
        batch_size = 50
        for j in range(0, len(ids), batch_size):
            collection.upsert(
                ids=ids[j:j+batch_size],
                documents=docs[j:j+batch_size],
                metadatas=metas[j:j+batch_size]
            )
    
    resolution_rate = (resolved / len(llm_segments) * 100) if llm_segments else 0
    print(f"  Result: {resolved} resolved, {failed} failed ({resolution_rate:.1f}% success)")
    
    return {
        "id": video_id,
        "title": title,
        "segments": len(ids),
        "resolved": resolved,
        "failed": failed,
        "resolution_rate": resolution_rate,
        "sample_segments": sample_segments
    }


async def main():
    print("=" * 60)
    print("ECHO AUDIO BROWSER - Test Re-ingestion")
    print("=" * 60)
    
    # Connect to ChromaDB
    client = chromadb.PersistentClient(path="./chroma_data")
    embedding_fn = get_embedding_function()
    collection = client.get_or_create_collection("segments", embedding_function=embedding_fn)
    
    # Step 1: Clear old segments for test episodes
    print("\n1. Clearing old segments for test episodes...")
    for ep in TEST_EPISODES:
        try:
            results = collection.get(where={"episode_id": ep["id"]}, include=["metadatas"])
            if results["ids"]:
                print(f"   Deleting {len(results['ids'])} segments for {ep['id']}")
                collection.delete(ids=results["ids"])
            else:
                print(f"   No existing segments for {ep['id']}")
        except Exception as e:
            print(f"   Error clearing {ep['id']}: {e}")
    
    # Step 2: Re-ingest each episode
    print("\n2. Re-ingesting episodes with LLM segmentation...")
    results = []
    for ep in TEST_EPISODES:
        print(f"\n  Processing: {ep['title']} ({ep['id']})")
        result = await process_episode(ep, collection)
        results.append(result)
    
    # Step 3: Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    total_segments = 0
    total_resolved = 0
    total_failed = 0
    
    for r in results:
        print(f"\n{r.get('title', r['id'])} ({r['id']}):")
        if "error" in r:
            print(f"  ERROR: {r['error']}")
        else:
            print(f"  Segments created: {r['segments']}")
            print(f"  Timestamps resolved: {r['resolved']} / {r['resolved'] + r['failed']} ({r['resolution_rate']:.1f}%)")
            total_segments += r['segments']
            total_resolved += r['resolved']
            total_failed += r['failed']
            
            if r.get('sample_segments'):
                print(f"  Sample segments:")
                for s in r['sample_segments']:
                    print(f"    - [{s['timestamp']}] {s['topic']} ({s['duration']})")
                    print(f"      {s['summary']}")
    
    print(f"\n" + "-" * 40)
    total_all = total_resolved + total_failed
    overall_rate = (total_resolved / total_all * 100) if total_all > 0 else 0
    print(f"TOTAL: {total_segments} segments created")
    print(f"RESOLUTION: {total_resolved}/{total_all} timestamps resolved ({overall_rate:.1f}%)")
    
    if overall_rate >= 80:
        print("\n✅ SUCCESS: Resolution rate >= 80%")
    else:
        print("\n⚠️ WARNING: Resolution rate < 80%")


if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    asyncio.run(main())
