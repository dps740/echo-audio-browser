#!/usr/bin/env python3
"""Quick re-ingest of a single episode."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import re
import chromadb
from app.config import settings, get_embedding_function
from app.services.segmentation import segment_transcript
from app.services.transcription import TranscriptResult, TranscriptWord

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

EPISODE = {"id": "w2BqPnVKVo4", "title": "Future of Everything", "podcast": "All-In Podcast"}


def parse_vtt(vtt_path: str) -> list:
    with open(vtt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    segments = []
    pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})[^\n]*\n([^\n]+(?:\n[^\n]+)*?)(?=\n\n|\n\d{2}:\d{2}|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    def to_ms(t):
        h, m, rest = t.split(':')
        s, ms = rest.split('.')
        return int(h)*3600000 + int(m)*60000 + int(s)*1000 + int(ms)
    
    seen = set()
    for start, end, text in matches:
        clean = re.sub(r'<[^>]+>', '', text)
        clean = ' '.join(clean.split())
        if clean and clean not in seen:
            seen.add(clean)
            segments.append({"start_ms": to_ms(start), "end_ms": to_ms(end), "text": clean})
    return segments


def segments_to_words(segments: list) -> list:
    words = []
    for seg in segments:
        text = seg["text"].replace("&gt;", ">").replace("&lt;", "<").replace(">>", "")
        seg_words = text.split()
        if not seg_words:
            continue
        duration = seg["end_ms"] - seg["start_ms"]
        word_duration = duration / len(seg_words)
        for i, word in enumerate(seg_words):
            words.append(TranscriptWord(
                word=word,
                start_ms=seg["start_ms"] + int(i * word_duration),
                end_ms=seg["start_ms"] + int((i + 1) * word_duration),
                confidence=1.0
            ))
    return words


def extract_key_terms(transcript: str, summary: str) -> list:
    import openai
    if not settings.openai_api_key:
        return []
    try:
        client = openai.OpenAI(api_key=settings.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": f"Extract 3-7 key terms from:\nSummary: {summary}\nTranscript: {transcript[:1000]}\n\nKey terms:"}],
            temperature=0.3,
            max_tokens=100,
        )
        return [t.strip() for t in response.choices[0].message.content.strip().split(",") if t.strip()][:10]
    except:
        return []


async def main():
    print(f"Processing: {EPISODE['title']} ({EPISODE['id']})")
    
    client = chromadb.PersistentClient(path="./chroma_data")
    ef = get_embedding_function()
    collection = client.get_or_create_collection("segments", embedding_function=ef)
    
    video_id = EPISODE["id"]
    vtt_path = f"audio/{video_id}.en.vtt"
    
    print(f"Parsing VTT: {vtt_path}")
    vtt_segments = parse_vtt(vtt_path)
    print(f"Found {len(vtt_segments)} VTT segments")
    
    words = segments_to_words(vtt_segments)
    print(f"Converted to {len(words)} words")
    
    total_duration_ms = max([s["end_ms"] for s in vtt_segments]) if vtt_segments else 0
    transcript = TranscriptResult(
        text=" ".join([s["text"] for s in vtt_segments]),
        words=words,
        duration_ms=total_duration_ms,
        speakers=[]
    )
    
    print("Running LLM segmentation...")
    llm_segments = await segment_transcript(transcript, EPISODE["title"])
    print(f"LLM returned {len(llm_segments)} segments")
    
    ids, docs, metas = [], [], []
    resolved = 0
    for i, seg in enumerate(llm_segments):
        if seg.start_ms is not None and seg.end_ms is not None and seg.end_ms > seg.start_ms:
            resolved += 1
            key_terms = extract_key_terms(seg.transcript_text, seg.summary)
            
            enriched_doc = f"""TOPIC: {', '.join(seg.topic_tags)}
SUMMARY: {seg.summary}
KEY TERMS: {', '.join(key_terms)}
TRANSCRIPT: {seg.transcript_text[:2000]}"""
            
            ids.append(f"{video_id}_{i}")
            docs.append(enriched_doc)
            metas.append({
                "episode_id": video_id,
                "episode_title": EPISODE["title"],
                "podcast_title": EPISODE["podcast"],
                "audio_url": f"/audio/{video_id}.mp3",
                "start_ms": seg.start_ms,
                "end_ms": seg.end_ms,
                "summary": seg.summary,
                "topic_tags": ",".join(seg.topic_tags),
                "key_terms": ",".join(key_terms),
                "density_score": seg.density_score,
                "source": "youtube",
            })
    
    if ids:
        print(f"Storing {len(ids)} segments...")
        for j in range(0, len(ids), 50):
            collection.upsert(ids=ids[j:j+50], documents=docs[j:j+50], metadatas=metas[j:j+50])
    
    print(f"\nResult: {resolved} segments stored ({resolved}/{len(llm_segments)} resolved)")
    
    # Verify
    results = collection.get(where={"episode_id": video_id}, include=["metadatas"])
    print(f"Verified: {len(results['ids'])} segments in DB")


if __name__ == "__main__":
    asyncio.run(main())
