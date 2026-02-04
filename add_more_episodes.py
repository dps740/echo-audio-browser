#!/usr/bin/env python3
"""Add more Lex Fridman episodes to Echo."""

import asyncio
import json
import os
import sys
import re
import urllib.request

sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import chromadb

# Episodes to add (guest name, approximate audio URL pattern)
EPISODES = [
    ("Sam Altman 2", "sam-altman-2"),
    ("Elon Musk 4", "elon-musk-4"),
    ("Mark Zuckerberg 3", "mark-zuckerberg-3"),
    ("Andrej Karpathy", "andrej-karpathy-2"),
    ("Yann LeCun 3", "yann-lecun-3"),
]

def fetch_transcript(slug):
    """Fetch transcript from lexfridman.com."""
    url = f"https://lexfridman.com/{slug}-transcript"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode('utf-8')
        
        # Parse segments
        pattern = r'\((\d{1,2}:\d{2}:\d{2})\)\s*([^(]+?)(?=\(\d{1,2}:\d{2}:\d{2}\)|$)'
        matches = re.findall(pattern, html, re.DOTALL)
        
        if not matches:
            return None
        
        segments = []
        for ts, text in matches:
            clean_text = re.sub(r'<[^>]+>', ' ', text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            clean_text = clean_text.replace('&#8217;', "'").replace('&#8220;', '"').replace('&#8221;', '"')
            
            # Parse timestamp to ms
            parts = ts.split(':')
            ms = (int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])) * 1000
            
            if len(clean_text) > 10:
                segments.append({'ms': ms, 'text': clean_text})
        
        return segments
    except Exception as e:
        print(f"  ❌ Failed to fetch {slug}: {e}")
        return None

def segments_to_text(segments, max_chars=30000):
    """Convert segments to text for LLM."""
    lines = []
    for seg in segments:
        mins = seg['ms'] // 60000
        lines.append(f"[{mins}m] {seg['text']}")
    return "\n".join(lines)[:max_chars]

async def main():
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    chroma_client = chromadb.PersistentClient(path="./chroma_data")
    collection = chroma_client.get_or_create_collection("segments")
    
    for guest, slug in EPISODES:
        print(f"\n📥 Processing: {guest}...")
        
        # Fetch transcript
        segments = fetch_transcript(slug)
        if not segments:
            continue
        
        print(f"  ✅ Got {len(segments)} transcript segments")
        
        # Segment with LLM
        transcript_text = segments_to_text(segments)
        
        prompt = f"""Analyze this podcast transcript from Lex Fridman's interview with {guest}.
Identify 8-12 distinct topic segments of 3-10 minutes each.

For each segment:
- start_min, end_min: time range
- summary: 1-2 sentence key insight
- topic_tags: 2-3 relevant tags
- density_score: 0.0-1.0

Return JSON array only.

TRANSCRIPT:
{transcript_text}"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        
        content = response.choices[0].message.content
        if "```" in content:
            content = content.split("```")[1].replace("json", "").strip()
        
        topic_segments = json.loads(content)
        print(f"  ✅ Identified {len(topic_segments)} topics")
        
        # Get audio URL (pattern for Lex episodes)
        audio_url = f"https://media.blubrry.com/lex_fridman/content.blubrry.com/lex_fridman/{slug.replace('-', '_').title().replace('_', '')}.mp3"
        
        # Store in ChromaDB
        ids = []
        documents = []
        metadatas = []
        
        for i, seg in enumerate(topic_segments):
            seg_id = f"{slug}_{i}"
            start_ms = seg['start_min'] * 60000
            end_ms = seg['end_min'] * 60000
            
            seg_text = " ".join([
                s['text'] for s in segments 
                if s['ms'] >= start_ms and s['ms'] < end_ms
            ])[:2000]
            
            doc = f"{seg['summary']}\n\n{seg_text}"
            
            ids.append(seg_id)
            documents.append(doc)
            metadatas.append({
                "episode_id": slug,
                "episode_title": f"{guest} - Lex Fridman Podcast",
                "podcast_title": "Lex Fridman Podcast",
                "audio_url": audio_url,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "summary": seg['summary'],
                "topic_tags": ",".join(seg['topic_tags']),
                "density_score": seg['density_score'],
            })
        
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        print(f"  ✅ Stored {len(topic_segments)} segments")
    
    # Summary
    total = collection.count()
    print(f"\n{'='*50}")
    print(f"✅ DONE! Total segments in database: {total}")
    print(f"{'='*50}")

if __name__ == "__main__":
    asyncio.run(main())
