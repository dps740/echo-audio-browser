#!/usr/bin/env python3
"""Seed the database with test data for web UI testing."""

import asyncio
import json
import os
import sys
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()

async def main():
    print("Seeding Echo with Dario Amodei episode...")
    
    from app.services.transcript_resolver import fetch_lex_fridman_transcript, transcript_to_text_with_times
    from openai import OpenAI
    import chromadb
    from chromadb.config import Settings
    
    # Fetch transcript
    transcript = await fetch_lex_fridman_transcript("Dario Amodei")
    print(f"✅ Got transcript: {len(transcript.segments)} segments")
    
    # Segment with LLM
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    transcript_text = transcript_to_text_with_times(transcript)[:30000]
    
    prompt = """Analyze this podcast transcript and identify 8-12 distinct topic segments of 3-10 minutes each.

For each segment, provide:
- start_min: Starting minute
- end_min: Ending minute  
- summary: 1-2 sentence summary of the key insight
- topic_tags: 2-3 relevant tags
- density_score: 0.0-1.0 (how much valuable insight vs filler)

Return JSON array only.

TRANSCRIPT:
""" + transcript_text
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    
    content = response.choices[0].message.content
    if "```" in content:
        content = content.split("```")[1].replace("json", "").strip()
    
    segments = json.loads(content)
    print(f"✅ Identified {len(segments)} segments")
    
    # Store in persistent ChromaDB
    os.makedirs('./chroma_data', exist_ok=True)
    chroma_client = chromadb.PersistentClient(path="./chroma_data")
    
    # Clear and recreate collection
    try:
        chroma_client.delete_collection("segments")
    except:
        pass
    
    collection = chroma_client.create_collection("segments")
    
    # Audio URL for the episode (from RSS feed)
    audio_url = "https://media.blubrry.com/lex_fridman/content.blubrry.com/lex_fridman/Dario_Amodei.mp3"
    
    ids = []
    documents = []
    metadatas = []
    
    for i, seg in enumerate(segments):
        seg_id = f"dario-amodei_{i}"
        start_ms = seg['start_min'] * 60000
        end_ms = seg['end_min'] * 60000
        
        seg_text = " ".join([
            s['text'] for s in transcript.segments 
            if s['ms'] >= start_ms and s['ms'] < end_ms
        ])[:2000]
        
        doc = f"{seg['summary']}\n\n{seg_text}"
        
        ids.append(seg_id)
        documents.append(doc)
        metadatas.append({
            "episode_id": "dario-amodei",
            "episode_title": "Dario Amodei: Anthropic CEO on Claude, AGI & AI Safety",
            "podcast_title": "Lex Fridman Podcast",
            "audio_url": audio_url,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "summary": seg['summary'],
            "topic_tags": ",".join(seg['topic_tags']),
            "density_score": seg['density_score'],
        })
    
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"✅ Stored {len(segments)} segments")
    
    print("\n📋 Segments created:")
    for i, seg in enumerate(segments):
        print(f"   {i+1}. [{seg['start_min']}-{seg['end_min']}m] {seg['summary'][:60]}...")
    
    print("\n✅ Ready! Start server with: uvicorn app.main:app --reload")

if __name__ == "__main__":
    asyncio.run(main())
