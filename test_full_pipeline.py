#!/usr/bin/env python3
"""Test the full Echo pipeline with a real episode."""

import asyncio
import json
import os
import sys

# Add app to path
sys.path.insert(0, '.')

from dotenv import load_dotenv
load_dotenv()

async def main():
    print("=" * 60)
    print("ECHO FULL PIPELINE TEST")
    print("=" * 60)
    
    # 1. FETCH TRANSCRIPT
    print("\n📥 Step 1: Fetching transcript from lexfridman.com...")
    
    from app.services.transcript_resolver import fetch_lex_fridman_transcript, transcript_to_text_with_times
    
    transcript = await fetch_lex_fridman_transcript("Dario Amodei")
    
    if not transcript:
        print("❌ Failed to fetch transcript")
        return
    
    print(f"✅ Got {len(transcript.segments)} segments")
    print(f"   Duration: {transcript.duration_ms / 1000 / 60:.0f} minutes")
    print(f"   Source: {transcript.source}")
    
    # 2. LLM SEGMENTATION
    print("\n🧠 Step 2: Segmenting with LLM...")
    
    openai_key = os.getenv('OPENAI_API_KEY')
    if not openai_key:
        print("❌ No OPENAI_API_KEY set")
        return
    
    from openai import OpenAI
    client = OpenAI(api_key=openai_key)
    
    # Prepare transcript text (first 30 mins for test)
    transcript_text = transcript_to_text_with_times(transcript)[:20000]
    
    prompt = """Analyze this podcast transcript and identify 5-10 distinct topic segments.

For each segment, provide:
- start_min: Starting minute
- end_min: Ending minute  
- summary: 1-2 sentence summary
- topic_tags: 2-3 relevant tags
- density_score: 0.0-1.0 (how much valuable insight vs filler)

Return JSON array only:
[{"start_min": 0, "end_min": 5, "summary": "...", "topic_tags": ["AI", "Safety"], "density_score": 0.8}]

TRANSCRIPT:
""" + transcript_text
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    
    content = response.choices[0].message.content
    
    # Parse JSON
    if "```" in content:
        content = content.split("```")[1].replace("json", "").strip()
    
    segments = json.loads(content)
    print(f"✅ Identified {len(segments)} topic segments")
    
    for i, seg in enumerate(segments[:3]):
        print(f"\n   Segment {i+1}: [{seg['start_min']}-{seg['end_min']}m]")
        print(f"   {seg['summary'][:80]}...")
        print(f"   Tags: {seg['topic_tags']}, Density: {seg['density_score']}")
    
    if len(segments) > 3:
        print(f"\n   ... and {len(segments)-3} more segments")
    
    # 3. STORE IN CHROMADB
    print("\n💾 Step 3: Storing in ChromaDB...")
    
    import chromadb
    
    client_db = chromadb.Client()
    collection = client_db.get_or_create_collection("echo_test")
    
    ids = []
    documents = []
    metadatas = []
    
    for i, seg in enumerate(segments):
        seg_id = f"dario-amodei_{i}"
        
        # Get transcript text for this segment
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
            "episode": "Dario Amodei: Anthropic CEO",
            "podcast": "Lex Fridman Podcast",
            "start_min": seg['start_min'],
            "end_min": seg['end_min'],
            "summary": seg['summary'],
            "tags": ",".join(seg['topic_tags']),
            "density": seg['density_score'],
        })
    
    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    print(f"✅ Stored {len(segments)} segments in ChromaDB")
    
    # 4. TEST SEARCH
    print("\n🔍 Step 4: Testing search...")
    
    queries = ["AI safety risks", "scaling laws", "meaning and purpose"]
    
    for query in queries:
        results = collection.query(query_texts=[query], n_results=2)
        print(f"\n   Query: '{query}'")
        for i, (id, meta) in enumerate(zip(results['ids'][0], results['metadatas'][0])):
            print(f"   → {meta['summary'][:60]}... [{meta['start_min']}-{meta['end_min']}m]")
    
    print("\n" + "=" * 60)
    print("✅ FULL PIPELINE TEST COMPLETE!")
    print("=" * 60)
    
    # Summary
    print(f"""
SUMMARY:
- Transcript: FREE (scraped from website)
- Segmentation: ~$0.01 (GPT-4o-mini)
- Storage: FREE (ChromaDB)
- Total cost: ~$0.01 per episode

Ready for playback testing via web UI!
""")

if __name__ == "__main__":
    asyncio.run(main())
