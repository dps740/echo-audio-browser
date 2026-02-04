#!/usr/bin/env python3
"""Re-index with granular transcript segments (~30-60 sec each)."""

import chromadb
import urllib.request
import re

# Get audio URLs from RSS
def get_audio_urls():
    url = "https://lexfridman.com/feed/podcast/"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=15) as response:
        content = response.read().decode('utf-8')
    
    urls = {}
    searches = [
        ('dario-amodei', 'Dario Amodei'),
        ('sam-altman-2', 'Sam Altman'),
        ('elon-musk-4', 'Elon Musk'),
        ('mark-zuckerberg-3', 'Zuckerberg'),
        ('yann-lecun-3', 'Yann LeCun'),
    ]
    for slug, search in searches:
        pattern = rf'<title>[^<]*{search}[^<]*</title>.*?<enclosure[^>]*url="([^"]+)"'
        match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
        if match:
            urls[slug] = match.group(1)
    return urls

def fetch_transcript(slug):
    """Fetch transcript segments from lexfridman.com."""
    url = f"https://lexfridman.com/{slug}-transcript"
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode('utf-8')
        
        pattern = r'\((\d{1,2}:\d{2}:\d{2})\)\s*([^(]+?)(?=\(\d{1,2}:\d{2}:\d{2}\)|$)'
        matches = re.findall(pattern, html, re.DOTALL)
        
        segments = []
        for ts, text in matches:
            clean_text = re.sub(r'<[^>]+>', ' ', text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            clean_text = clean_text.replace('&#8217;', "'").replace('&#8220;', '"').replace('&#8221;', '"')
            
            parts = ts.split(':')
            ms = (int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])) * 1000
            
            if len(clean_text) > 20:
                segments.append({'timestamp': ts, 'ms': ms, 'text': clean_text})
        
        return segments
    except:
        return None

def main():
    audio_urls = get_audio_urls()
    
    episodes = [
        ('dario-amodei', 'Dario Amodei: Anthropic CEO'),
        ('sam-altman-2', 'Sam Altman: OpenAI CEO'),
        ('elon-musk-4', 'Elon Musk: Tesla, SpaceX, xAI'),
        ('mark-zuckerberg-3', 'Mark Zuckerberg: Meta CEO'),
        ('yann-lecun-3', 'Yann LeCun: Meta AI'),
    ]
    
    # Clear and recreate collection
    client = chromadb.PersistentClient(path="./chroma_data")
    try:
        client.delete_collection("segments")
    except:
        pass
    collection = client.create_collection("segments")
    
    total = 0
    
    for slug, title in episodes:
        print(f"\n📥 {title}...")
        
        transcript = fetch_transcript(slug)
        if not transcript:
            print(f"  ❌ Failed to fetch")
            continue
        
        audio_url = audio_urls.get(slug, '')
        if not audio_url:
            print(f"  ❌ No audio URL")
            continue
        
        # Group into ~60 second chunks (2-3 transcript segments)
        ids, docs, metas = [], [], []
        
        i = 0
        chunk_num = 0
        while i < len(transcript):
            # Take 2-3 segments for ~60 sec chunk
            chunk_segs = transcript[i:i+3]
            if not chunk_segs:
                break
            
            start_ms = chunk_segs[0]['ms']
            end_ms = chunk_segs[-1]['ms'] + 45000  # Assume ~45 sec for last segment
            
            # Combine text
            chunk_text = " ".join(s['text'] for s in chunk_segs)
            
            # Create summary (first 150 chars)
            summary = chunk_text[:150] + "..." if len(chunk_text) > 150 else chunk_text
            
            seg_id = f"{slug}_chunk_{chunk_num}"
            
            ids.append(seg_id)
            docs.append(chunk_text)
            metas.append({
                "episode_id": slug,
                "episode_title": f"{title} - Lex Fridman Podcast",
                "podcast_title": "Lex Fridman Podcast",
                "audio_url": audio_url,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "summary": summary,
                "topic_tags": "",
                "density_score": 0.7,
            })
            
            chunk_num += 1
            i += 3
        
        collection.add(ids=ids, documents=docs, metadatas=metas)
        print(f"  ✅ {chunk_num} chunks (~60 sec each)")
        total += chunk_num
    
    print(f"\n{'='*50}")
    print(f"✅ Total: {total} granular segments indexed")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
