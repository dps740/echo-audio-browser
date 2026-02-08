#!/usr/bin/env python3
"""Ingest locally downloaded episodes into Echo."""

import json
import re
import requests
from pathlib import Path

SERVER_URL = "http://localhost:8765"
INPUT_DIR = Path("/home/ubuntu/clawd/projects/echo-processing/input")

def parse_vtt(vtt_path: str) -> list:
    """Parse VTT to get timestamped segments."""
    with open(vtt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    segments = []
    cue_pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})'
    
    def ts_to_ms(ts: str) -> int:
        parts = ts.split(':')
        h, m = int(parts[0]), int(parts[1])
        s_ms = parts[2].split('.')
        s, ms = int(s_ms[0]), int(s_ms[1])
        return h * 3600000 + m * 60000 + s * 1000 + ms
    
    lines = content.split('\n')
    i = 0
    while i < len(lines):
        line = lines[i]
        match = re.match(cue_pattern, line)
        if match:
            start = ts_to_ms(match.group(1))
            end = ts_to_ms(match.group(2))
            
            text_lines = []
            i += 1
            while i < len(lines) and lines[i].strip():
                # Remove timestamp tags
                clean = re.sub(r'<[^>]+>', '', lines[i])
                if clean.strip():
                    text_lines.append(clean.strip())
                i += 1
            
            text = ' '.join(text_lines)
            if text:
                segments.append({
                    'start_ms': start,
                    'end_ms': end,
                    'text': text
                })
        i += 1
    
    # Deduplicate consecutive identical texts
    merged = []
    for seg in segments:
        if not merged or seg['text'] != merged[-1]['text']:
            merged.append(seg)
        else:
            merged[-1]['end_ms'] = seg['end_ms']
    
    return merged


def ingest_episode(episode_dir: Path):
    """Ingest a single episode."""
    ep_id = episode_dir.name
    
    # Find VTT file
    vtt_files = list(episode_dir.glob('*.vtt'))
    if not vtt_files:
        print(f"  No VTT file found for {ep_id}")
        return False
    
    vtt_path = vtt_files[0]
    
    # Load metadata
    metadata_path = episode_dir / 'metadata.json'
    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
    
    title = metadata.get('title', ep_id)
    channel = metadata.get('channel', 'Unknown')
    
    print(f"  Parsing VTT: {vtt_path.name}")
    segments = parse_vtt(str(vtt_path))
    print(f"  Segments: {len(segments)}")
    
    if not segments:
        print(f"  No segments found!")
        return False
    
    # Build request
    request_data = {
        "episode_id": ep_id,
        "episode_title": title,
        "podcast_title": channel,
        "audio_url": f"/audio/{ep_id}.wav",
        "segments": segments,
        "source": "youtube"
    }
    
    print(f"  Ingesting {ep_id}...")
    try:
        resp = requests.post(
            f"{SERVER_URL}/ingest/episode",
            json=request_data,
            timeout=60
        )
        if resp.status_code == 200:
            result = resp.json()
            print(f"  ✅ Success: {result.get('chunks_created', 0)} chunks")
            return True
        else:
            print(f"  ❌ Failed: {resp.status_code} - {resp.text[:200]}")
            return False
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False


def main():
    print("=== Ingesting Local Episodes ===\n")
    
    success = 0
    failed = 0
    
    for ep_dir in INPUT_DIR.iterdir():
        if ep_dir.is_dir():
            print(f"\n📺 {ep_dir.name}")
            if ingest_episode(ep_dir):
                success += 1
            else:
                failed += 1
    
    print(f"\n=== Done: {success} success, {failed} failed ===")


if __name__ == '__main__':
    main()
