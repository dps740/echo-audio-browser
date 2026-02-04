#!/usr/bin/env python3
"""
Echo Ingest Client for Windows
Extracts YouTube captions and sends to Echo service.

Requirements:
    pip install yt-dlp requests

Usage:
    python echo_ingest.py "https://www.youtube.com/watch?v=VIDEO_ID"
    python echo_ingest.py "https://www.youtube.com/watch?v=VIDEO_ID" --whisper
"""

import sys
import json
import argparse
import subprocess
import tempfile
import re
import requests
from pathlib import Path

# Configuration - UPDATE THIS to your server URL
SERVER_URL = "http://localhost:8765"


def get_video_info(url):
    """Get video metadata using yt-dlp."""
    cmd = ["yt-dlp", "--dump-json", "--no-download", url]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"yt-dlp failed: {result.stderr}")
    return json.loads(result.stdout)


def get_captions(url):
    """Extract captions with timestamps using yt-dlp."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "yt-dlp",
            "--write-auto-sub",
            "--write-sub",
            "--sub-lang", "en",
            "--sub-format", "json3",
            "--skip-download",
            "-o", f"{tmpdir}/subs",
            url
        ]
        subprocess.run(cmd, capture_output=True)
        
        sub_files = list(Path(tmpdir).glob("*.json3"))
        
        if not sub_files:
            cmd[5] = "vtt"
            subprocess.run(cmd, capture_output=True)
            sub_files = list(Path(tmpdir).glob("*.vtt"))
        
        if not sub_files:
            raise Exception("No captions found")
        
        sub_file = sub_files[0]
        
        if sub_file.suffix == ".json3":
            with open(sub_file, encoding='utf-8') as f:
                data = json.load(f)
            
            segments = []
            for event in data.get("events", []):
                if "segs" in event:
                    start_ms = event.get("tStartMs", 0)
                    duration_ms = event.get("dDurationMs", 0)
                    text = "".join(s.get("utf8", "") for s in event["segs"]).strip()
                    text = text.replace("\n", " ")
                    if text and text != " ":
                        segments.append({
                            "start_ms": start_ms,
                            "end_ms": start_ms + duration_ms,
                            "text": text
                        })
            return segments
        else:
            return parse_vtt(sub_file)
    
    return []


def parse_vtt(vtt_path):
    """Parse VTT subtitle file."""
    segments = []
    with open(vtt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\n(.+?)(?=\n\n|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    def to_ms(t):
        h, m, rest = t.split(':')
        s, ms = rest.split('.')
        return int(h)*3600000 + int(m)*60000 + int(s)*1000 + int(ms)
    
    for start, end, text in matches:
        clean_text = re.sub(r'<[^>]+>', '', text).strip().replace("\n", " ")
        if clean_text:
            segments.append({
                "start_ms": to_ms(start),
                "end_ms": to_ms(end),
                "text": clean_text
            })
    
    return segments


def get_audio_url(url):
    """Get direct audio stream URL."""
    cmd = ["yt-dlp", "-f", "bestaudio", "-g", url]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return url
    return result.stdout.strip()


def transcribe_with_whisper(url, model="base"):
    """Download audio and transcribe with local Whisper."""
    print("Downloading audio...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = f"{tmpdir}/audio.mp3"
        
        cmd = ["yt-dlp", "-f", "bestaudio", "-x", "--audio-format", "mp3", 
               "-o", audio_path.replace('.mp3', '.%(ext)s'), url]
        subprocess.run(cmd, check=True)
        
        # Find the actual file
        audio_files = list(Path(tmpdir).glob("audio.*"))
        if not audio_files:
            raise Exception("Failed to download audio")
        audio_path = str(audio_files[0])
        
        print(f"Transcribing with Whisper ({model} model)...")
        
        try:
            from faster_whisper import WhisperModel
            whisper_model = WhisperModel(model, device="cuda", compute_type="float16")
            segments_gen, info = whisper_model.transcribe(audio_path)
            
            segments = []
            for seg in segments_gen:
                segments.append({
                    "start_ms": int(seg.start * 1000),
                    "end_ms": int(seg.end * 1000),
                    "text": seg.text.strip()
                })
                print(f"  [{seg.start:.1f}s] {seg.text.strip()[:50]}...")
            return segments
            
        except ImportError:
            print("faster-whisper not found, trying openai-whisper...")
            import whisper
            whisper_model = whisper.load_model(model)
            result = whisper_model.transcribe(audio_path)
            
            segments = []
            for seg in result["segments"]:
                segments.append({
                    "start_ms": int(seg["start"] * 1000),
                    "end_ms": int(seg["end"] * 1000),
                    "text": seg["text"].strip()
                })
            return segments


def send_to_server(episode_id, title, podcast, audio_url, segments, source="youtube"):
    """Send transcript to Echo server."""
    payload = {
        "episode_id": episode_id,
        "episode_title": title,
        "podcast_title": podcast,
        "audio_url": audio_url,
        "segments": segments,
        "source": source
    }
    
    response = requests.post(
        f"{SERVER_URL}/ingest/episode",
        json=payload,
        timeout=120
    )
    response.raise_for_status()
    return response.json()


def main():
    parser = argparse.ArgumentParser(description="Ingest YouTube video into Echo")
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument("--whisper", action="store_true", 
                        help="Use local Whisper instead of YouTube captions")
    parser.add_argument("--model", default="base", 
                        help="Whisper model: tiny, base, small, medium, large")
    parser.add_argument("--server", default=SERVER_URL, help="Echo server URL")
    args = parser.parse_args()
    
    global SERVER_URL
    SERVER_URL = args.server
    
    print(f"Server: {SERVER_URL}")
    print(f"Fetching video info...")
    
    info = get_video_info(args.url)
    
    video_id = info["id"]
    title = info["title"]
    channel = info.get("channel", info.get("uploader", "Unknown"))
    duration = info.get("duration", 0)
    
    print(f"Video: {title}")
    print(f"Channel: {channel}")
    print(f"Duration: {duration // 60}m {duration % 60}s")
    
    if args.whisper:
        print(f"\nUsing local Whisper ({args.model} model)...")
        segments = transcribe_with_whisper(args.url, args.model)
        source = "whisper"
    else:
        print(f"\nExtracting YouTube captions...")
        segments = get_captions(args.url)
        source = "youtube"
    
    if not segments:
        print("\n❌ No captions found!")
        print("Try: python echo_ingest.py URL --whisper")
        sys.exit(1)
    
    print(f"✓ Found {len(segments)} caption segments")
    
    print(f"\nGetting audio URL...")
    audio_url = get_audio_url(args.url)
    print(f"✓ Audio URL obtained")
    
    print(f"\nSending to Echo server...")
    result = send_to_server(
        episode_id=video_id,
        title=title,
        podcast=channel,
        audio_url=audio_url,
        segments=segments,
        source=source
    )
    
    print(f"\n{'='*50}")
    print(f"✅ SUCCESS!")
    print(f"{'='*50}")
    print(f"Episode ID: {result['episode_id']}")
    print(f"Chunks created: {result['chunks_created']}")
    print(f"Duration: {result['total_duration_ms'] // 60000} minutes")
    print(f"\nYou can now search for this content in Echo!")


if __name__ == "__main__":
    main()
