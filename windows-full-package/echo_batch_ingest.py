#!/usr/bin/env python3
"""
Echo Batch Ingest - Ingest entire YouTube channels/playlists
Automatically processes all videos with captions.

Usage:
    python echo_batch_ingest.py                     # Ingest all curated podcasts
    python echo_batch_ingest.py --channel "@lexfridman"  # Specific channel
    python echo_batch_ingest.py --playlist "PLxxxxx"     # Specific playlist
    python echo_batch_ingest.py --limit 10               # Limit per channel
"""

import subprocess
import json
import sys
import os
import argparse
import time
import requests
from pathlib import Path
from echo_ingest import get_captions, download_audio, send_to_server

SERVER_URL = "http://localhost:8765"

# ============================================================
# CURATED PODCAST LIST
# High information density, serious audience
# ============================================================

PODCASTS = [
    {
        "name": "Lex Fridman Podcast",
        "channel": "@lexfridman",
        "playlist": "PLrAXtmErZgOeiKm4sgNOknGvNjby9efdf",
        "description": "Deep conversations on AI, science, philosophy",
    },
    {
        "name": "The Knowledge Project (Farnam Street)",
        "channel": "@tkppodcast",
        "description": "Mental models, decision making, wisdom",
    },
    {
        "name": "Joe Rogan Experience",
        "channel": "@joerogan",
        "description": "Long-form conversations, varied topics",
    },
    {
        "name": "Huberman Lab",
        "channel": "@hubaboratorylab",
        "description": "Neuroscience, health, performance",
    },
    {
        "name": "Tim Ferriss Show",
        "channel": "@timferriss",
        "description": "World-class performers, tactics, routines",
    },
    {
        "name": "All-In Podcast",
        "channel": "@alaborin",
        "description": "Tech, economics, politics from investors",
    },
    {
        "name": "Acquired",
        "channel": "@AcquiredFM",
        "description": "Deep dives into great companies",
    },
    {
        "name": "Dwarkesh Podcast",
        "channel": "@DwarkeshPatel",
        "description": "Deep conversations on progress, AI, history",
    },
    {
        "name": "EconTalk",
        "channel": "@EconTalk",
        "description": "Economics, philosophy, ideas",
    },
    {
        "name": "Invest Like the Best",
        "channel": "@investlikethebest",
        "description": "Investing, business, decision making",
    },
    {
        "name": "Naval Ravikant",
        "channel": "@NavalR",
        "description": "Wealth, happiness, philosophy",
    },
    {
        "name": "Making Sense (Sam Harris)",
        "channel": "@samharrisorg",
        "description": "Neuroscience, philosophy, AI, society",
    },
    {
        "name": "Conversations with Tyler (Cowen)",
        "channel": "@MercatusCenter",
        "description": "Economics, culture, ideas",
    },
    {
        "name": "Peter Attia - The Drive",
        "channel": "@PeterAttiaMD",
        "description": "Longevity, health, medicine",
    },
    {
        "name": "My First Million",
        "channel": "@MyFirstMillionPod",
        "description": "Business ideas, entrepreneurship",
    },
    {
        "name": "Bankless",
        "channel": "@Bankless",
        "description": "Crypto, DeFi, digital economy",
    },
    {
        "name": "The Lunar Society (Dwarkesh)",
        "channel": "@DwarkeshPatel",
        "description": "AI, progress, civilization",
    },
    {
        "name": "80,000 Hours",
        "channel": "@eaborightytwozerozerozerozerohours",
        "description": "Effective altruism, careers, AI safety",
    },
    {
        "name": "Founders Podcast",
        "channel": "@FoundersPodcast",
        "description": "Biographies of entrepreneurs and innovators",
    },
    {
        "name": "Sean Carroll Mindscape",
        "channel": "@saboreancarroll",
        "description": "Physics, philosophy, science",
    },
]


def list_channel_videos(channel, limit=None):
    """Get video list from a YouTube channel."""
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--dump-json",
        f"https://www.youtube.com/{channel}/videos",
    ]
    if limit:
        cmd.extend(["--playlist-end", str(limit)])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    videos = []
    for line in result.stdout.strip().split('\n'):
        if line:
            try:
                data = json.loads(line)
                videos.append({
                    "id": data.get("id"),
                    "title": data.get("title", "Unknown"),
                    "url": data.get("url") or f"https://www.youtube.com/watch?v={data['id']}",
                    "duration": data.get("duration"),
                })
            except json.JSONDecodeError:
                continue
    
    return videos


def list_playlist_videos(playlist_id, limit=None):
    """Get video list from a YouTube playlist."""
    cmd = [
        "yt-dlp",
        "--flat-playlist",
        "--dump-json",
        f"https://www.youtube.com/playlist?list={playlist_id}",
    ]
    if limit:
        cmd.extend(["--playlist-end", str(limit)])
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    videos = []
    for line in result.stdout.strip().split('\n'):
        if line:
            try:
                data = json.loads(line)
                videos.append({
                    "id": data.get("id"),
                    "title": data.get("title", "Unknown"),
                    "url": data.get("url") or f"https://www.youtube.com/watch?v={data['id']}",
                    "duration": data.get("duration"),
                })
            except json.JSONDecodeError:
                continue
    
    return videos


def is_already_ingested(video_id):
    """Check if video is already in the database."""
    try:
        response = requests.get(f"{SERVER_URL}/ingest/stats", timeout=5)
        data = response.json()
        return video_id in data.get("episodes", [])
    except:
        return False


def ingest_video(video, podcast_name):
    """Ingest a single video."""
    video_id = video["id"]
    url = f"https://www.youtube.com/watch?v={video_id}"
    
    # Skip if already ingested
    if is_already_ingested(video_id):
        print(f"  ⏭️  Already ingested, skipping")
        return "skipped"
    
    # Skip very short videos (< 5 min) - probably not full episodes
    if video.get("duration") and video["duration"] < 300:
        print(f"  ⏭️  Too short ({video['duration']}s), skipping")
        return "skipped"
    
    try:
        # Get captions
        segments = get_captions(url)
        if not segments:
            print(f"  ❌ No captions available")
            return "no_captions"
        
        print(f"  ✓ {len(segments)} caption segments")
        
        # Download audio locally
        audio_url = download_audio(url, video_id)
        
        # Send to server
        result = send_to_server(
            episode_id=video_id,
            title=video["title"],
            podcast=podcast_name,
            audio_url=audio_url,
            segments=segments,
            source="youtube"
        )
        
        print(f"  ✅ {result['chunks_created']} chunks indexed")
        return "success"
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)[:80]}")
        return "error"


def save_progress(progress_file, data):
    """Save ingestion progress."""
    with open(progress_file, 'w') as f:
        json.dump(data, f, indent=2)


def load_progress(progress_file):
    """Load ingestion progress."""
    if os.path.exists(progress_file):
        with open(progress_file) as f:
            return json.load(f)
    return {"completed": [], "failed": [], "skipped": []}


def main():
    parser = argparse.ArgumentParser(description="Batch ingest YouTube podcasts into Echo")
    parser.add_argument("--channel", help="Specific YouTube channel handle (e.g., @lexfridman)")
    parser.add_argument("--playlist", help="YouTube playlist ID")
    parser.add_argument("--limit", type=int, default=20, help="Max videos per channel (default: 20)")
    parser.add_argument("--server", default=SERVER_URL, help="Echo server URL")
    parser.add_argument("--all", action="store_true", help="Ingest ALL curated podcasts")
    args = parser.parse_args()
    
    global SERVER_URL
    SERVER_URL = args.server
    
    # Check server is running
    try:
        r = requests.get(f"{SERVER_URL}/health", timeout=5)
        if r.json().get("status") != "healthy":
            raise Exception("unhealthy")
        print(f"✅ Server is running at {SERVER_URL}")
    except:
        print(f"❌ Server not reachable at {SERVER_URL}")
        print(f"   Start it first: start_server.bat")
        sys.exit(1)
    
    progress_file = "ingest_progress.json"
    progress = load_progress(progress_file)
    
    # Determine what to ingest
    if args.channel:
        podcasts = [{"name": args.channel, "channel": args.channel}]
    elif args.playlist:
        podcasts = [{"name": "Playlist", "playlist": args.playlist}]
    elif args.all:
        podcasts = PODCASTS
    else:
        # Default: show menu
        print("\n🎧 Available podcasts:\n")
        for i, p in enumerate(PODCASTS):
            print(f"  {i+1:2}. {p['name']}")
            print(f"      {p['description']}")
        
        print(f"\n  Options:")
        print(f"    python echo_batch_ingest.py --all           # Ingest all")
        print(f"    python echo_batch_ingest.py --limit 5 --all # 5 per channel")
        print(f"    python echo_batch_ingest.py --channel @lexfridman")
        print(f"    python echo_batch_ingest.py --channel @lexfridman --limit 50")
        return
    
    # Process each podcast
    total_success = 0
    total_failed = 0
    total_skipped = 0
    
    for podcast in podcasts:
        name = podcast["name"]
        print(f"\n{'='*60}")
        print(f"📺 {name}")
        print(f"{'='*60}")
        
        # Get video list
        if "playlist" in podcast and podcast["playlist"]:
            print(f"Fetching playlist videos (limit: {args.limit})...")
            videos = list_playlist_videos(podcast["playlist"], args.limit)
        elif "channel" in podcast:
            print(f"Fetching channel videos (limit: {args.limit})...")
            videos = list_channel_videos(podcast["channel"], args.limit)
        else:
            print("No channel or playlist specified, skipping")
            continue
        
        if not videos:
            print("  No videos found")
            continue
        
        print(f"Found {len(videos)} videos\n")
        
        for i, video in enumerate(videos):
            duration_str = f"{video['duration']//60}m" if video.get('duration') else "?"
            print(f"  [{i+1}/{len(videos)}] {video['title'][:60]}... ({duration_str})")
            
            # Skip already completed
            if video["id"] in progress["completed"]:
                print(f"  ⏭️  Already done")
                total_skipped += 1
                continue
            
            result = ingest_video(video, name)
            
            if result == "success":
                progress["completed"].append(video["id"])
                total_success += 1
            elif result == "error" or result == "no_captions":
                progress["failed"].append(video["id"])
                total_failed += 1
            else:
                total_skipped += 1
            
            # Save progress after each video
            save_progress(progress_file, progress)
            
            # Small delay to be nice to YouTube
            time.sleep(1)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"✅ BATCH INGESTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Success: {total_success}")
    print(f"  Failed:  {total_failed}")
    print(f"  Skipped: {total_skipped}")
    
    # Get total stats
    try:
        r = requests.get(f"{SERVER_URL}/ingest/stats", timeout=5)
        stats = r.json()
        print(f"\n  Total in database: {stats['total_segments']} segments from {stats['total_episodes']} episodes")
    except:
        pass


if __name__ == "__main__":
    main()
