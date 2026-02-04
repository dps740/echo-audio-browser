# Echo - Topic-First Audio Browser

## Project Overview
**Status:** MVP Development (Phase 1 Complete)
**Repo:** https://github.com/dps740/echo-audio-browser
**Started:** 2026-01-26
**Last Updated:** 2026-02-04

## Core Concept
Browse podcasts by **topic** rather than episode. Search "AI safety" and get a playlist of relevant 60-second segments from multiple podcasts. Audio streams directly from source (YouTube/podcast CDN) — zero hosting costs.

## Architecture

```
┌──────────────────────┐     ┌───────────────────┐     ┌─────────────┐
│  Windows Client      │────▶│  Cloud Server     │────▶│  Web Player │
│  (yt-dlp + Whisper)  │     │  (FastAPI)        │     │  (browser)  │
│  - YouTube captions  │     │  - ChromaDB       │     │  - Search   │
│  - Whisper fallback  │     │  - Hybrid search  │     │  - Play     │
└──────────────────────┘     │  - Ingest API     │     │  - Playlist │
                             └───────────────────┘     └─────────────┘
                                      │
                            Audio streams from YouTube/CDN
                                  (no hosting needed!)
```

## What's Built

### Server (FastAPI) ~2,000 lines
- **Feed management** — RSS feed parsing, episode listing
- **Transcript resolver** — Scrapes free transcripts from podcast websites
- **Hybrid search** — Semantic (ChromaDB) + keyword boosting
- **Playlist API** — Topic playlists, search playlists, daily mix
- **Ingest API** — Receives transcripts from local clients
- **Web player** — Dark-themed UI, search → play → auto-advance

### Windows Client
- **echo_ingest.py** — Extracts YouTube captions with timestamps
- **Whisper fallback** — GPU-accelerated local transcription
- **start_server.bat** — One-click local server startup

### Key Technical Decisions
1. **Virtual stitching** — Stream from source URLs, no audio hosting ($0 bandwidth)
2. **Transcript-first** — Use existing transcripts before paying for transcription
3. **Hybrid search** — Semantic similarity + keyword matching for precision
4. **~60 sec segments** — Granular enough for topic precision, long enough for context
5. **YouTube as source** — Captions synced to audio (website transcripts aren't!)

## Cost Structure

| Component | Cost |
|-----------|------|
| YouTube captions | $0 (free) |
| Whisper transcription | $0 (local GPU) or $0.006/min API |
| LLM segmentation | ~$0.02/episode (GPT-4o-mini) — OPTIONAL |
| ChromaDB | $0 (self-hosted) |
| Audio hosting | $0 (virtual stitching) |
| Server hosting | $0-5/mo (Railway free tier) |

**Per episode cost: ~$0** (YouTube captions) or **~$0.02** (with LLM enrichment)

## Key Finding: Timestamp Mismatch ⚠️
- Podcast website transcripts (e.g., lexfridman.com) have timestamps for the YouTube version
- The podcast MP3 file has different timing (intros, ads, different edits)
- **Solution: Use YouTube as both audio AND transcript source** — timestamps match!

## MVP Roadmap

### ✅ Phase 1: Proof of Concept (DONE - Feb 4, 2026)
- [x] FastAPI server with search + playback
- [x] Transcript scraping from podcast websites
- [x] LLM segmentation (GPT-4o-mini)
- [x] ChromaDB vector search
- [x] Web player with virtual stitching
- [x] Fixed timestamp mismatch issue
- [x] Windows client for YouTube caption extraction
- [x] Hybrid search (semantic + keyword)

### 🔲 Phase 2: Content Pipeline (NEXT)
- [ ] David ingests 50 popular podcasts from Windows PC
- [ ] Validate YouTube caption quality at scale
- [ ] Test Whisper fallback on GTX 1050 Ti
- [ ] Build batch ingestion script (process full YouTube playlists)

### 🔲 Phase 3: Deploy for Testing
- [ ] Deploy server to Railway (stable URL)
- [ ] Upload pre-indexed database
- [ ] Share for feedback
- [ ] Mobile-friendly player improvements

### 🔲 Phase 4: Growth Features
- [ ] User accounts & personalization
- [ ] "Add your own podcast" workflow
- [ ] Browser extension for one-click ingestion
- [ ] Auto-ingest new episodes (residential proxy or Whisper API)

## Files & Locations

| Item | Location |
|------|----------|
| Server code | `~/clawd/projects/echo-audio-browser/app/` |
| Web player | `~/clawd/projects/echo-audio-browser/static/index.html` |
| Windows client | `~/clawd/projects/echo-audio-browser/windows-full-package/` |
| Windows zip (Drive) | See Google Drive link in project notes |
| ChromaDB data | `~/clawd/projects/echo-audio-browser/chroma_data/` |
| Git repo | https://github.com/dps740/echo-audio-browser |

## Audio Storage
- Audio downloads locally via yt-dlp as MP3
- Server serves from `/audio/` directory (static files)
- ~50-100 MB per episode, ~20-40 GB for 400 episodes
- Local = instant seeking, no buffering, no expiring URLs
- For production: S3/Backblaze (~$0.005/GB/mo → $0.20/mo for 40GB)

## David's Windows Setup
- **PC:** Windows desktop with GTX 1050 Ti
- **Python:** Already installed
- **GPU use:** faster-whisper for local transcription when YouTube captions unavailable
- **Package (v4):** https://drive.google.com/file/d/1R13s57cR7DOM6w_vUgrWFa7uWWlVNtHi/view

## Quick Start (David)
1. Download zip, extract to C:\Echo
2. `pip install yt-dlp requests fastapi uvicorn chromadb pydantic-settings httpx`
3. Double-click `start_server.bat`
4. New cmd: `python echo_batch_ingest.py --channel @lexfridman --limit 5`
5. Browser: http://localhost:8765/player
