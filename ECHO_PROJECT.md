# Echo - Topic-First Audio Browser

## Project Overview
**Status:** V2 Segmentation — Testing improved search quality
**Repo:** https://github.com/dps740/echo-audio-browser
**Started:** 2026-01-26
**Last Updated:** 2026-02-06

## Key Documentation
- `SEGMENTATION_V2.md` — V2 segmentation approach (1-2 min segments, primary+secondary topics)
- `FUTURE_FEATURES.md` — Actionable intelligence layer (not yet built)
- `META_INTELLIGENCE_LAYER.md` — Advanced meta-analysis concepts (not yet built)
- `TOOLS_OF_TITANS_FEATURE.md` — Auto-generate "Tools of Titans" from podcasts (not yet built)

## Core Concept
Browse podcasts by **topic** rather than episode. Search "AI safety" and get a playlist of relevant segments from multiple podcasts. Audio stored locally — zero hosting costs.

## Architecture

```
┌──────────────────────┐
│  Windows Desktop     │
│  (all-in-one)        │
│  ├─ FastAPI server   │
│  ├─ ChromaDB         │
│  ├─ Web UI (browser) │
│  ├─ yt-dlp (captions + audio)
│  └─ OpenAI API (segmentation + embeddings)
└──────────────────────┘
```

David runs everything locally on his Windows PC. Server at `localhost:8765`, UI at `/player`.

## Full Ingestion Pipeline

New content goes through this pipeline (both on ingest and re-analyze):

```
YouTube URL
  → yt-dlp: extract captions (timestamped) + download audio MP3
  → GPT-4o-mini: LLM segmentation (5-15 segments per episode)
     - Identifies "atomic ideas" — standalone topic discussions
     - Sets segment boundaries (2-10 min each)
     - Generates summary, topic tags, density score per segment
  → GPT-4o-mini: key term extraction per segment
     - Named entities, technical terms, concepts (3-7 per segment)
  → Build enriched document per segment:
     TOPIC: [tags]
     SUMMARY: [LLM summary]
     KEY TERMS: [extracted terms]
     TRANSCRIPT: [raw text]
  → OpenAI text-embedding-3-small: embed enriched documents
  → ChromaDB: store segments with embeddings + metadata
```

**Cost per episode:** ~$0.02-0.05 (GPT-4o-mini segmentation + key terms + embeddings)

## Search Pipeline

```
User query (e.g. "AGI")
  → Synonym expansion (AGI → artificial general intelligence, superintelligence, etc.)
  → OpenAI embedding of query (text-embedding-3-small)
  → ChromaDB semantic search: top 50 candidates
  → Keyword boost: match query terms against transcript, summary, tags, key terms
  → Diversity filter: max 3 per episode, max 4 per podcast
  → Quality threshold: filter low-scoring results
  → Return top N ranked segments
```

## Current State (Feb 5, 2026)

### What's Working
- ✅ YouTube ingestion (single URL + batch channel)
- ✅ Full LLM segmentation pipeline
- ✅ Hybrid search (semantic + keyword)
- ✅ Web UI: Dashboard, Library, Ingest, Player tabs
- ✅ Audio playback with seek + crossfade
- ✅ Deep Re-analyze (full pipeline re-run on existing content)

### Known Issues Being Tested
- **Library tab failing** — was caused by fragile reindex swap (delete/recreate collection). Fixed: reindex now updates in place. Also added `/ingest/youtube/repair` endpoint for recovery.
- **Player not playing** — was failing silently. Fixed: now shows visible error messages. Likely cause: audio files not downloaded (expired streaming URLs stored instead of local paths). Fix: re-ingest affected episodes.
- **Same search results after old reanalyze** — old reanalyze only re-embedded, didn't re-segment. Fixed: replaced with Deep Re-analyze that runs full LLM pipeline.
- **Embedding function mismatch** — `hybrid_search` was using default MiniLM embeddings to query against OpenAI-embedded data. Fixed: all code paths now use `get_embedding_function()` consistently.

### David Testing (Feb 5)
David needs to:
1. `git pull` latest changes
2. Hit `POST http://localhost:8765/ingest/youtube/repair` first (clean up any corrupted collections from old reindex)
3. Restart server
4. Check Library tab loads
5. Run **🧠 Re-analyze All** from Ingest tab (full LLM pipeline on all existing content)
6. Test search — should return different/better segments with proper boundaries
7. Test player — check if audio plays. If errors, check error message in player bar
   - If "SRC_NOT_SUPPORTED" or 404 → audio wasn't downloaded, need to re-ingest
   - If "expired streaming URL" → same issue, re-ingest

## Bugs Fixed (Feb 5, 2026)

### 1. Reindex Collection Swap (CRITICAL)
**Problem:** Old reindex created `segments_new`, deleted `segments`, then tried to copy data back. If anything failed mid-swap, `segments` was empty/corrupted.
**Fix:** Reindex now upserts directly to existing `segments` collection. No delete/create/copy dance. Added `/repair` endpoint to recover from botched swaps.

### 2. Embedding Function Mismatch (CRITICAL)
**Problem:** `hybrid_search` and `library.py` accessed ChromaDB collection without specifying the embedding function. If data was embedded with OpenAI (1536-dim), but queries used default MiniLM (384-dim), search results would be garbage.
**Fix:** All code paths now use `get_embedding_function()` from config. Embedding function matches however data was ingested.

### 3. Player Silent Failure
**Problem:** When audio failed to load, player just reset the play button with no feedback. User had no idea why.
**Fix:** Player now shows error message in the player bar (red text). Detects and warns about expired streaming URLs. Proper event listener cleanup prevents race conditions.

### 4. Shallow vs Deep Reanalyze
**Problem:** Old reanalyze only re-enriched text and re-embedded. Same segment boundaries, same clips. Can't evaluate pipeline quality.
**Fix:** Replaced with single "Re-analyze All" button that runs the full pipeline: reconstruct transcript → LLM segmentation → key terms → enriched docs → embed.

## Files & Locations

| Item | Location |
|------|----------|
| Server code | `app/main.py`, `app/routers/`, `app/services/` |
| Web UI | `static/index.html` |
| Config | `app/config.py` (reads `.env` for API keys) |
| ChromaDB data | `chroma_data/` |
| Audio files | `app/audio/` (MP3s downloaded by yt-dlp) |
| Windows package | `windows-full-package/` |
| Git repo | https://github.com/dps740/echo-audio-browser |

### Key Source Files
- `app/routers/youtube_ingest.py` — Ingest, reindex, deep reindex, repair endpoints
- `app/services/segmentation.py` — LLM segmentation prompt + parsing
- `app/services/hybrid_search.py` — 5-step search pipeline
- `app/routers/library.py` — Library browsing (podcasts → episodes → segments)
- `app/routers/playlists.py` — Search → playlist generation
- `app/config.py` — Settings, embedding function selection

### Environment Variables (.env)
```
OPENAI_API_KEY=sk-...          # Required for segmentation, key terms, embeddings
EMBEDDING_MODEL=text-embedding-3-small
SEGMENTATION_MODEL=gpt-4o-mini
USE_OPENAI_EMBEDDINGS=true
CHROMA_PERSIST_DIR=./chroma_data
```

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/player` | Web UI |
| GET | `/health` | Health check |
| GET | `/library/overview` | Dashboard stats |
| GET | `/library/podcasts/{name}/episodes` | Episode list |
| GET | `/library/episodes/{id}/segments` | Segment list |
| DELETE | `/library/episodes/{id}` | Delete episode |
| GET | `/playlists/topic/{topic}` | Search → playlist |
| POST | `/ingest/youtube/url` | Ingest single video |
| POST | `/ingest/youtube/channel` | Batch ingest channel |
| POST | `/ingest/youtube/deep-reindex` | Full LLM re-segmentation |
| POST | `/ingest/youtube/repair` | Fix corrupted collections |
| POST | `/ingest/youtube/reindex` | Re-embed existing segments (legacy) |
| GET | `/ingest/youtube/jobs` | List active jobs |
| GET | `/debug/search?q=...` | Debug search pipeline |

## Next Steps

After David validates the deep re-analyze results:
1. **Evaluate search quality** — Do the new segments + embeddings produce better results?
2. **Fix audio playback** — Ensure all episodes have local MP3s (not expired streaming URLs)
3. **Scale test** — Ingest 50+ podcasts, test performance
4. **Deploy** — Railway or similar for stable URL
5. **Mobile UI** — Make player touch-friendly

## Cost Structure

| Component | Cost |
|-----------|------|
| YouTube captions | $0 (yt-dlp) |
| Audio download | $0 (yt-dlp, stored locally) |
| LLM segmentation | ~$0.02/episode (GPT-4o-mini) |
| Key term extraction | ~$0.01/episode (GPT-4o-mini) |
| OpenAI embeddings | ~$0.01/episode (text-embedding-3-small) |
| ChromaDB | $0 (local) |
| **Total per episode** | **~$0.03-0.05** |

## David's Windows Setup
- **PC:** Windows desktop with GTX 1050 Ti
- **Python:** Installed
- **GPU use:** faster-whisper for local transcription (fallback when no captions)
- **Quick start:**
  1. `git pull` in echo-audio-browser directory
  2. `pip install -r requirements.txt`
  3. Create `.env` with `OPENAI_API_KEY=sk-...`
  4. `start_server.bat` or `python -m uvicorn app.main:app --port 8765`
  5. Browser: http://localhost:8765/player

## Google Drive Audio Folder
- **URL:** https://drive.google.com/drive/folders/1-ctNQPU0s2zqzdq74h-FNQFJItgrLMbT
- **Purpose:** Source audio files for Echo ingestion

