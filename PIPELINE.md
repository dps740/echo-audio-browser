# Echo Audio Browser - Pipeline Documentation

## Overview

Topic-first podcast search. Index episodes once, search instantly, get playable MP3 clips.

---

## Two-Stage Workflow

The pipeline is split between **Desktop** (has YouTube access) and **Server** (has compute):

```
┌─────────────────────────────────────────────────────────────┐
│  DESKTOP (Windows PC)                                       │
│  ─────────────────────                                      │
│  • Run Echo UI with "Download Only" mode                    │
│  • Batch download MP3 + VTT subtitles from YouTube          │
│  • Files saved to app/downloads/{podcast_name}/             │
│  • Upload folder to Google Drive                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼ (Google Drive)
┌─────────────────────────────────────────────────────────────┐
│  SERVER (AWS)                                               │
│  ────────────                                               │
│  • Download files from Google Drive                         │
│  • Convert MP3 → WAV (16kHz mono)                           │
│  • Parse VTT subtitles                                      │
│  • Run V3 indexing pipeline (embeddings + LLM refinement)   │
│  • Store in ChromaDB for search                             │
└─────────────────────────────────────────────────────────────┘
```

**Why split?** AWS servers are geo-blocked from YouTube. Desktop has YouTube access but limited compute. Server has compute but no YouTube access.

---

## Stage 1: Desktop Download

### Using the UI

1. Run Echo Audio Browser: `python -m uvicorn app.main:app --port 8765`
2. Open http://localhost:8765/player
3. Select **"📥 Download Only (MP3 + VTT)"** from dropdown
4. Add podcasts to queue (curated list or custom channels)
5. Click **Start Batch**
6. Files saved to `app/downloads/{podcast_name}/`:
   - `{video_id}.mp3` - Audio file
   - `{video_id}.en.vtt` - YouTube auto-captions

### API Endpoint

```bash
POST /ingest/youtube/download-only
{
  "channel": "@lexfridman",
  "podcast_name": "Lex Fridman",
  "limit": 10
}
```

### Output Structure

```
app/downloads/
├── Lex Fridman/
│   ├── abc123xyz.mp3
│   ├── abc123xyz.en.vtt
│   ├── def456uvw.mp3
│   └── def456uvw.en.vtt
└── All-In Podcast/
    ├── ghi789rst.mp3
    └── ghi789rst.en.vtt
```

### Upload to Drive

Upload entire podcast folder to shared Google Drive location for server processing.

---

## Stage 2: Server Processing

### Step 1: Download from Drive

Download podcast folder from Google Drive to server.

### Step 2: Convert to WAV

```bash
# For each MP3, create WAV (required for accurate clip extraction)
ffmpeg -i video.mp3 -ar 16000 -ac 1 video.wav
```

**Why WAV?** MP3 has variable bitrate seeking issues. WAV allows frame-accurate clip extraction.

### Step 3: Place Files

```bash
# Move to audio/ folder with correct structure
mv video.wav app/audio/{video_id}.wav
mv video.en.vtt app/audio/{video_id}.en.vtt
```

### Step 4: Run V3 Indexing

```bash
POST /v3/segment-refined/{video_id}
```

**Pipeline steps:**
1. **Parse VTT** → Extract words with timestamps from YouTube captions
2. **Sentence segmentation** → Group words by 800ms pauses
3. **Embed sentences** → text-embedding-3-small for each sentence
4. **Detect topic boundaries** → Find where embedding similarity drops
5. **LLM refinement** → For each segment:
   - Find true topic start (not mid-conversation)
   - Generate SPECIFIC snippet (not vague summaries)
6. **Store** → Segments with boundaries, snippets, embeddings in ChromaDB

**Persistence:** Data stored in `chroma_data/` directory. Survives server restarts.
- Collection: `v3_segments`
- Stores: sentences (with embeddings), refined segments (with snippets)
- Search loads from ChromaDB if not in memory cache

**Cost:** ~$0.02 per episode (embeddings + LLM calls)

---

## Search-Time Pipeline

Instant search using pre-computed segments.

```
GET /v3/search-refined/{video_id}?q=AI
```

**Steps:**
1. **Embed query** → Same embedding model
2. **Vector search** → Find sentences matching query
3. **Segment topic filtering** → Check if segment is *about* the topic (not just mentions it)
4. **Return results** → Pre-computed snippets, no LLM calls
5. **Generate clip** → Extract MP3 from WAV (cached after first request)

**Cost:** Free (no API calls, just vector similarity)

### Segment Topic Relevance

A segment might contain a keyword but not be *about* that topic.

**Example:** Query "AI" might match a geopolitics segment that briefly mentions "AI" — but the segment isn't about AI.

**Solution:** Check if the segment's overall topic relates to the query:
1. Average all sentence embeddings in segment
2. Compare segment topic embedding to query embedding
3. Only include if relevance > 0.40

---

## Clip Extraction

When a search result is returned, clips are extracted on-demand:

```
WAV source file
    ↓
ffmpeg extracts segment (start_ms to end_ms)
    ↓
Convert to MP3 (128kbps)
    ↓
Cache in audio/clips/
    ↓
Serve to user
```

Clips are cached - subsequent requests for same segment are instant.

---

## API Endpoints

### Desktop (Download Only)

| Endpoint | Purpose |
|----------|---------|
| `POST /ingest/youtube/download-only` | Batch download MP3 + VTT |
| `GET /ingest/youtube/downloads` | List downloaded files |
| `GET /ingest/youtube/jobs` | Check download progress |

### Server (Processing)

| Endpoint | Purpose |
|----------|---------|
| `POST /v3/segment-refined/{video_id}` | Index episode (run once) |
| `GET /v3/search-refined/{video_id}?q=` | Search with pre-computed snippets |
| `GET /v3/clip/{video_id}?start_ms=&end_ms=` | Generate MP3 clip |

### Either (Full Pipeline - if YouTube accessible)

| Endpoint | Purpose |
|----------|---------|
| `POST /ingest/youtube/channel` | Download + index (v1, YouTube captions) |
| `POST /ingest/youtube/v2/channel` | Download + index (v2, MFA alignment) |

---

## Example Search Response

```json
{
  "query": "AI",
  "results": [
    {
      "start_ms": 3094640,
      "end_ms": 3144120,
      "start_formatted": "51:34",
      "end_formatted": "52:24",
      "duration_s": 49.5,
      "score": 0.447,
      "snippet": "The speaker predicts that 2024 will mark the rise of personal AI assistants...",
      "clip_url": "/audio/clips/gXY1kx7zlkk_c8461b2c91d1.mp3",
      "boundary_refined": true
    }
  ]
}
```

---

## Key Design Decisions

1. **Split workflow** - Desktop downloads (has YouTube), Server processes (has compute)
2. **YouTube VTT timestamps are accurate** - No MFA/Whisper needed
3. **WAV for extraction, MP3 for delivery** - WAV ensures accurate seeking
4. **LLM at index time only** - Boundaries and snippets computed once. Search is free.
5. **Specific snippets** - LLM prompted to be specific, not vague
6. **Segment topic filtering** - Filters passing mentions (relevance < 0.40)

---

## Cost Summary

| Operation | Cost |
|-----------|------|
| Download (Desktop) | Free |
| Index 1 episode (90 min) | ~$0.02 |
| Search query | Free |
| 1000 episodes indexed | ~$20 |

---

## Files

| File | Purpose |
|------|---------|
| `app/routers/youtube_ingest.py` | Download endpoints |
| `app/services/segmentation_v3.py` | VTT parsing, embeddings |
| `app/services/segment_refiner.py` | LLM refinement |
| `app/services/search_v3.py` | Vector search |
| `app/services/clip_extractor.py` | WAV → MP3 extraction |
| `app/routers/test_v3.py` | V3 API endpoints |

---

## Verified: 2026-02-08

Split workflow implemented. Download Only mode tested.
