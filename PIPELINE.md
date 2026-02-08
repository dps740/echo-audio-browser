# Echo Audio Browser - Pipeline Documentation

## Overview

Topic-first podcast search. Index episodes once, search instantly, get playable MP3 clips.

---

## Pre-Indexing (Manual Steps)

Before running the API, prepare audio files:

```bash
# 1. Download audio + transcript from YouTube
yt-dlp -x --audio-format mp3 --write-auto-sub --sub-lang en "VIDEO_URL"

# 2. Convert to WAV (required for accurate clip extraction)
ffmpeg -i video.mp3 -ar 16000 -ac 1 video.wav

# 3. Place files in audio/ folder
#    - VIDEO_ID.wav
#    - VIDEO_ID.en.vtt
```

**Why WAV?** MP3 has variable bitrate seeking issues. WAV allows frame-accurate clip extraction.

---

## Index-Time Pipeline

Run once per episode. Creates searchable segments with LLM-refined boundaries.

```
POST /v3/segment-refined/{video_id}
```

**Steps:**

1. **Parse VTT** → Extract words with timestamps from YouTube captions
2. **Sentence segmentation** → Group words by 800ms pauses
3. **Embed sentences** → text-embedding-3-small for each sentence
4. **Detect topic boundaries** → Find where embedding similarity drops
5. **LLM refinement** → For each segment:
   - Find true topic start (not mid-conversation)
   - Generate SPECIFIC snippet (not vague summaries)
6. **Store** → Segments with boundaries, snippets, embeddings

**Cost:** ~$0.02 per episode (embeddings + LLM calls)

**Output:** Pre-computed segments ready for instant search.

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

The key insight: a segment might contain a keyword but not be *about* that topic.

**Example:** Query "AI" might match a geopolitics segment that briefly mentions "AI or best-in-class" — but the segment isn't about AI.

**Solution:** Check if the segment's overall topic relates to the query:

1. Average all sentence embeddings in segment → "what is this segment about?"
2. Compare segment topic embedding to query embedding
3. Only include if relevance > 0.40

This filters out passing mentions and ensures results are genuinely about the query topic.

**Before:** Query "AI" returned geopolitics segment (relevance 0.38)
**After:** Only AI segments returned (relevance 0.41-0.48)

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

| Endpoint | Purpose |
|----------|---------|
| `POST /v3/segment-refined/{video_id}` | Index episode (run once) |
| `GET /v3/search-refined/{video_id}?q=` | Search with pre-computed snippets |
| `GET /v3/clip/{video_id}?start_ms=&end_ms=` | Generate MP3 clip |

---

## Example Response

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
      "snippet": "The speaker predicts that 2024 will mark the rise of personal AI assistants that will evolve beyond chatbots, transitioning from primarily serving as research tools to performing a variety of tasks more effectively.",
      "clip_url": "/audio/clips/gXY1kx7zlkk_c8461b2c91d1.mp3",
      "boundary_refined": true
    }
  ]
}
```

---

## Key Design Decisions

1. **YouTube VTT timestamps are accurate** - Tested and verified. No MFA/Whisper needed.

2. **WAV for extraction, MP3 for delivery** - WAV ensures accurate seeking; MP3 reduces file size for users.

3. **LLM at index time only** - Boundaries and snippets computed once. Search is free.

4. **Specific snippets** - LLM prompted to be specific ("personal AI assistants replacing chatbots") not vague ("discussion about AI").

5. **Boundary refinement** - LLM finds where topic actually starts, not just first matching sentence.

6. **Segment topic filtering** - Search checks if segment is *about* the query topic, not just contains keywords. Filters passing mentions (relevance < 0.40).

---

## Files

| File | Purpose |
|------|---------|
| `app/services/segmentation_v3.py` | VTT parsing, sentence segmentation, embeddings |
| `app/services/segment_refiner.py` | LLM boundary refinement + snippet generation |
| `app/services/search_v3.py` | Vector search |
| `app/services/clip_extractor.py` | WAV → MP3 extraction |
| `app/routers/test_v3.py` | API endpoints |

---

## Cost Summary

| Operation | Cost |
|-----------|------|
| Index 1 episode (90 min) | ~$0.02 |
| Search query | Free |
| 1000 episodes indexed | ~$20 |

---

## Verified: 2026-02-08

Tested on All-In Podcast episode (90 min, 700+ sentences, 144 segments).
Timestamps accurate. Clips play correctly. Snippets are specific.
