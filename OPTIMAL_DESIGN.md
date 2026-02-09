# Optimal Indexing & Search Design (Revised)

*Updated after deep code review — incorporates all findings from tracing
actual data flows, cross-file imports, ChromaDB schemas, and frontend dependencies.*

---

## Current Flow (traced through actual code)

`POST /v3/segment-refined/{video_id}` in test_v3.py:399-479:

```
Step 1: segment_transcript_v3(vtt_content)      [segmentation_v3.py]
  → parse VTT → words → sentences (800ms pause)
  → embed ALL sentences (1 OpenAI API call, ~700 sentences)
  → find boundaries (percentile-based, 10th percentile)     ← ALGO #1
  → create segments + merge short
  → return (segments, sentences)

Step 2: detect_topic_boundaries(sentences, 0.5)  [segment_refiner.py]
  → THROW AWAY segments from step 1                         ← WASTED WORK
  → re-detect boundaries (fixed threshold 0.5)              ← ALGO #2
  → return (start_idx, end_idx) tuples

Step 3: refine_segments(sentences, boundaries)   [segment_refiner.py]
  → for each segment (~15):
      → LLM call: refine start boundary                     ← ~15 API calls
      → LLM call: generate snippet                          ← ~15 API calls
  → return List[RefinedSegment]

Step 4: save_segments_v4(video_id, "Episode {video_id}", refined)
  → embed snippets (1 OpenAI API call)                      [v4_segments.py]
  → store in ChromaDB v4_segments
  → HARDCODED TITLE: f"Episode {video_id}"                  ← NO REAL TITLES
```

### Problems Found

1. **Two boundary algorithms, first is wasted.** segmentation_v3 runs full
   boundary detection + segment creation + merge, then test_v3.py throws it
   all away and re-detects with a different algorithm. segment_transcript_v3()
   is called purely for its side effect of embedding sentences.

2. **~30 individual LLM calls per episode.** 15 for boundary refinement
   (marginal — adjusts timestamps by ~5s) + 15 for snippet generation
   (necessary). Sequential. ~45 seconds wall time.

3. **Episode titles are placeholders.** `save_segments_v4(video_id,
   f"Episode {video_id}", refined)` — every episode stored as "Episode
   gXY1kx7zlkk". No mechanism to pass real titles or podcast names. The
   indexing endpoint doesn't accept title/podcast params.

4. **No podcast-level grouping.** V4 metadata has `episode_title` but no
   `podcast_title`. library.py uses episode_title as podcast proxy. The
   download step knows the podcast name but it's never propagated to indexing.

5. **Frontend searches via playlists.py, not v4_segments.py.** The search
   bar calls `/playlists/topic/{topic}` → `search_segments_v4()` → ChromaDB.
   The `/v4/search` endpoint exists but the frontend never calls it.

6. **ChromaDB path hardcoded in 15 places.** All active files use
   `"./chroma_data"` instead of `settings.chroma_persist_dir`. The config
   setting is only used by dead code.

7. **V4 collection has no embedding_function.** Works because all queries
   pass pre-computed embeddings, but if anyone writes
   `collection.query(query_texts=["..."])` it would use ChromaDB's default
   model (384 dims) against OpenAI embeddings (1536 dims) → crash.

8. **Distance-to-similarity conversion assumes L2 metric.** Correct for
   current setup but would break silently if collection config changes.

9. **models.py not fully dead.** playlists.py imports PlaybackManifest +
   PlaybackSegment. These must survive.

10. **6-second timing offset in frontend** is dead code — only fires on
    fallback path when clips aren't pre-extracted. Clip extraction from WAV
    solved this.

---

## New Design

### Indexing Pipeline: 3 API calls, with natural starts

```
VTT file + episode title + podcast name
  │
  ▼
┌──────────────────────────────────────────────────────┐
│  PARSE + DETECT BOUNDARIES  (1 embedding API call)   │
│                                                      │
│  parse_vtt_to_words(vtt)                             │
│  → group into sentences (800ms pause)                │
│  → embed sentences (1 batched OpenAI call)           │
│  → find boundaries (percentile-based, from V3)       │
│  → merge short segments (<15s)                       │
│                                                      │
│  Output: ~15 segments as (start_idx, end_idx) +      │
│          3 context sentences before each boundary     │
└──────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────┐
│  NATURAL STARTS + SNIPPETS  (1 LLM call)             │
│                                                      │
│  Send ALL segments in ONE prompt, each with           │
│  3 context sentences before the boundary.            │
│  LLM picks natural start AND generates snippet.      │
│                                                      │
│  Output: 15 × {natural_start_sentence, snippet}      │
└──────────────────────────────────────────────────────┘
  │
  ▼
┌──────────────────────────────────────────────────────┐
│  STORE  (1 embedding API call)                       │
│                                                      │
│  Embed all snippets (1 batched OpenAI call)          │
│  Store in ChromaDB v4_segments with metadata:        │
│    video_id, episode_title, podcast_name,            │
│    start_ms, end_ms, duration_s                      │
│                                                      │
│  Output: ~15 vectors in v4_segments collection       │
└──────────────────────────────────────────────────────┘
```

**API calls:** 3 (sentence embeddings, batch LLM, snippet embeddings)
**vs current:** ~32
**Cost:** ~$0.003/episode (down from ~$0.02)
**Wall time:** ~5s (down from ~45s)

### Key improvements over current

- **One boundary algorithm** — percentile-based from segmentation_v3 (adaptive,
  respects max duration). Not the fixed-threshold one from segment_refiner.
- **Natural start detection** — LLM sees 3 context sentences before each
  boundary, picks where the topic introduction begins. Combined with snippet
  generation in one call. No extra API calls.
- **Real titles and podcast names** — indexing endpoint accepts title +
  podcast_name. Stored in ChromaDB metadata. Library and search show real names.
- **Batch snippet generation** — LLM sees all segments together, produces
  more differentiated snippets.

### Search: One endpoint, one path

```
Query → embed (1 API call) → ChromaDB vector search → clip URLs → response
```

Frontend search goes through `/search?q=...` (new unified endpoint).
Playlists use the same underlying `storage.search()`.
Per-episode search: `/search?q=...&video_id=...`

---

## File Structure

```
app/
├── main.py                    (~50 lines)
├── config.py                  (~30 lines)
├── models.py                  (~25 lines)   PlaybackManifest + PlaybackSegment only
├── services/
│   ├── vtt_parser.py          (~80 lines)   parse VTT → sentences
│   ├── segmentation.py        (~250 lines)  embed, boundaries, batch LLM
│   ├── storage.py             (~150 lines)  ChromaDB save/search/delete
│   └── clip_extractor.py      (~140 lines)  FFmpeg clip extraction (keep as-is)
├── routers/
│   ├── indexing.py             (~80 lines)   POST /index/{video_id}
│   ├── search.py              (~100 lines)  GET /search, playlist generation
│   ├── library.py             (~150 lines)  browsing, stats (updated)
│   ├── download.py            (~250 lines)  download-only + curated + jobs
│   └── playlists.py           (~200 lines)  playlist manifests (updated)
```

**Total: ~1,500 lines** (vs current ~11,500)

### ChromaDB Schema (v4_segments)

```python
{
    "id": "{video_id}_{index}",
    "embedding": [1536-dim OpenAI vector],
    "document": "snippet text",
    "metadata": {
        "video_id": "gXY1kx7zlkk",
        "episode_title": "Lex Fridman #412: AI and China",
        "podcast_name": "Lex Fridman Podcast",
        "segment_index": 3,
        "start_ms": 180000,
        "end_ms": 360000,
        "duration_s": 180.0
    }
}
```

New fields vs current: `podcast_name` added, `episode_title` gets real
values instead of placeholders. Removed: `original_start_ms`,
`boundary_refined` (internal debugging, not useful in storage).

Existing indexed episodes keep working — new fields are additive. Library
falls back to episode_title when podcast_name is absent.

---

## Migration: What to Keep / Delete / Build

### Keep as-is
- `clip_extractor.py` — working, clean

### Extract and refactor
- `segmentation_v3.py` → VTT parsing + sentence embedding + boundary detection
  into `services/vtt_parser.py` + `services/segmentation.py`
- `v4_segments.py` → ChromaDB operations into `services/storage.py`,
  API endpoints into `routers/search.py`
- `youtube_ingest.py` → download-only + curated list + job tracking
  (~250 lines) into `routers/download.py`
- `library.py` → update to use singleton ChromaDB client, read podcast_name
- `playlists.py` → update imports to use `services.storage.search`
- `models.py` → keep PlaybackManifest + PlaybackSegment, delete rest

### Build new
- `services/segmentation.py:generate_snippets_with_natural_starts()` —
  batch LLM call for snippets + natural start detection
- `routers/indexing.py` — clean indexing endpoint with title + podcast_name

### Delete (all dead)
```
services/segmentation.py        (997 lines - V1)
services/segmentation_v2.py     (217 lines)
services/mfa_align.py           (335 lines)
services/textgrid_parser.py     (302 lines)
services/pipeline.py            (682 lines)
services/hybrid_search.py       (222 lines)
services/smart_search.py        (350 lines)
services/vectordb.py            (215 lines)
services/transcription.py       (133 lines)
services/transcript_resolver.py (143 lines)
services/rss.py                 (121 lines)
services/search_v3.py           (265 lines)
routers/segments.py             (373 lines)
routers/feeds.py                (90 lines)
routers/ingest.py               (130 lines)
routers/test_v3.py              (593 lines)
routers/youtube_ingest.py       (1936 lines)
routers/v4_segments.py          (267 lines - replaced by services/storage.py + routers/search.py)
```
