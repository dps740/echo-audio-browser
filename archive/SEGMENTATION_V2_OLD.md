# Echo Segmentation V2

**Last Updated:** 2026-02-06
**Status:** In production

---

## Overview

V2 segmentation extracts short, focused segments from podcast transcripts with rich topic tagging for better searchability.

## Key Improvements Over V1

| Aspect | V1 | V2 |
|--------|----|----|
| Segment length | 5-10 min avg | 1-2 min avg |
| Topic tagging | Single topic | Primary + secondary topics |
| Ad filtering | None | Skips ads, intros, outros |
| Transcript chunking | Truncated at 50K chars | 15-min chunks processed separately |

---

## Segment Structure

```json
{
  "start_ms": 120000,
  "end_ms": 240000,
  "content_type": "content",
  "primary_topic": "AI job displacement fears",
  "secondary_topics": ["hotel industry automation", "call center jobs", "ChatGPT"],
  "summary": "2-3 sentences capturing the specific insight...",
  "density_score": 0.85
}
```

### Fields

| Field | Description |
|-------|-------------|
| `start_ms` / `end_ms` | Timestamp bounds in milliseconds |
| `content_type` | `content` (indexed), `ad`, `intro`, `outro` (skipped) |
| `primary_topic` | Main subject discussed (2-5 words, specific) |
| `secondary_topics` | Other things mentioned (entities, examples, tangents) |
| `summary` | 2-3 sentence description with names, numbers, details |
| `density_score` | 0-1 rating of information density |

---

## Segmentation Prompt

The LLM prompt instructs:

1. **Skip non-content**: Ads, sponsor reads, intros, outros, theme music
2. **Target 1-3 minute segments**: Break long discussions into atomic ideas
3. **Primary + secondary topics**: Capture main theme AND other things discussed
4. **Specific tags**: "hotel cleanliness standards" not "business"

### Content Type Detection

- `content` — Actual discussion (indexed & searchable)
- `ad` — Sponsor reads ("this episode brought to you by...")
- `intro` — Show intro, theme music, housekeeping
- `outro` — Closing remarks, subscribe prompts

Only `content` segments are stored in the database.

---

## Chunking for Long Episodes

Episodes longer than ~22 minutes are chunked into 15-minute pieces:

```
Episode: 2.5 hours (150 min)
→ Chunk 1: 0-15m → LLM call → segments
→ Chunk 2: 15-30m → LLM call → segments
→ ... (10 chunks total)
→ All segments combined
```

This prevents truncation that caused V1 to only segment the beginning of long episodes.

---

## Search Pipeline

1. **Smart Search** with LLM relevance filtering
2. Query expansion (synonyms, related terms)
3. Full-text search on transcript + summary + tags
4. LLM filters for actual relevance (not just keyword match)

### Why This Matters

Search "hotel" now finds:
- Segments where hotels are the PRIMARY topic
- Segments where hotels are mentioned as SECONDARY topic
- Segments where hotels appear in transcript but weren't tagged

V1 would miss the secondary mentions entirely.

---

## Files

| File | Purpose |
|------|---------|
| `app/services/segmentation.py` | Core segmentation logic + prompt |
| `app/services/smart_search.py` | LLM-filtered search |
| `app/services/vectordb.py` | ChromaDB storage |
| `app/routers/playlists.py` | API endpoints (uses smart search) |
| `scripts/reingest_v2.py` | Batch re-ingestion script |

---

## Re-ingestion Process

```bash
cd ~/clawd/projects/echo-audio-browser
source venv/bin/activate

# Re-ingest all episodes
python3 scripts/reingest_v2.py

# Or limit to N episodes
python3 scripts/reingest_v2.py --limit 6

# Dry run (no changes)
python3 scripts/reingest_v2.py --dry-run
```

### Requirements

- Audio files in `audio/` folder (MP3)
- Transcript files in `audio/` folder (JSON3 from YouTube captions)
- OpenAI API key configured

### Timing

- ~15 seconds per 15-min chunk (LLM call)
- ~2-3 minutes per 1-hour episode
- ~15 minutes for 6 episodes

---

## Google Drive Audio Source

**URL:** https://drive.google.com/drive/folders/1-ctNQPU0s2zqzdq74h-FNQFJItgrLMbT

Contains:
- MP3 audio files
- JSON3 transcript files (YouTube captions)

Download with:
```bash
cd audio
gdown --folder "https://drive.google.com/drive/folders/1-ctNQPU0s2zqzdq74h-FNQFJItgrLMbT"
```

---

## Example Results

**Episode:** Joe Rogan #2446 - Greg Fitzsimmons (2.5 hours)

**V1:** 6 segments, avg 6+ min
**V2:** 58 segments, avg 0.9 min

**Sample V2 segments:**

| Time | Primary | Secondary |
|------|---------|-----------|
| 1-2m | Impact of social media exposure | Dunbar's number, mental health |
| 6-7m | FBI gang takedown operation | Latin Kings, drug seizures |
| 9-10m | Bodies found in Lake Mead | drought, crime investigation |
| 35-36m | AI and mental health concerns | ChatGPT, suicide encouragement |

---

## Future Enhancements (Documented)

See `FUTURE_FEATURES.md` and `META_INTELLIGENCE_LAYER.md` for:
- Actionable intelligence extraction
- "Tools of Titans" auto-generation
- Consensus/disagreement mapping
- Opinion shift detection
- Named entity extraction

These are documented but not yet implemented.

---

## Cost

Using gpt-4o-mini:
- ~$0.02-0.03 per episode
- Full library (10 episodes): ~$0.20-0.30
