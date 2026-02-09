# Optimal Indexing & Search Design

## Current Flow (traced through the actual code)

Here's what `POST /v3/segment-refined/{video_id}` actually does today
(`test_v3.py:399-479`):

```
Step 1: segment_transcript_v3(vtt_content)      [segmentation_v3.py]
  → parse VTT to words
  → group words into sentences (800ms pause)
  → embed ALL sentences (1 OpenAI API call, ~700 sentences)
  → find boundaries via percentile-based similarity drops    ← BOUNDARY ALGO #1
  → merge short segments
  → return (segments, sentences)

Step 2: detect_topic_boundaries(sentences)       [segment_refiner.py]
  → DISCARD the segments from step 1                        ← WASTED WORK
  → re-detect boundaries with fixed threshold (0.35)        ← BOUNDARY ALGO #2
  → return new (start_idx, end_idx) tuples

Step 3: refine_segments(sentences, boundaries)   [segment_refiner.py]
  → for each segment (~15):
      → LLM call: "where does this topic actually start?"   ← 15 API calls
      → LLM call: "generate a specific snippet"             ← 15 API calls
  → return List[RefinedSegment]

Step 4: save_segments_v4(video_id, title, refined)  [v4_segments.py]
  → embed all snippets (1 OpenAI API call)
  → store in ChromaDB v4_segments
```

**Problems:**

1. **Two different boundary detection algorithms** — segmentation_v3 uses percentile-based thresholds, then segment_refiner throws that away and re-detects with a fixed 0.35 threshold. The first boundary detection is wasted.

2. **~30 individual LLM calls per episode** — 15 for boundary refinement (marginal benefit) + 15 for snippet generation (necessary). These run sequentially. For a 90-min episode, this takes ~45 seconds of wall time.

3. **Boundary refinement is marginal** — the LLM is asked "should the segment start a few seconds earlier?" This adjusts timestamps by ~5 seconds. The 500ms padding in clip_extractor.py already compensates for imprecise boundaries.

4. **Two competing search endpoints** — `/v3/search-refined/{video_id}` does per-episode search with in-memory sentence embeddings (nuanced but doesn't scale). `/v4/search` does cross-episode ChromaDB search on snippet embeddings (simpler, scalable). Both exist, users don't know which to use.

---

## Optimal Design

### Indexing Pipeline: 3 API calls instead of 32

```
VTT file
  │
  ▼
┌─────────────────────────────────────────┐
│  PARSE + SEGMENT  (local, free)         │
│                                         │
│  parse_vtt_to_words(vtt)                │
│  → group into sentences (800ms pause)   │
│  → embed sentences (1 batched API call) │
│  → find boundaries (similarity drops)   │
│  → merge short segments (<15s)          │
│                                         │
│  Output: ~15 segments with text + times │
└─────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────┐
│  GENERATE SNIPPETS  (1 LLM call)        │
│                                         │
│  Send ALL segments in ONE prompt:       │
│  "Here are 15 segments from a podcast.  │
│   For each, write a specific one-       │
│   sentence description."                │
│                                         │
│  Output: 15 snippets                    │
└─────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────┐
│  STORE  (1 embedding API call)          │
│                                         │
│  Embed all snippets (1 batched call)    │
│  Store in ChromaDB: snippet + embedding │
│  + metadata (video_id, timestamps, etc) │
│                                         │
│  Output: ~15 vectors in v4_segments     │
└─────────────────────────────────────────┘
```

**API calls:** 3 (sentence embeddings, batch snippet generation, snippet embeddings)
**vs current:** ~32 (sentence embeddings, 15 boundary refinements, 15 snippet generations, snippet embeddings)
**Cost:** ~$0.003/episode (down from ~$0.02)
**Wall time:** ~5s (down from ~45s)

### Search Pipeline: Already optimal, just pick one

```
Query "Chinese AI policy"
  │
  ▼
┌─────────────────────────────────────────┐
│  SEARCH  (1 embedding API call)         │
│                                         │
│  Embed query                            │
│  ChromaDB vector search on snippets     │
│  (optional: filter by video_id)         │
│                                         │
│  Output: ranked segments with scores    │
└─────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────────────────┐
│  SERVE CLIPS  (cached ffmpeg)           │
│                                         │
│  For each result:                       │
│    extract WAV segment → MP3            │
│    cache in audio/clips/                │
│                                         │
│  Output: clip URLs in response          │
└─────────────────────────────────────────┘
```

This is exactly what V4 search already does. The only change is: **kill the V3 search path entirely.** One search endpoint, one approach.

---

## Why This Works Better

### Why drop boundary refinement?

The LLM boundary refinement (segment_refiner.py:63-124) asks the LLM to look at ~10 surrounding sentences and decide "should the segment start 5 seconds earlier?" This adds ~15 API calls per episode for a ~5-second timestamp adjustment that's masked by clip padding anyway.

If a boundary lands mid-sentence, the listener hears "...and that's why I think the real problem is—" which is slightly awkward. But the 500ms padding in clip_extractor already handles this. And embedding-based boundaries almost always land at natural topic shifts because that's literally what they detect.

### Why batch snippet generation?

Current: 15 individual LLM calls, each seeing one segment in isolation.
Proposed: 1 LLM call seeing all 15 segments together.

Benefits:
- **10x fewer API calls** → faster, cheaper, fewer failure points
- **Better snippets** — the LLM sees all segments, so it can differentiate: "this segment is about AI in China" vs "this one is about AI in healthcare" instead of both getting "discussion about AI"
- **Single prompt to optimize** — one well-crafted prompt instead of two (boundary + snippet)

The prompt:

```
Here are {n} segments from a podcast episode "{title}".
For each segment, write ONE specific sentence describing what is discussed.

BE SPECIFIC:
- Include names, numbers, specific claims
- Describe WHAT is said, not just the topic
- Bad: "Discussion about AI" → Good: "Why ChatGPT-style chatbots are just phase one of AI"

Segments:
[1] (0:00-3:45) {segment_1_text_truncated_to_500_chars}
[2] (3:45-8:20) {segment_2_text_truncated_to_500_chars}
...

Return a JSON array of {n} strings, one snippet per segment.
```

### Why snippet embedding is the right search representation

The key insight: you want to search by **what a segment is about**, not by what words appear in it.

A raw transcript segment about Chinese AI policy will contain words like "and", "you know", "he said", "I think", "the thing is" — noise that dilutes the embedding. The snippet "China's plan to lead AI by 2030 through massive government investment in chip manufacturing" is a pure signal of what the segment is about.

This also implicitly solves the "passing mention" problem. If a segment briefly mentions "AI" while mainly discussing stock markets, the snippet will say something about stock markets, and a search for "AI" won't match strongly. No explicit relevance filtering needed.

### Why one search endpoint, not two

The V3 per-episode search (`/v3/search-refined/{video_id}`) loads all sentence embeddings into memory and computes segment-level relevance with a 60/40 weighted score. It's more nuanced, but:

1. **It doesn't work cross-episode** — the entire point of Echo is searching across your library
2. **It requires sentence embeddings in memory** — at ~700 sentences × 1536 dimensions × 4 bytes = ~4MB per episode. For 100 episodes that's 400MB just for search
3. **The nuance is redundant** — snippet embeddings already capture segment-level topic relevance (see above)

V4 search (`/v4/search`) is the right answer: embed query, ChromaDB query, done. It already supports filtering by `video_id` for per-episode search.

---

## Implementation: What the Code Looks Like

### File 1: `services/segmentation.py` (~250 lines)

Merges the useful parts of segmentation_v3.py and segment_refiner.py:

```python
# VTT parsing (from segmentation_v3.py — keep as-is)
def parse_vtt_to_words(vtt_content: str) -> List[dict]
def words_to_sentences(words, pause_threshold_ms=800) -> List[Sentence]

# Embedding (from segmentation_v3.py — keep as-is)
def get_embeddings(texts: List[str]) -> np.ndarray

# Boundary detection (from segmentation_v3.py — keep as-is)
def find_boundaries(embeddings, sentences, max_segment_ms, percentile) -> List[int]
def create_segments(sentences, boundaries) -> List[Segment]
def merge_short_segments(segments, min_duration_ms) -> List[Segment]

# NEW: Batch snippet generation (replaces per-segment LLM calls)
def generate_snippets_batch(segments: List[Segment], episode_title: str) -> List[str]

# Main orchestrator
def index_episode(vtt_content: str, episode_title: str) -> List[IndexedSegment]:
    words = parse_vtt_to_words(vtt_content)
    sentences = words_to_sentences(words)
    embeddings = get_embeddings([s.text for s in sentences])
    # attach embeddings to sentences...
    boundaries = find_boundaries(embeddings, sentences)
    segments = create_segments(sentences, boundaries)
    segments = merge_short_segments(segments)
    snippets = generate_snippets_batch(segments, episode_title)
    return [IndexedSegment(seg, snippet) for seg, snippet in zip(segments, snippets)]
```

### File 2: `services/storage.py` (~120 lines)

Clean ChromaDB operations (evolved from v4_segments.py):

```python
def save_episode(video_id, title, segments: List[IndexedSegment])
    # embed snippets, store in ChromaDB

def search(query, top_k=10, video_id=None) -> List[SearchResult]
    # embed query, ChromaDB query, format results

def delete_episode(video_id)
def get_stats() -> dict
```

### File 3: `routers/indexing.py` (~80 lines)

```python
@router.post("/index/{video_id}")
async def index_episode(video_id: str):
    vtt = read_vtt(video_id)
    segments = segmentation.index_episode(vtt, title)
    storage.save_episode(video_id, title, segments)
    return {"segments": len(segments), "cost": ...}
```

### File 4: `routers/search.py` (~60 lines)

```python
@router.get("/search")
async def search_all(q: str, top_k: int = 10):
    results = storage.search(q, top_k)
    # add clip URLs
    return results

@router.get("/search/{video_id}")
async def search_episode(video_id: str, q: str, top_k: int = 10):
    results = storage.search(q, top_k, video_id=video_id)
    return results
```

### That's it.

~510 lines for the core indexing + search logic. Plus clip_extractor (140), download router (200), library (150), playlists (200), config (60), main (50) = **~1,300 lines total**.

---

## Migration Path

You don't need to rewrite from scratch. The code already exists — it just needs extraction:

1. **Copy** `parse_vtt_to_words`, `words_to_sentences`, `get_embeddings`, `find_boundaries`, `create_segments`, `merge_short_segments` from `segmentation_v3.py` into new `services/segmentation.py`

2. **Write** `generate_snippets_batch` — this is the only genuinely new function. ~40 lines. It's `generate_specific_snippet` from segment_refiner.py but batched into one prompt.

3. **Copy** `save_segments_v4`, `search_segments_v4`, `embed_snippets` from `v4_segments.py` into new `services/storage.py`

4. **Copy** the `/v4/search` endpoints from `v4_segments.py` into new `routers/search.py`

5. **Write** a new indexing endpoint that calls `segmentation.index_episode` → `storage.save_episode`. ~20 lines.

6. **Delete everything else.**

The data in ChromaDB (`v4_segments` collection) doesn't need to change. Existing indexed episodes keep working.
