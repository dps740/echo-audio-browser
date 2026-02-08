# Echo Segmentation V3 Design

## ✅ STATUS: VERIFIED WORKING (2026-02-08)

**YouTube VTT timestamps are accurate** when used with WAV audio files.

Testing confirmed: clip content matched expected timestamps. Initial confusion was a UX issue (snippet showed best-scoring match from middle of clip, not clip start).

**Final Pipeline (No MFA needed):**
1. Download audio + YouTube VTT/json3 transcript
2. Convert audio to WAV (for accurate seeking)
3. Parse VTT word-level timestamps
4. Segment into sentences, embed, index

**Remaining fix:** Change snippet to show content from clip START, not best-scoring match.

---

## Overview

V3 replaces the LLM-based segmentation with a sentence-level embedding approach that achieves:
- **99% coverage** (vs 49% with current approach)
- **80-85% cheaper** indexing ($0.01 vs $0.06 per episode)
- **Adaptive clip length** based on query specificity
- **No fixed segments** - clips sized by the data itself

## Key Innovation: Cluster-Based Adaptive Clips

Instead of fixed-length segments, V3:
1. Indexes every sentence with embeddings
2. At search time, finds matching sentences
3. Clusters matches by time proximity
4. Clip length = cluster extent

**Result:** Broad queries ("AI") get long clips. Specific queries ("Chinese AI infrastructure") get short clips. The data determines the boundaries, not arbitrary rules.

## Architecture

### Index-Time Pipeline

```
YouTube Video
    ↓
Download: Audio (any format) + VTT/json3 transcript
    ↓
Convert audio to WAV (ffmpeg, 16kHz mono)
    ↓
Parse VTT → word-level timestamps
    ↓
Sentence Segmentation (pause-based, 800ms threshold)
    ↓
Embed Each Sentence (text-embedding-3-small)
    ↓
Store: sentences + embeddings + timestamps
```

**No MFA. No Whisper. No LLM segmentation.** YouTube VTT timestamps are accurate when clips are extracted from WAV.

### Search-Time Pipeline

```
Query "Chinese AI"
    ↓
Embed query
    ↓
Vector search: find top 50 matching sentences
    ↓
Keyword boost: +0.15 for literal matches
    ↓
Cluster by time: gap > 60s = new cluster
    ↓
Score clusters:
    - 40% best match score
    - 40% match count (density)
    - 20% average match score
    ↓
Return top 5 clusters as playable clips
```

## Example Results

From testing on All-In Podcast episode (90 min):

| Query | #1 Result | Why |
|-------|-----------|-----|
| "AI" | 4.6 min (11 matches) | Broad topic, many matches clustered |
| "Chinese AI" | 6.3 min (16 matches) | Substantial discussion |
| "Davos" | 8.8 min (23 matches) | Main topic segment |
| "Trump Greenland" | 2 min (7 matches) | Focused discussion |
| "immigration" | 3.7 min (11 matches) | Medium topic |

**Key insight:** Passing mentions (1 match) rank last. Dense discussions (many matches) rank first.

## Why No Hard Caps

We considered adding a 3-minute max segment length, but rejected it:
- A 22-minute continuous conversation about AI is legitimate
- Forcing arbitrary breaks creates artificial cuts
- The cluster approach handles this naturally: specific queries still return the relevant portion

## Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Pause threshold | 800ms | Natural sentence boundaries |
| Embedding model | text-embedding-3-small | Cheap ($0.02/1M tokens) |
| Vector search top-k | 50 | Enough for good clustering |
| Gap threshold | 60s | Separate discussions |
| Keyword boost | 0.15 | Ensures literal matches surface |

## Cluster Scoring Formula

```python
cluster_score = (
    0.4 * best_individual_score +      # Best sentence match
    0.4 * min(match_count / 10, 1.0) +  # Density (caps at 10)
    0.2 * average_match_score           # Overall quality
)
```

This ensures:
- Single mentions don't outrank discussions
- Dense clusters rank higher
- Quality still matters

## Cost Comparison

| Operation | Current (LLM) | V3 (Embeddings) |
|-----------|---------------|-----------------|
| Index 1 episode | ~$0.06 | ~$0.01 |
| 1 search query | ~$0.003 | ~$0.0001 |
| 1000 episodes | ~$60 | ~$10 |

## Coverage Comparison

| Approach | Coverage | Reason |
|----------|----------|--------|
| Current (V2) | ~49% | LLM only indexes "identified topics" |
| V3 | ~99% | Every sentence indexed |

## API Endpoints

### Segment an Episode
```
POST /v3/segment/{video_id}

Response:
{
    "episode": "gXY1kx7zlkk",
    "total_sentences": 728,
    "total_segments": 57,
    "coverage_pct": 98.6,
    "segments": [...],
    "stats": {
        "mean_duration_s": 93.4,
        "min_duration_s": 22.0,
        "max_duration_s": 212.4
    }
}
```

### Search with Adaptive Clips
```
GET /v3/search/{video_id}?q=Chinese+AI

Response:
{
    "query": "Chinese AI",
    "clusters": [
        {
            "start_ms": 381700,
            "end_ms": 419500,
            "start_formatted": "63:37",
            "end_formatted": "69:55",
            "duration_s": 378.0,
            "score": 0.712,
            "match_count": 16,
            "snippet": "of AI from being centrally..."
        },
        ...
    ],
    "total_sentences": 728,
    "total_matches": 50,
    "note": "Medium topic - several focused discussions found"
}
```

### Compare V3 vs Current
```
GET /v3/compare/{video_id}

Response:
{
    "v3": {"coverage_pct": 98.6, "segments": 57},
    "current": {"coverage_pct": 49.4, "segments": 28},
    "winner": {"coverage": "V3", "coverage_delta_pct": 49.2}
}
```

## Files

| File | Purpose |
|------|---------|
| `app/services/segmentation_v3.py` | Sentence extraction + embedding |
| `app/services/search_v3.py` | Cluster-based search |
| `app/routers/test_v3.py` | API endpoints |
| `SEGMENTATION_V3.md` | This doc |

## Design History

### 2026-02-08: Initial Design

**Process:**
1. Identified current approach only indexes 49% of content
2. Spawned adversarial agents to attack proposals (3 iterations)
3. Tested on 3 real episodes with embeddings
4. Discovered gradual topic drift problem (22-min segment)
5. Rejected hard caps in favor of cluster-based approach
6. Implemented and validated cluster scoring

**Key decisions:**
- No LLM in segmentation (cost, coverage)
- No fixed segment lengths (arbitrary)
- Cluster by time proximity (natural boundaries)
- Score by density, not just best match (quality ranking)

## Testing

```bash
cd ~/clawd/projects/echo-audio-browser
source venv/bin/activate

# Segment an episode
curl -X POST http://localhost:8765/v3/segment/gXY1kx7zlkk

# Search
curl "http://localhost:8765/v3/search/gXY1kx7zlkk?q=AI"
curl "http://localhost:8765/v3/search/gXY1kx7zlkk?q=Chinese+AI"
curl "http://localhost:8765/v3/search/gXY1kx7zlkk?q=Davos"

# Compare to current approach
curl http://localhost:8765/v3/compare/gXY1kx7zlkk
```

## Search Quality Controls

### Filtering Logic

A sentence matches if EITHER:
- Semantic similarity >= 0.40 (strong conceptual match), OR
- Query appears literally in text (keyword match)

This ensures:
- Nonsense queries return no matches
- Terms not mentioned don't return noise matches
- Strong semantic matches work even without exact keywords
- Weak semantic matches still surface if keyword appears

### Tested Edge Cases

| Query | Expected | Result |
|-------|----------|--------|
| "xyznotaword123" | No matches | ✓ Filtered |
| "DeepSeek" (not mentioned) | No matches | ✓ Filtered |
| "the" | Matches (appears literally) | ✓ Works |
| "AI" | Strong matches | ✓ 50 matches |

## Comprehensive Test Results

Tested on 3 episodes (1938 total sentences):

| Query | Episode 1 | Episode 2 | Episode 3 |
|-------|-----------|-----------|-----------|
| "healthcare" | 75s, 19 matches | no matches | no matches |
| "Trump" | 22s, 8 matches | 167s, 30 matches | 27s, 6 matches |
| "AI" | 355s, 50 matches | 472s, 50 matches | 239s, 50 matches |
| "money" | 112s, 21 matches | 158s, 10 matches | 189s, 32 matches |

Key observations:
- Clip length varies naturally by topic density
- Episodes with more discussion of a topic get longer clips
- Cross-episode search works correctly
- Scores differentiate relevance (healthcare 0.517 in Dr Oz, no matches elsewhere)

## Next Steps

1. **UI integration** - Show clusters with "play from here" buttons
2. **Async labeling** - Add LLM topic labels for browse UI
3. **Replace V2** - Once UI tested, deprecate old approach
