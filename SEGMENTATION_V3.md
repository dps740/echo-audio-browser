# Echo Segmentation V3 Design

## Overview

V3 replaces the LLM-based segmentation with a sentence-level embedding approach that achieves:
- **99% coverage** (vs 49% with current approach)
- **80-85% cheaper** indexing ($0.01 vs $0.06 per episode)
- **No LLM in search path** (faster, cheaper queries)

## Architecture

### Index-Time Pipeline

```
Audio File
    ↓
Whisper Transcription (word-level timestamps)
    ↓
Sentence Segmentation (pause-based, 800ms threshold)
    ↓
Embed Each Sentence (text-embedding-3-small)
    ↓
Store: sentences + embeddings + timestamps
    ↓
Async: Generate topic labels via LLM (non-blocking)
```

### Search-Time Pipeline

```
Query "AI"
    ↓
Parallel:
    a) Embed query → Vector search sentences (top-30)
    b) Full-text keyword search (top-20)
    ↓
Merge results (Reciprocal Rank Fusion)
    ↓
Expand each hit to playable context:
    - Backward to previous pause/topic shift
    - Forward to next pause/topic shift
    - CAP at 3 minutes max
    ↓
Deduplicate overlapping results
    ↓
Return top-5 with snippets + play buttons
```

## Key Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Pause threshold | 800ms | Natural sentence boundaries |
| Embedding model | text-embedding-3-small | Cheap ($0.02/1M tokens), good quality |
| Vector search top-k | 30 | Enough candidates for variety |
| Keyword search top-k | 20 | Safety net for exact matches |
| Max segment length | 180s (3 min) | Prevents runaway segments |
| Min segment length | 15s | Merge shorter segments |
| Context expansion | ±30s or to boundary | Natural playback start/end |

## Boundary Detection Algorithm

```python
def find_boundaries(embeddings, sentences):
    """
    Find topic boundaries using similarity drops.
    
    Two mechanisms:
    1. Sharp drop: adjacent sentence similarity < 0.15 (10th percentile)
    2. Max length: force split if segment exceeds 180 seconds
    """
    boundaries = []
    
    # Calculate all adjacent similarities
    sims = [cosine_sim(embeddings[i-1], embeddings[i]) 
            for i in range(1, len(embeddings))]
    
    # Adaptive threshold: 10th percentile of similarities
    threshold = np.percentile(sims, 10)
    
    last_boundary_time = sentences[0].start_ms
    
    for i in range(1, len(embeddings)):
        current_time = sentences[i].start_ms
        segment_duration = current_time - last_boundary_time
        
        # Force boundary if segment too long
        if segment_duration > 180_000:  # 3 minutes
            boundaries.append(i)
            last_boundary_time = current_time
            continue
        
        # Natural boundary if similarity drops
        if sims[i-1] < threshold:
            boundaries.append(i)
            last_boundary_time = current_time
    
    return boundaries
```

## Post-Processing

### Merge Short Segments
```python
def merge_short_segments(segments, min_duration=15_000):
    """Merge segments shorter than 15 seconds with neighbors."""
    merged = []
    for seg in segments:
        if merged and (seg.end_ms - seg.start_ms) < min_duration:
            # Merge with previous
            merged[-1].end_ms = seg.end_ms
        else:
            merged.append(seg)
    return merged
```

### Context Expansion for Search Results
```python
def expand_to_context(sentence_idx, sentences, boundaries):
    """Expand a search hit to playable context."""
    # Find containing segment
    seg_start = 0
    seg_end = len(sentences) - 1
    
    for b in boundaries:
        if b <= sentence_idx:
            seg_start = b
        if b > sentence_idx:
            seg_end = b - 1
            break
    
    # Return with padding
    start_ms = max(0, sentences[seg_start].start_ms - 3000)  # 3s padding
    end_ms = sentences[seg_end].end_ms + 3000
    
    return start_ms, end_ms
```

## Cost Comparison

| Operation | Current (LLM) | V3 (Embeddings) |
|-----------|---------------|-----------------|
| Index 1 episode | ~$0.06 | ~$0.01 |
| 1 search query | ~$0.003 | ~$0.00001 |
| 1000 episodes | ~$60 | ~$10 |
| 10,000 queries | ~$30 | ~$0.10 |

## Storage Requirements

Per episode (90 min, ~700 sentences):
- Embeddings: 700 × 1536 × 4 bytes = ~4.3 MB
- Metadata: ~50 KB
- **Total: ~4.5 MB per episode**

1000 episodes = ~4.5 GB (fits easily in Qdrant/Chroma)

## Quality Metrics

Target metrics for V3:
- Coverage: >95% (vs current ~50%)
- Search latency p95: <200ms
- No segments >3 min
- No segments <15s (after merging)
- Topic label accuracy: >80% (async LLM labeling)

## Migration Path

1. **Phase 1**: Implement V3 indexing alongside V2
2. **Phase 2**: Run both, compare search results
3. **Phase 3**: A/B test with users
4. **Phase 4**: Deprecate V2 if V3 wins

## Files Changed

- `app/services/segmentation_v3.py` — New segmentation logic
- `app/services/search_v3.py` — New search with expansion
- `app/routers/ingest.py` — Add V3 ingest endpoint
- `app/routers/search.py` — Add V3 search endpoint

## Testing

Run evaluation script:
```bash
cd ~/clawd/projects/echo-audio-browser
source venv/bin/activate
python scripts/final_comparison.py
```

## History

- 2026-02-08: Initial design after adversarial review with sub-agents
- Key insight: Current approach only indexes 49% of content
- 10 flaws identified in iterative review, addressed in V3
