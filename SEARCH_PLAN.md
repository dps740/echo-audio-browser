# Echo Search & Indexing: The Plan

## The Problem

Searching "AGI" returns clips about Michael Ovitz, Dr. Oz on healthcare, and homelessness — none of which mention AGI. The current system has **three compounding failures**:

### Failure 1: Dumb Chunking
The YouTube ingest path (`youtube_ingest.py`) chunks transcripts into **fixed 60-second segments** with no semantic awareness. A 60-second chunk might start mid-sentence and end mid-thought. The document stored in ChromaDB is raw transcript text with a truncated "summary" (`text[:150]...`). No topic tags, no real summary, density score hardcoded at 0.7.

The LLM-powered segmentation (`segmentation.py`) exists but **isn't used** by the YouTube ingest path — it's only wired up for the older RSS ingest flow.

### Failure 2: Weak Embeddings
ChromaDB's default embedding model is `all-MiniLM-L6-v2` — a small, general-purpose model. It embeds raw transcript text, which is conversational, full of filler words, and lacks the semantic density needed for topic matching. "AGI" as a concept gets diluted in a sea of "uh", "you know", "like".

### Failure 3: No Keyword Awareness
Pure vector search has no concept of "this document literally contains the word AGI." A hybrid search module exists (`hybrid_search.py`) but **isn't wired into the playlist endpoints** — they still call the basic `vectordb.search_segments()`.

---

## The Solution: 3-Layer Search Architecture

### Layer 1: Smart Indexing (Ingest-Time)

**Goal:** Every segment stored in ChromaDB should be semantically meaningful, well-summarized, and richly tagged.

**Approach:**
1. **Use LLM segmentation for YouTube ingest** — wire `segmentation.py` into the YouTube pipeline instead of dumb 60-sec chunking
2. **Enrich the embedding document** — instead of raw transcript, embed a structured document:
   ```
   TOPIC: [topic tags]
   SUMMARY: [1-2 sentence summary]
   KEY TERMS: [extracted entities/concepts]
   TRANSCRIPT: [cleaned transcript excerpt]
   ```
3. **Extract key terms at ingest time** — use a lightweight LLM call (gpt-4o-mini) to extract named entities, concepts, and keywords from each segment. Store these as metadata.
4. **Better topic tags** — the LLM segmentation already produces these, but they need to be actually used and stored properly.

**Cost:** ~$0.01-0.02 per episode (gpt-4o-mini for segmentation + key term extraction). Negligible.

**Implementation:**
- Modify `_ingest_to_chromadb()` in `youtube_ingest.py` to call the LLM segmentation pipeline
- Add a key term extraction step
- Build the enriched document for embedding
- Store key terms in metadata for keyword filtering

### Layer 2: Better Embeddings

**Goal:** Use an embedding model that understands topical relevance, not just surface similarity.

**Options (ranked by quality/practicality):**

| Model | Dims | Quality | Speed | Cost |
|-------|------|---------|-------|------|
| ChromaDB default (all-MiniLM-L6-v2) | 384 | ⭐⭐ | Fast, local | Free |
| OpenAI text-embedding-3-small | 1536 | ⭐⭐⭐⭐ | API call | $0.02/1M tokens |
| OpenAI text-embedding-3-large | 3072 | ⭐⭐⭐⭐⭐ | API call | $0.13/1M tokens |
| Cohere embed-v3 | 1024 | ⭐⭐⭐⭐ | API call | $0.10/1M tokens |
| Local: nomic-embed-text | 768 | ⭐⭐⭐ | Local, slower | Free |

**Recommendation:** `text-embedding-3-small` — best bang for buck. A full podcast episode (~50 segments) costs less than $0.001 to embed. Quality is dramatically better than MiniLM for topical matching.

**Implementation:**
- Create a custom ChromaDB embedding function using OpenAI's API
- Re-index existing content with new embeddings (one-time migration)
- Configure in `config.py` as a setting

### Layer 3: Hybrid Retrieval (Query-Time)

**Goal:** Combine semantic search with keyword matching and intelligent re-ranking.

**Pipeline:**
```
User Query: "AGI"
    │
    ├─ Step 1: Semantic Search (ChromaDB) → top 50 candidates
    │
    ├─ Step 2: Keyword Filter & Boost
    │   ├─ Exact match in transcript → +0.5 score
    │   ├─ Match in topic tags → +0.3 score
    │   ├─ Match in key terms → +0.4 score
    │   ├─ Match in summary → +0.3 score
    │   └─ No keyword match at all → score × 0.3 (heavy penalty)
    │
    ├─ Step 3: Diversity Filter
    │   └─ Max 3 segments per episode, max 4 per podcast
    │
    ├─ Step 4: Quality Filter
    │   └─ Minimum combined score threshold
    │
    └─ Step 5: Return top N
```

**Key insight:** The "no keyword match" penalty is crucial. If someone searches "AGI" and a segment doesn't contain "AGI", "artificial general intelligence", or closely related terms, it should be heavily penalized regardless of vector similarity. This is what prevents Dr. Oz clips from appearing in AGI searches.

**Synonym/expansion:** For common terms, expand the query:
- "AGI" → also match "artificial general intelligence", "superintelligence", "ASI"
- "AI" → also match "artificial intelligence", "machine learning", "ML"
- This can be a simple lookup table or a lightweight LLM call

**Implementation:**
- Upgrade `hybrid_search.py` with the full pipeline above
- Wire it into the playlist endpoints (replace `vectordb.search_segments`)
- Add synonym expansion (start with manual table, optionally LLM-powered later)

---

## Migration Plan

### Phase 1: Quick Win — Wire Hybrid Search (30 min)
- Connect existing `hybrid_search.py` to playlist endpoints
- Add keyword penalty for non-matching results
- **Immediate improvement** with zero re-indexing

### Phase 2: Smart Indexing (2-3 hours)
- Wire LLM segmentation into YouTube ingest
- Add key term extraction
- Build enriched embedding documents
- Test with a few episodes

### Phase 3: Better Embeddings (1-2 hours)
- Implement OpenAI embedding function for ChromaDB
- Re-index all existing content
- Compare search quality before/after

### Phase 4: Full Hybrid Pipeline (1-2 hours)
- Implement the full 5-step retrieval pipeline
- Add synonym expansion
- Add relevance score display in UI
- Tune scoring weights based on testing

### Phase 5: Re-index Everything (1 hour, mostly waiting)
- Re-ingest all episodes with the new pipeline
- Verify search quality across diverse queries

---

## Bonus: Clip Transitions
While we're at it, add a short transition between clips:
- **Option A:** 500ms silence gap + subtle chime (bundled as a small audio file)
- **Option B:** Web Audio API crossfade (fade out 300ms → 200ms silence → fade in 300ms)
- **Recommendation:** Option B — no extra files needed, smooth UX

---

## Summary

The current system stores **raw transcript chunks with no semantic enrichment** and searches them with a **weak embedding model** using **pure vector similarity**. It's like searching Google by vibes only — no keywords, no ranking, no relevance signals.

The fix is three layers working together:
1. **Store smarter** — LLM-enriched segments with topics, summaries, key terms
2. **Embed better** — OpenAI embeddings instead of MiniLM
3. **Search harder** — Hybrid retrieval with keyword boosting and quality filtering

Total incremental cost per episode: ~$0.02-0.05 (LLM segmentation + embeddings). For a library of 100 episodes, that's under $5 to re-index everything.

---

## Decision Points for David

1. **OpenAI embeddings vs free local?** (Recommended: OpenAI — $0.02/1M tokens is basically free)
2. **Re-index existing content?** (Recommended: Yes — current index is junk)
3. **Start with Phase 1 quick win or go straight to full rebuild?** (Recommended: Phase 1 first for instant improvement, then iterate)
