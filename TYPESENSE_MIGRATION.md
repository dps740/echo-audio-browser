# Echo Audio Browser — Typesense Migration Plan

**Date:** 2026-02-10
**Status:** IN PROGRESS
**Goal:** Commercial product-ready podcast search with optimal architecture

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     DATA PIPELINE                          │
├─────────────────────────────────────────────────────────────┤
│ 1. INGEST: Existing VTT files (12 podcasts)                │
│ 2. PARSE: VTT → Sentences with timestamps                  │
│ 3. NER: Extract entities per sentence (spaCy)              │
│ 4. SEGMENT: LLM topic boundaries + summaries (4o-mini)     │
│ 5. INDEX: Sentences + Topics in Typesense                  │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     STORAGE                                │
├─────────────────────────────────────────────────────────────┤
│ Typesense "sentences":                                     │
│   {id, episode_id, topic_id, text, start_ms, end_ms,       │
│    people[], companies[], topics[]}                        │
│                                                            │
│ Typesense "topics":                                        │
│   {id, episode_id, summary, full_text, start_ms, end_ms,   │
│    people[], companies[], keywords[]}                      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                     SEARCH FLOW                            │
├─────────────────────────────────────────────────────────────┤
│ 1. Query → Typesense sentences (BM25 on text + entities)   │
│ 2. Group matched sentences by topic_id                     │
│ 3. Score topics: best_match × match_density                │
│ 4. Return topics ranked, with best sentence highlighted    │
└─────────────────────────────────────────────────────────────┘
```

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Search engine | Typesense (self-hosted) | Free, fast, hybrid search built-in |
| NER | spaCy (en_core_web_sm) | Fast, free, reliable |
| LLM | GPT-4o-mini | Cost-effective, optimized prompt |
| ChromaDB | Keep but don't use | No migration needed, clean separation |
| Audio clips | Dynamic slicing | Flexible while tuning, zero storage |
| UI | Keep existing | Works well, just swap backend |

---

## Implementation Phases

### Phase 1: Infrastructure ✅
- [x] Install Typesense on server (v27.1)
- [x] Create sentence schema
- [x] Create topic schema  
- [x] Health check endpoint
- **Checkpoint:** `typesense_installed = True`

### Phase 2: NER Pipeline ✅
- [x] Install spaCy + en_core_web_sm model
- [x] Build sentence parser with NER (sentence_parser.py)
- [x] Test on one episode (523 sentences, 89 people, 74 companies, 47 topics)
- **Checkpoint:** `ner_pipeline_ready = True`

### Phase 3: Topic Segmentation ✅
- [x] Implement optimized LLM prompt v2 (merge-biased, multi-criteria)
- [x] Chunked processing for long episodes (200 sentences/chunk)
- [x] Document final prompt in topic_segmentation_v2.py
- **Checkpoint:** `segmentation_optimized = True`

### Phase 4: Indexing ✅
- [x] Build index_sentences() function
- [x] Build index_topics() function  
- [x] Orchestrate full pipeline (pipeline_v2.py)
- [x] Index one episode end-to-end (523 sentences, 27 topics)
- **Checkpoint:** `indexing_works = True`

### Phase 5: Search API ✅
- [x] Implement /v2/search endpoint
- [x] Sentence search → topic grouping
- [x] Ranking: best_score × sqrt(match_count)
- [x] Return format with highlights and timestamps
- **Checkpoint:** `search_api_ready = True`

### Phase 6: Bulk Processing ✅
- [x] Process all 16 local episodes
- [x] Progress tracking
- [x] Error handling
- **Result:** 10,217 sentences, 473 topics, 16 episodes
- **Checkpoint:** `all_episodes_indexed = True`

### Phase 7: UI Integration ⬜
- [ ] Wire search to existing UI
- [ ] Test end-to-end
- **Checkpoint:** `integration_complete = True`

---

## ✅ MIGRATION COMPLETE (2026-02-10)

**Final Stats:**
- 10,217 sentences indexed
- 473 topics indexed  
- 16 episodes processed
- Search API: `/v2/search?q=...`

**Key Wins:**
- "Tesla" search returns only content mentioning Tesla (no semantic bleed)
- BM25 ranking by match frequency
- Highlights with exact timestamps
- ~50ms search response time

**Ready for UI integration.**

---

## Current Status

```json
{
  "current_phase": 7,
  "typesense_installed": true,
  "ner_pipeline_ready": true,
  "segmentation_optimized": true,
  "indexing_works": true,
  "search_api_ready": true,
  "all_episodes_indexed": true,
  "integration_complete": false,
  "last_updated": "2026-02-10T06:45:00Z",
  "stats": {
    "sentences": 10217,
    "topics": 473,
    "episodes": 16
  }
}
```

---

## Recovery Instructions

If context is lost mid-implementation:

1. Read this file for overall plan
2. Check "Current Status" section for progress
3. Check `app/services/typesense_*.py` files for implemented code
4. Run `python -c "import typesense; print('OK')"` to verify install
5. Check Typesense: `curl http://localhost:8108/health`
6. Resume from next unchecked item in current phase

---

## Files Created

| File | Purpose |
|------|---------|
| `TYPESENSE_MIGRATION.md` | This plan document |
| `app/services/typesense_client.py` | Typesense connection |
| `app/services/ner_extraction.py` | spaCy NER pipeline |
| `app/services/topic_segmentation_v2.py` | Optimized LLM segmentation |
| `app/services/typesense_indexer.py` | Indexing logic |
| `app/services/typesense_search.py` | Search logic |
| `app/routers/search_v2.py` | New search endpoints |

---

## Optimized Prompt (v2)

See `OPTIMIZED_PROMPT.md` for the final tuned prompt and iteration history.

---

## Episode Inventory

Local episodes to process:
```
audio/*.vtt — 12-16 VTT files
```

To list: `ls -la audio/*.vtt`
