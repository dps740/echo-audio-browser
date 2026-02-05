# Testing Guide for Search & Indexing Overhaul

## Quick Verification Tests

### 1. Test Import & Syntax ✅ DONE
```bash
cd ~/clawd/projects/echo-audio-browser
python3 -c "from app.routers import playlists; from app.services import hybrid_search; from app.config import get_embedding_function; print('✓ All imports successful')"
```

### 2. Start the Server
```bash
cd ~/clawd/projects/echo-audio-browser
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Test Hybrid Search API

**Test basic search:**
```bash
curl "http://localhost:8000/playlists/search?q=AGI&limit=5"
```

**Test topic search:**
```bash
curl "http://localhost:8000/playlists/topic/AI%20Safety?limit=5"
```

**Test daily mix:**
```bash
curl "http://localhost:8000/playlists/daily-mix?topics=AGI,Philosophy&limit=10"
```

**Expected:**
- Results should have `relevance_score`, `has_keyword`, `semantic_score`, `keyword_boost` fields
- Results without keyword matches should have low scores (penalty applied)
- Topic tags and key_terms should be populated

### 4. Test New Ingestion

**Ingest a single video:**
```bash
curl -X POST "http://localhost:8000/ingest/youtube/url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=VIDEO_ID"}'
```

**Check job status:**
```bash
curl "http://localhost:8000/ingest/youtube/jobs"
```

**Expected:**
- New segments should have:
  - `key_terms` in metadata
  - `topic_tags` from LLM
  - Real summaries (not just truncated text)
  - Density scores from LLM

### 5. Test Re-Index Endpoint

**Trigger re-index:**
```bash
curl -X POST "http://localhost:8000/ingest/reindex"
```

**Monitor progress:**
```bash
curl "http://localhost:8000/ingest/youtube/jobs"
```

**Expected:**
- Job type: "reindex"
- Status updates: queued → loading_segments → reprocessing → finalizing → complete
- All existing segments get key_terms extracted
- Collection rebuilt with OpenAI embeddings

### 6. Test Frontend Crossfade

1. Open browser to: `http://localhost:8000`
2. Navigate to "Player" tab
3. Search for a topic or load a playlist
4. Play multiple segments in sequence
5. Listen for crossfade transitions between clips

**Expected:**
- Smooth fade out → silence → fade in between clips
- No jarring audio cuts
- ~800ms total transition time

### 7. Verify Metadata Structure

**Query a segment directly from ChromaDB:**
```python
import chromadb
client = chromadb.PersistentClient(path="./chroma_data")
collection = client.get_collection("segments")

# Get first 5 segments
results = collection.get(limit=5, include=["metadatas", "documents"])

for i, (seg_id, meta, doc) in enumerate(zip(results["ids"], results["metadatas"], results["documents"])):
    print(f"\n=== Segment {i+1}: {seg_id} ===")
    print(f"Podcast: {meta.get('podcast_title')}")
    print(f"Episode: {meta.get('episode_title')}")
    print(f"Summary: {meta.get('summary')[:100]}...")
    print(f"Topic Tags: {meta.get('topic_tags')}")
    print(f"Key Terms: {meta.get('key_terms')}")
    print(f"Density: {meta.get('density_score')}")
    print(f"Document preview: {doc[:200]}...")
```

**Expected metadata fields:**
- `episode_id`
- `episode_title`
- `podcast_title`
- `audio_url`
- `start_ms`
- `end_ms`
- `summary` (real LLM summary)
- `topic_tags` (comma-separated, from LLM)
- `key_terms` (comma-separated, from GPT-4o-mini)
- `density_score` (0.0-1.0, from LLM)
- `source`

**Expected document format:**
```
TOPIC: [tags]
SUMMARY: [summary]
KEY TERMS: [terms]
TRANSCRIPT: [text...]
```

---

## Quality Tests

### Search Quality Before/After

**Test queries to verify improvement:**

1. **"AGI"** - Should return AGI-specific content, not random topics
2. **"AI Safety"** - Should return AI safety discussions
3. **"Stoicism"** - Should return philosophy content, not random
4. **"Bitcoin"** - Should return crypto content, not unrelated

**For each query, check:**
- ✅ Results contain the search term or synonyms
- ✅ Results ranked by relevance (keyword matches first)
- ✅ Diverse sources (not all from same podcast)
- ✅ No obviously irrelevant results in top 10

### Diversity Test

```bash
curl "http://localhost:8000/playlists/topic/AI?limit=20&diverse=true"
```

**Verify:**
- Max 3 segments per episode
- Max 4 segments per podcast
- Good variety in results

### Embedding Quality Test

**Compare semantic understanding:**
- Search "artificial general intelligence" and "AGI" should return similar results
- Search "machine learning" and "AI" should overlap significantly
- Synonym expansion should work automatically

---

## Performance Checks

### Ingestion Performance

**Measure time for single video:**
```bash
time curl -X POST "http://localhost:8000/ingest/youtube/url" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://www.youtube.com/watch?v=VIDEO_ID"}'
```

**Expected:**
- 1-hour podcast: ~2-5 minutes total
- Breakdown:
  - Caption extraction: 10-30s
  - Audio download: 30-120s
  - LLM segmentation: 10-30s
  - Key term extraction: 5-15s per segment
  - ChromaDB indexing: 5-10s

### Search Performance

```bash
time curl "http://localhost:8000/playlists/search?q=AGI&limit=10"
```

**Expected:**
- < 500ms for most queries
- OpenAI embedding API call adds ~100-200ms
- ChromaDB query is fast

### Re-Index Performance

**For 100 episodes with ~1000 segments total:**
- Expected time: 30-60 minutes
- Key term extraction: ~1-2s per segment
- Embedding: batched by ChromaDB
- Cost: ~$2-5 total

---

## Troubleshooting

### Common Issues

**Issue: "No module named 'openai'"**
```bash
pip install openai
```

**Issue: "No OpenAI API key configured"**
- Check `.env` file has `OPENAI_API_KEY=sk-...`
- Or set environment variable

**Issue: ChromaDB collection mismatch**
```python
# Reset collection if needed
import chromadb
client = chromadb.PersistentClient(path="./chroma_data")
client.delete_collection("segments")
```

**Issue: Crossfade not working**
- Check browser console for errors
- Ensure Web Audio API is supported (modern browsers)
- Try Chrome/Firefox/Safari

**Issue: LLM segmentation failing**
- Check OpenAI API key is valid
- Check API rate limits
- Fallback chunking should activate automatically

---

## Success Criteria

✅ **Phase 1:** All playlist endpoints return `has_keyword` and boosted scores  
✅ **Phase 2:** New segments have `key_terms` and enriched documents  
✅ **Phase 3:** ChromaDB uses OpenAI embeddings (check collection metadata)  
✅ **Phase 4:** Search results show diversity and keyword matching  
✅ **Phase 5:** Audio transitions are smooth with crossfade  

**Overall success = dramatically better search relevance + smooth UX**

---

## Next Steps After Testing

1. **Re-index existing library** (if search quality is good on new content)
2. **Tune scoring weights** if needed (in `hybrid_search.py`)
3. **Add more synonyms** to `SYNONYM_MAP` based on common queries
4. **Monitor costs** for OpenAI API usage
5. **Consider caching** key terms to reduce re-extraction costs

---

## Cost Tracking

Track API costs for the first week:
- OpenAI embeddings: $0.02 per 1M tokens
- GPT-4o-mini calls: $0.15 per 1M input tokens, $0.60 per 1M output tokens

**Typical usage for 1 episode (~50 segments):**
- Embeddings: ~$0.001
- Segmentation: ~$0.01
- Key terms: ~$0.005
- **Total: ~$0.02 per episode**

For 100 episodes = ~$2 total (very affordable!)
