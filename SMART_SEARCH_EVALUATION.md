# Smart Search Evaluation

## Summary

**Smart search is significantly better than hybrid search for precision.**

| Query | Hybrid Search | Smart Search | Winner |
|-------|--------------|--------------|--------|
| AGI | 3 ❌ (irrelevant results) | 0 ✅ (correctly empty) | **Smart** |
| Tesla | 0 | 0 | Tie |
| infinity | 3 ✅ | 3 ✅ | Tie |
| cryptocurrency | 3 ✅ | 3 ✅ | Tie |
| dementia | 3 ✅ | 3 ✅ | Tie |

**Key difference:** Hybrid search returns garbage for queries where no relevant content exists. Smart search correctly returns empty results.

## Architecture

### Old: Hybrid Search (~0.5s)
```
Query → Embedding → Vector similarity → Keyword boost → Filter → Results
```
- Fast but imprecise
- Returns semantically similar but irrelevant content
- "AGI" matches "healthcare AI" because embeddings are close

### New: Smart Search (~2-5s)
```
Query → LLM Expansion → Fulltext Search → LLM Relevance Filter → Results
```
1. **Query Expansion**: "AGI" → ["AGI", "artificial general intelligence", "superintelligence", ...]
2. **Fulltext Search**: Find segments containing ANY expanded term
3. **LLM Filter**: Review candidates and keep only substantive discussions

## Latency

| Scenario | Time |
|----------|------|
| Cold cache (first query) | 5.4s |
| Warm cache (same query) | 1.9s |
| New query (segments cached) | 3.0s |

Acceptable for a quality-first search experience.

## Cost

Per query:
- Expansion: ~$0.0003 (100 tokens in, 50 out)
- Filtering: ~$0.002 (2000 tokens in, 200 out)
- **Total: ~$0.002/query**

At 1000 queries/day = $2/day = $60/month

## Recommendation

**Use Smart Search as the primary search method.**

The hybrid search can remain as a "fast but fuzzy" fallback for:
- Autocomplete suggestions
- Quick browsing
- When speed > precision

Smart search should be used for:
- Actual topic searches
- Playlist generation
- Any user-facing search

## API Endpoints

- `/segments/search` - Old hybrid search (fast, imprecise)
- `/segments/smart-search` - New smart search (slower, precise)

## Future Improvements

1. **Batch LLM calls**: Send multiple segments per request
2. **Streaming**: Return results as they're validated
3. **Pre-compute popular expansions**: Cache common topic expansions
4. **Smaller model for filtering**: Try gpt-4o-mini for filtering (already using it)
5. **Async processing**: Non-blocking expansion + filtering
