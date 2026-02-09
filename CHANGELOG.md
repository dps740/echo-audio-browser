
## 2026-02-08: V3 Persistence Fix

### Problem
V3 segment-refined endpoint only cached results in memory. Data was lost on server restart.

### Root Cause
`test_v3.py` used `_cache` (Python dict) instead of ChromaDB for storing refined segments.

### Fix
1. **Added ChromaDB persistence** to `test_v3.py`:
   - `_get_v3_collection()` - Gets/creates `v3_segments` collection
   - `_save_to_chromadb()` - Stores sentences + refined segments after LLM processing
   - `_load_from_chromadb()` - Loads from ChromaDB when cache is empty
   - `LoadedRefinedSegment` class - Reconstructs objects from stored JSON

2. **Updated endpoints**:
   - `POST /v3/segment-refined/{video_id}` - Now calls `_save_to_chromadb()` after processing
   - `GET /v3/search-refined/{video_id}` - Loads from ChromaDB if not in memory cache

### Result
- Indexed episodes now survive server restarts
- Search works immediately after restart (loads from ChromaDB)
- Cost: ~$0.02/episode for indexing, search is free

---

## 2026-02-06: Timestamp Resolution Fix

### Problem
LLM segmentation identified good topics but 100% of timestamp resolutions failed because VTT parser created entire subtitle lines as single "words" instead of individual tokens.

### Root Cause
`youtube_ingest.py` was creating TranscriptWord objects where each "word" was an entire subtitle line:
- Expected: `["Uh", "any", "uh", "post", "Davos", "WF", "impressions?"]`
- Actual: `["Uh any uh post Davos WF impressions? We...", ...]`

### Fix
1. **youtube_ingest.py**: Added `_segments_to_words()` function that splits VTT segments into individual word-level tokens with interpolated timestamps.

2. **segmentation.py**: Improved matching functions:
   - Added `_normalize_word()` for aggressive text normalization
   - Updated `_find_quote_position()` to use normalized comparison
   - Enhanced `_fuzzy_find_quote()` with filler word filtering and difflib fallback

### Result
- Before: 0% success rate
- After: 100% success rate (5/5 test segments resolved)
