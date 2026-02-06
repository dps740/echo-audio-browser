# Echo Search Improvements

**Last Updated:** 2026-02-06 19:30 UTC
**Status:** Both fixes implemented - quote-anchored timestamps + satisfaction-based search

---

## Recent Fixes (2026-02-06)

### Fix 1: Quote-Anchored Timestamps for Segmentation
**Problem:** LLM-generated timestamps were often wrong - summary/tags were correct but start_ms/end_ms didn't match the actual content.

**Solution:** Removed timestamp fields from LLM output. Instead require quote anchors:
- `starts_with`: First ~5 words of the segment (exact quote from transcript)
- `ends_with`: Last ~5 words of the segment (exact quote from transcript)

Then `resolve_timestamps()` function finds those quotes in the word-level transcript data:
1. Exact match - slides through words looking for quote sequence
2. Fuzzy match fallback - allows partial matches (60%+ overlap)
3. Duration validation - flags segments outside 30s-5min range

**Files changed:** `app/services/segmentation.py`

**Test results:**
```
[0] 89s - 127s (38s) - Topic: Davos 2023 impressions ✓
[1] 207s - 257s (50s) - Topic: Howard Lutnik's speech ✓
[2] 421s - 463s (42s) - Topic: Europe's reliance on US ✓
[3] 494s - 523s (30s) - Topic: Trump's Greenland comments (flagged: 29.5s < 30s min)
[4] 601s - 888s (287s) - Topic: Minneapolis immigration operations ✓
```

Timestamps now match actual content. Anomalies flagged but still included.

---

### Fix 2: Search Relevance Filter - Satisfaction-Based
**Problem:** Relevance filter asked "is this related?" but should ask "would user be satisfied?"

**Solution:** Changed `filter_with_llm()` to:
1. Evaluate on TRANSCRIPT text (not just summary) for ground truth
2. Ask: "Would someone searching for X feel SATISFIED playing this clip? Does it DELIVER?"
3. Score based on satisfaction, not mere relevance

**Files changed:** `app/services/smart_search.py`

**Key prompt change:**
```
The question is NOT "is this related to {query}?"
The question IS "Would someone searching for '{query}' feel SATISFIED after playing this clip?"
```

**Test results for "AI" search:**
- Results now include satisfaction-based reasoning
- Filter correctly identifies substantive AI discussions
- Reason examples: "provides substantial content relevant to the search" vs just "mentions AI"

---

## Current Problems Identified

### 1. Short Segments
- Some segments are <5 seconds (e.g., 3 sec clips)
- Should have minimum duration filter (30 sec?)
- Fix: Filter at index time or search time

### 2. Query Expansion Not Smart Enough
- Current: LLM generates related terms but may miss domain-specific entities
- Example: Search "AI" should find "Kimmy K2.5" (an AI model) even if "Kimmy" isn't in hardcoded list
- **David's direction: NO HARDCODING - use AI to be smart dynamically**

### 3. Results Don't Always Match Query
- Second result for "AI" didn't actually mention AI in audio
- Need better verification that content actually discusses the search term

---

## David's Key Insight

**"We need to use AI to be smart, not hardcode endless shit"**
**"We can use an element of keyword matching, but it shouldn't be our main basis"**

The search should:
1. Use keywords for **candidate finding** (fast, efficient)
2. Use LLM for **actual relevance decisions** (smart, semantic)
3. LLM understands that "Kimmy K2.5" IS relevant to "AI" without hardcoding
4. Keywords are a tool, not the answer

---

## Current Architecture

### Segmentation (V2 - Working)
```
Transcript → LLM chunks into 1-2 min segments
Each segment gets:
- primary_topic: Main subject
- secondary_topics: Other things mentioned  
- summary: 2-3 sentences
- density_score: 0-1
- content_type: content/ad/intro/outro
```

### Search Pipeline
```
Query: "AI"
  ↓
1. Query Expansion (LLM)
   - Generates related terms dynamically
   - Current prompt asks for specific models, companies, people
  ↓
2. Full-Text Search
   - Searches transcript, summary, tags for expanded terms
  ↓
3. LLM Relevance Filter
   - Reviews candidates: "Is this substantive discussion?"
  ↓
4. Return ranked results
```

### The Gap
The relevance filter should understand that "Kimmy K2.5" IS an AI model without us telling it. The LLM has this knowledge - we just need to ask it correctly.

---

## Correct Architecture (David's Design)

```
1. User searches "AI"
      ↓
2. LLM generates keyword list:
   "What keywords would appear in discussions about AI?"
   → [GPT-4, ChatGPT, OpenAI, Kimmy, Llama, neural network, Sam Altman...]
      ↓
3. Basic keyword search across all segments
   (fast, finds candidates with ANY of those terms)
      ↓
4. LLM filters and ranks:
   "Which of these are the BEST clips for someone interested in AI?"
   → Returns top ranked clips
```

**Key principles:**
- Keywords for speed (step 3)
- LLM for smarts (steps 2 and 4)
- LLM generates keywords dynamically (knows Kimmy = AI term)
- LLM picks BEST clips, not just "relevant enough"

---

## Files to Modify

| File | Purpose |
|------|---------|
| `app/services/smart_search.py` | Query expansion + relevance filtering |
| `app/services/segmentation.py` | V2 segmentation prompt |
| `scripts/reingest_v2.py` | Re-ingestion script |

---

## Current State (2026-02-06 18:10 UTC)

- **Segments indexed:** 203 across 4 episodes
- **Server:** Running on port 8765
- **Tunnel:** https://maximize-pdas-snowboard-dns.trycloudflare.com
- **Search:** Using smart-search with LLM expansion + filtering

### Episodes Indexed
1. 10 Years of Acquired (with Michael Lewis) - 62 segments
2. ICE Chaos in Minneapolis (All-In) - 37 segments  
3. Morgan Housel: What You Need to Endure - 53 segments
4. Joe Rogan #2447 - Mike Benz - 51 segments

### Episodes Missing Transcripts
- lYXKjuIHQIU
- EV7WhVT270Q

---

## Google Drive
- Audio files: https://drive.google.com/drive/folders/1-ctNQPU0s2zqzdq74h-FNQFJItgrLMbT

---

## Key Documentation Files
- `SEGMENTATION_V2.md` - V2 segmentation approach
- `FUTURE_FEATURES.md` - Intelligence layer concepts
- `META_INTELLIGENCE_LAYER.md` - Advanced analysis ideas
- `TOOLS_OF_TITANS_FEATURE.md` - Auto-generate compilations
- `ECHO_PROJECT.md` - Main project overview
