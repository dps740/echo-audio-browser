"""
Smart Search Pipeline - Quality over speed

Design:
1. Query Expansion (LLM): "AGI" → ["AGI", "artificial general intelligence", "superintelligence", ...]
2. Full-Text Search: Find segments containing ANY expanded term in transcript/summary/tags
3. LLM Relevance Filter: Batch review to find SUBSTANTIVE discussions (not passing mentions)

This trades some latency (~2-3s) for dramatically better precision.
"""

import json
import re
from typing import List, Dict, Any, Optional
import chromadb
from openai import OpenAI

from app.config import settings, get_embedding_function

# Caching
_segments_cache = None
_segments_cache_time = 0
CACHE_TTL = 300  # 5 minutes

_expansion_cache = {}


def get_openai_client():
    return OpenAI(api_key=settings.openai_api_key)


def get_all_segments(chroma_path: str = "./chroma_data") -> List[Dict]:
    """Load all segments from ChromaDB with caching."""
    global _segments_cache, _segments_cache_time
    import time
    
    now = time.time()
    if _segments_cache and (now - _segments_cache_time) < CACHE_TTL:
        return _segments_cache
    
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection("segments")
    
    # Get all segments
    results = collection.get(
        include=["documents", "metadatas"]
    )
    
    segments = []
    for i, seg_id in enumerate(results['ids']):
        meta = results['metadatas'][i]
        doc = results['documents'][i] if results['documents'] else ""
        segments.append({
            "segment_id": seg_id,
            "transcript": doc,
            "summary": meta.get("summary", ""),
            "topic_tags": meta.get("topic_tags", ""),
            "episode_id": meta.get("episode_id", ""),
            "episode_title": meta.get("episode_title", ""),
            "podcast_title": meta.get("podcast_title", ""),
            "audio_url": meta.get("audio_url", ""),
            "start_ms": meta.get("start_ms", 0),
            "end_ms": meta.get("end_ms", 0),
            "density_score": meta.get("density_score", 0.5),
        })
    
    # Update cache
    _segments_cache = segments
    _segments_cache_time = now
    
    return segments


def expand_query(query: str) -> List[str]:
    """Use LLM to expand query into related search terms (with caching)."""
    # Check cache
    cache_key = query.lower().strip()
    if cache_key in _expansion_cache:
        return _expansion_cache[cache_key]
    
    client = get_openai_client()
    
    prompt = f"""Expand this search query into related terms for finding podcast discussions.

Query: "{query}"

Think about what SPECIFIC things would be discussed in a podcast about this topic:
- The original term and synonyms/abbreviations
- Specific products, models, or tools (e.g., "AI" → GPT-4, Claude, Llama, Gemini)
- Companies and organizations (e.g., "AI" → OpenAI, Anthropic, Google DeepMind)
- Key people (e.g., "AI" → Sam Altman, Dario Amodei, Yann LeCun)
- Technical concepts that indicate discussion of this topic
- Slang or informal terms podcasters might use

Return 10-20 search terms as a JSON array. Be SPECIFIC - include actual names, products, models that would appear in discussions.

Examples:
- "AI" → ["AI", "artificial intelligence", "GPT-4", "ChatGPT", "Claude", "Llama", "OpenAI", "machine learning", "neural network", "Sam Altman", "large language model", "LLM"]
- "investing" → ["investing", "stocks", "Warren Buffett", "Berkshire", "index fund", "S&P 500", "portfolio", "dividends", "P/E ratio", "value investing"]
- "health" → ["health", "longevity", "creatine", "Zone 2", "VO2 max", "sleep", "Peter Attia", "Huberman", "supplements", "metabolic health"]

Return ONLY a valid JSON array, nothing else."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=500,
    )
    
    try:
        content = response.choices[0].message.content.strip()
        # Strip markdown code blocks if present
        if content.startswith("```"):
            content = re.sub(r'^```(?:json)?\n?', '', content)
            content = re.sub(r'\n?```$', '', content)
        terms = json.loads(content)
        # Always include original query
        if query not in terms and query.lower() not in [t.lower() for t in terms]:
            terms.insert(0, query)
        # Cache result
        _expansion_cache[cache_key] = terms
        return terms
    except Exception as e:
        print(f"Query expansion parse error: {e}")
        return [query]


def fulltext_search(segments: List[Dict], terms: List[str]) -> List[Dict]:
    """
    Search for segments containing any of the terms.
    Searches across transcript, summary, and topic_tags.
    Returns segments with match info.
    """
    matches = []
    
    # Build regex pattern for case-insensitive matching
    # Use word boundaries to avoid partial matches
    patterns = [re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE) for term in terms]
    
    for seg in segments:
        searchable = f"{seg['transcript']} {seg['summary']} {seg['topic_tags']}"
        
        matched_terms = []
        match_count = 0
        
        for term, pattern in zip(terms, patterns):
            finds = pattern.findall(searchable)
            if finds:
                matched_terms.append(term)
                match_count += len(finds)
        
        if matched_terms:
            seg_copy = seg.copy()
            seg_copy['matched_terms'] = matched_terms
            seg_copy['match_count'] = match_count
            matches.append(seg_copy)
    
    # Sort by match count (more matches = likely more substantive)
    matches.sort(key=lambda x: -x['match_count'])
    
    return matches


def filter_with_llm(query: str, candidates: List[Dict], limit: int = 10) -> List[Dict]:
    """
    Use LLM to filter candidates based on whether playing the clip would SATISFY the searcher.
    
    Key insight: Don't ask "is this related?" - ask "would the user be satisfied playing this?"
    Evaluates on TRANSCRIPT text (not just summary) for ground truth.
    """
    if not candidates:
        return []
    
    client = get_openai_client()
    
    # Prepare segments for review - use TRANSCRIPT for evaluation
    segments_text = ""
    for i, seg in enumerate(candidates[:20]):  # Review up to 20 candidates
        # Use transcript for ground truth (truncate to ~400 chars for token efficiency)
        transcript = seg['transcript'][:400] if seg['transcript'] else ""
        if len(seg['transcript']) > 400:
            transcript += "..."
        
        segments_text += f"""
[{i}] ID: {seg['segment_id']}
TRANSCRIPT: {transcript}
Episode: {seg.get('episode_title', 'Unknown')}
---"""

    prompt = f"""You're helping someone who searched for "{query}".

Your job: Rate each clip on whether playing it would SATISFY the searcher.

The question is NOT "is this related to {query}?"
The question IS "Would someone searching for '{query}' feel SATISFIED after playing this clip? Does it DELIVER?"

SATISFACTION means:
- They get actual content about what they searched for
- Not just a brief mention or tangent
- They wouldn't feel like "that's not what I was looking for"

Rate each segment:
- 9-10: Perfect hit - exactly what they searched for, substantive content
- 7-8: Good hit - clearly delivers on the search intent
- 5-6: Okay - touches on the topic but might feel incomplete
- 1-4: Miss - they'd feel unsatisfied playing this (EXCLUDE)

IMPORTANT: Judge based on the TRANSCRIPT, not just keyword presence.

Segments to review:
{segments_text}

Return a JSON array with ONLY segments scoring >= 6:
[{{"index": <number>, "score": <1-10>, "reason": "<why would/wouldn't user be satisfied>"}}]

Return ONLY valid JSON. If nothing satisfies the search, return []."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1000,
    )
    
    try:
        content = response.choices[0].message.content.strip()
        # Strip markdown code blocks if present
        if content.startswith("```"):
            content = re.sub(r'^```(?:json)?\n?', '', content)
            content = re.sub(r'\n?```$', '', content)
        ratings = json.loads(content)
    except:
        # Try to extract JSON from response
        content = response.choices[0].message.content
        match = re.search(r'\[.*\]', content, re.DOTALL)
        if match:
            try:
                ratings = json.loads(match.group())
            except:
                return []
        else:
            return []
    
    # Build results
    results = []
    for rating in ratings:
        if rating.get('score', 0) >= 6:
            idx = rating.get('index', -1)
            if 0 <= idx < len(candidates):
                seg = candidates[idx].copy()
                seg['relevance_score'] = rating['score'] / 10.0
                seg['relevance_reason'] = rating.get('reason', '')
                results.append(seg)
    
    # Sort by score
    results.sort(key=lambda x: -x['relevance_score'])
    
    return results[:limit]


def smart_search(
    query: str,
    limit: int = 10,
    chroma_path: str = "./chroma_data",
    debug: bool = False
) -> Dict[str, Any]:
    """
    Smart search pipeline:
    1. Expand query using LLM
    2. Full-text search for expanded terms
    3. LLM filter for substantive discussions
    
    Returns dict with results and debug info.
    """
    result = {
        "query": query,
        "expanded_terms": [],
        "fulltext_matches": 0,
        "final_results": [],
        "debug": {} if debug else None
    }
    
    # Step 1: Load all segments (cache this in production)
    segments = get_all_segments(chroma_path)
    if debug:
        result["debug"]["total_segments"] = len(segments)
    
    # Step 2: Expand query
    expanded = expand_query(query)
    result["expanded_terms"] = expanded
    
    # Step 3: Full-text search
    candidates = fulltext_search(segments, expanded)
    result["fulltext_matches"] = len(candidates)
    
    if debug:
        result["debug"]["candidates_preview"] = [
            {"id": c["segment_id"], "matched": c["matched_terms"], "count": c["match_count"]}
            for c in candidates[:10]
        ]
    
    if not candidates:
        return result
    
    # Step 4: LLM filter
    filtered = filter_with_llm(query, candidates, limit)
    
    # Format results
    result["final_results"] = [
        {
            "segment_id": seg["segment_id"],
            "episode_id": seg["episode_id"],
            "episode_title": seg["episode_title"],
            "podcast_title": seg["podcast_title"],
            "audio_url": seg["audio_url"],
            "start_ms": seg["start_ms"],
            "end_ms": seg["end_ms"],
            "summary": seg["summary"],
            "topic_tags": seg["topic_tags"].split(",") if isinstance(seg["topic_tags"], str) else seg["topic_tags"],
            "density_score": seg["density_score"],
            "relevance_score": seg["relevance_score"],
            "relevance_reason": seg.get("relevance_reason", ""),
            "matched_terms": seg.get("matched_terms", []),
        }
        for seg in filtered
    ]
    
    return result


# Quick test function
if __name__ == "__main__":
    import sys
    query = sys.argv[1] if len(sys.argv) > 1 else "AGI"
    print(f"\n{'='*60}")
    print(f"SMART SEARCH TEST: {query}")
    print('='*60)
    
    result = smart_search(query, limit=5, debug=True)
    
    print(f"\nExpanded terms: {result['expanded_terms']}")
    print(f"Full-text matches: {result['fulltext_matches']}")
    print(f"Final results: {len(result['final_results'])}")
    
    for r in result['final_results']:
        print(f"\n  [{r['relevance_score']:.1f}] {r['segment_id']}")
        print(f"      {r['summary'][:100]}...")
        print(f"      Matched: {r['matched_terms']}")
        print(f"      Reason: {r['relevance_reason']}")
