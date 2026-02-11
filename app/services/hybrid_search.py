"""Enhanced hybrid search: semantic + keyword matching with full pipeline."""

import chromadb
import openai
from typing import List, Dict, Any, Tuple

from app.config import settings

# Synonym expansion lookup table
SYNONYM_MAP = {
    "agi": ["artificial general intelligence", "superintelligence", "asi", "strong ai"],
    "ai": ["artificial intelligence", "machine learning", "ml", "deep learning", "neural networks"],
    "llm": ["large language model", "language model", "gpt", "transformer"],
    "ml": ["machine learning", "ai", "artificial intelligence"],
    "crypto": ["cryptocurrency", "bitcoin", "blockchain", "btc", "eth"],
    "bitcoin": ["btc", "cryptocurrency", "crypto"],
    "ethereum": ["eth", "cryptocurrency", "crypto"],
    "startup": ["company", "business", "venture", "entrepreneurship"],
    "meditation": ["mindfulness", "contemplation", "awareness"],
    "stoicism": ["stoic philosophy", "marcus aurelius", "epictetus"],
    "habits": ["routine", "behavior", "practice", "discipline"],
    "productivity": ["efficiency", "output", "performance", "time management"],
    "health": ["wellness", "fitness", "exercise", "nutrition"],
}


def expand_query(query: str) -> List[str]:
    """Expand query with synonyms."""
    query_lower = query.lower().strip()
    
    # Check for direct synonym match
    if query_lower in SYNONYM_MAP:
        return [query] + SYNONYM_MAP[query_lower]
    
    # Check if any synonym appears in query
    expanded = [query]
    for key, synonyms in SYNONYM_MAP.items():
        if key in query_lower:
            expanded.extend(synonyms)
    
    return list(set(expanded))


def keyword_match_score(text: str, query_terms: List[str]) -> Tuple[float, bool]:
    """
    Calculate keyword match score for text.
    Returns (boost_score, matched_any).
    """
    if not text:
        return 0.0, False
        
    text_lower = text.lower()
    score = 0.0
    matched_any = False
    
    for term in query_terms:
        term_lower = term.lower()
        if term_lower in text_lower:
            matched_any = True
            # Position bonus: earlier matches score higher
            pos = text_lower.find(term_lower)
            early_bonus = max(0, 1 - (pos / max(len(text_lower), 1)))
            # Frequency bonus: multiple mentions matter
            count = text_lower.count(term_lower)
            score += 0.3 + (early_bonus * 0.2) + (min(count - 1, 3) * 0.1)
    
    return score, matched_any


def hybrid_search(
    query: str,
    limit: int = 10,
    min_relevance: float = 0.1,
    diversity: bool = True,
) -> List[Dict[str, Any]]:
    """
    Full 5-step hybrid search pipeline for v4_segments:
    
    1. Semantic search → top 50 candidates
    2. Synonym expansion + keyword boost
    3. Diversity filter (max 3 per episode)
    4. Quality filter (min relevance threshold)
    5. Return top N
    
    Args:
        query: Search query
        limit: Number of results to return
        min_relevance: Minimum relevance score (0-1)
        diversity: Apply diversity filtering
        
    Returns:
        List of ranked segments with relevance scores
    """
    # Get v4_segments collection
    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    
    try:
        collection = client.get_collection("v4_segments")
    except Exception:
        return []
    
    # Expand query with synonyms
    query_terms = expand_query(query)
    
    # Embed query with OpenAI
    oai = openai.OpenAI(api_key=settings.openai_api_key)
    response = oai.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    query_embedding = response.data[0].embedding
    
    # Step 1: Semantic search - get more candidates for filtering
    n_candidates = min(limit * 5, 50)
    
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_candidates,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        print(f"ChromaDB query failed: {e}")
        return []
    
    if not results['ids'] or not results['ids'][0]:
        return []
    
    # Step 2: Keyword filtering and boosting
    scored_segments = []
    
    for i in range(len(results['ids'][0])):
        seg_id = results['ids'][0][i]
        snippet = results['documents'][0][i] if results['documents'] else ""
        meta = results['metadatas'][0][i] if results['metadatas'] else {}
        dist = results['distances'][0][i] if results['distances'] else 1.0
        
        # Base semantic similarity (convert L2 distance to similarity)
        # For normalized vectors: similarity ≈ 1 - (distance²/2)
        semantic_score = max(0, 1 - (dist ** 2 / 2))
        
        # Keyword matching on snippet (the main searchable content)
        snippet_boost, snippet_match = keyword_match_score(snippet, query_terms)
        
        # Also check episode title
        title = meta.get("episode_title", "")
        title_boost, title_match = keyword_match_score(title, query_terms)
        
        has_keyword = snippet_match or title_match
        keyword_boost = snippet_boost + (title_boost * 0.5)
        
        # Combined score
        if has_keyword:
            final_score = semantic_score + keyword_boost
        else:
            # Penalty for no keyword match, but don't zero it out
            final_score = semantic_score * 0.7
        
        scored_segments.append({
            "id": seg_id,
            "snippet": snippet,
            "video_id": meta.get("video_id", ""),
            "episode_title": meta.get("episode_title", ""),
            "podcast_name": meta.get("podcast_name", meta.get("episode_title", "")),
            "start_ms": meta.get("start_ms", 0),
            "end_ms": meta.get("end_ms", 0),
            "duration_s": meta.get("duration_s", 0),
            "segment_index": meta.get("segment_index", 0),
            "score": round(final_score, 3),
            "semantic_score": round(semantic_score, 3),
            "keyword_boost": round(keyword_boost, 3),
            "has_keyword": has_keyword,
        })
    
    # Sort by score (keyword matches prioritized, then by score)
    scored_segments.sort(key=lambda x: (-x['has_keyword'], -x['score']))
    
    # Step 3: Diversity filter
    if diversity:
        diverse_segments = []
        episode_counts = {}
        
        for seg in scored_segments:
            video_id = seg["video_id"]
            
            # Max 3 per episode
            if episode_counts.get(video_id, 0) >= 3:
                continue
            
            diverse_segments.append(seg)
            episode_counts[video_id] = episode_counts.get(video_id, 0) + 1
            
            if len(diverse_segments) >= limit:
                break
        
        scored_segments = diverse_segments
    
    # Step 4: Quality filter
    if min_relevance > 0:
        scored_segments = [s for s in scored_segments if s['score'] >= min_relevance]
    
    # Step 5: Return top N
    return scored_segments[:limit]
