"""Enhanced hybrid search: semantic + keyword matching with full pipeline."""

import chromadb
from typing import List, Dict, Any, Optional
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
    
    return list(set(expanded))  # Remove duplicates


def keyword_match_score(text: str, query_terms: List[str]) -> float:
    """
    Calculate keyword match score for text.
    Returns boost score based on keyword presence.
    """
    text_lower = text.lower()
    score = 0.0
    matched_any = False
    
    for term in query_terms:
        term_lower = term.lower()
        if term_lower in text_lower:
            matched_any = True
            # Position bonus: earlier matches score higher
            pos = text_lower.find(term_lower)
            early_bonus = max(0, 1 - (pos / len(text_lower)))
            # Frequency bonus: multiple mentions matter
            count = text_lower.count(term_lower)
            score += 0.3 + (early_bonus * 0.2) + (min(count - 1, 3) * 0.1)
    
    return score, matched_any


def hybrid_search(
    query: str,
    limit: int = 10,
    chroma_path: str = "./chroma_data",
    min_density: float = 0.0,
    diversity: bool = False,
) -> List[Dict[str, Any]]:
    """
    Full 5-step hybrid search pipeline:
    1. Semantic search → top 50 candidates
    2. Keyword filter & boost
    3. Diversity filter (max 3 per episode, max 4 per podcast)
    4. Quality filter (min score threshold)
    5. Return top N
    
    Args:
        query: Search query
        limit: Number of results to return
        chroma_path: Path to ChromaDB storage
        min_density: Minimum density score filter
        diversity: Apply diversity filtering
        
    Returns:
        List of ranked segments with relevance scores
    """
    client = chromadb.PersistentClient(path=chroma_path)
    
    try:
        collection = client.get_collection("segments")
    except Exception:
        return []
    
    # Expand query with synonyms
    query_terms = expand_query(query)
    
    # Step 1: Semantic search - get more candidates for filtering
    n_candidates = min(limit * 5, 50)
    
    try:
        results = collection.query(
            query_texts=[query],
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
    
    for i, (seg_id, doc, meta, dist) in enumerate(zip(
        results['ids'][0],
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        # Check density filter
        density = meta.get("density_score", 0.7)
        if density < min_density:
            continue
        
        # Base semantic similarity (convert distance to similarity)
        semantic_score = 1 - dist
        
        # Keyword matching across all fields
        transcript = doc.lower()
        summary = meta.get("summary", "").lower()
        topic_tags = meta.get("topic_tags", "").lower()
        key_terms = meta.get("key_terms", "").lower()
        
        # Calculate keyword scores for each field
        transcript_boost, transcript_match = keyword_match_score(transcript, query_terms)
        summary_boost, summary_match = keyword_match_score(summary, query_terms)
        tags_boost, tags_match = keyword_match_score(topic_tags, query_terms)
        terms_boost, terms_match = keyword_match_score(key_terms, query_terms)
        
        # Check if ANY field matched keywords
        has_keyword = transcript_match or summary_match or tags_match or terms_match
        
        # Calculate total keyword boost
        keyword_boost = transcript_boost + summary_boost + tags_boost + terms_boost
        
        # Combined score with heavy penalty for zero keyword match
        if has_keyword:
            final_score = semantic_score + keyword_boost
        else:
            # Heavy penalty: multiply by 0.3
            final_score = semantic_score * 0.3
        
        scored_segments.append({
            "segment_id": seg_id,
            "episode_id": meta["episode_id"],
            "episode_title": meta["episode_title"],
            "podcast_title": meta["podcast_title"],
            "audio_url": meta["audio_url"],
            "start_ms": meta["start_ms"],
            "end_ms": meta["end_ms"],
            "summary": meta["summary"],
            "topic_tags": meta.get("topic_tags", "").split(",") if meta.get("topic_tags") else [],
            "density_score": density,
            "relevance_score": final_score,
            "has_keyword": has_keyword,
            "semantic_score": semantic_score,
            "keyword_boost": keyword_boost,
        })
    
    # Sort by score (keyword matches first, then by score)
    scored_segments.sort(key=lambda x: (-x['has_keyword'], -x['relevance_score']))
    
    # Step 3: Diversity filter
    if diversity:
        diverse_segments = []
        episode_counts = {}
        podcast_counts = {}
        
        for seg in scored_segments:
            episode_id = seg["episode_id"]
            podcast = seg["podcast_title"]
            
            # Track counts
            episode_count = episode_counts.get(episode_id, 0)
            podcast_count = podcast_counts.get(podcast, 0)
            
            # Apply diversity limits
            if episode_count >= 3:  # Max 3 per episode
                continue
            if podcast_count >= 4:  # Max 4 per podcast
                continue
            
            diverse_segments.append(seg)
            episode_counts[episode_id] = episode_count + 1
            podcast_counts[podcast] = podcast_count + 1
            
            if len(diverse_segments) >= limit:
                break
        
        scored_segments = diverse_segments
    
    # Step 4: Quality filter - minimum score threshold
    # Only apply if we have enough results
    if len(scored_segments) > limit:
        # Calculate dynamic threshold (median score of top results)
        sorted_scores = sorted([s['relevance_score'] for s in scored_segments[:limit * 2]], reverse=True)
        if sorted_scores:
            min_threshold = sorted_scores[len(sorted_scores) // 2] * 0.5
            scored_segments = [s for s in scored_segments if s['relevance_score'] >= min_threshold]
    
    # Step 5: Return top N
    return scored_segments[:limit]
