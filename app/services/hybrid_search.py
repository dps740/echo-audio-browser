"""Hybrid search: semantic + keyword matching."""

import chromadb
from typing import List, Dict, Any

def hybrid_search(
    query: str,
    limit: int = 10,
    chroma_path: str = "./chroma_data"
) -> List[Dict[str, Any]]:
    """
    Search with keyword boosting.
    Segments containing the query terms rank higher.
    """
    client = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_collection("segments")
    
    # Get semantic search results (more than needed)
    results = collection.query(
        query_texts=[query],
        n_results=limit * 5,
        include=["documents", "metadatas", "distances"]
    )
    
    if not results['ids'] or not results['ids'][0]:
        return []
    
    # Score and re-rank
    query_lower = query.lower()
    query_words = query_lower.split()
    
    scored = []
    for i, (seg_id, doc, meta, dist) in enumerate(zip(
        results['ids'][0],
        results['documents'][0],
        results['metadatas'][0],
        results['distances'][0]
    )):
        doc_lower = doc.lower()
        
        # Base semantic score (convert distance to similarity)
        semantic_score = 1 - dist
        
        # Keyword boost
        keyword_boost = 0
        for word in query_words:
            if word in doc_lower:
                # Boost more if keyword appears early
                pos = doc_lower.find(word)
                early_bonus = max(0, 1 - (pos / len(doc_lower)))
                keyword_boost += 0.5 + (early_bonus * 0.5)
        
        # Combined score
        final_score = semantic_score + keyword_boost
        
        scored.append({
            "segment_id": seg_id,
            "episode_id": meta["episode_id"],
            "episode_title": meta["episode_title"],
            "podcast_title": meta["podcast_title"],
            "audio_url": meta["audio_url"],
            "start_ms": meta["start_ms"],
            "end_ms": meta["end_ms"],
            "summary": meta["summary"],
            "topic_tags": meta.get("topic_tags", "").split(",") if meta.get("topic_tags") else [],
            "density_score": meta.get("density_score", 0.7),
            "relevance_score": final_score,
            "has_keyword": any(w in doc_lower for w in query_words),
        })
    
    # Sort by score, keyword matches first
    scored.sort(key=lambda x: (-x['has_keyword'], -x['relevance_score']))
    
    return scored[:limit]
