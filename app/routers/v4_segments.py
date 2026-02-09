"""
V4 Segments - Snippet-based embedding storage.

Key changes from V3:
- Store segment-level, not sentence-level
- Embed snippets directly (LLM summaries)
- 160 vectors per episode instead of 1600
- Direct semantic search on what segments are ABOUT
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import chromadb
import openai

from app.config import settings

router = APIRouter(prefix="/v4", tags=["v4-segments"])

# Persistent ChromaDB client
_chroma_client = chromadb.PersistentClient(path="./chroma_data")


def _get_segments_collection():
    """Get or create the V4 segments collection."""
    return _chroma_client.get_or_create_collection(
        name="v4_segments",
        metadata={"description": "Snippet-embedded segments for semantic search"}
    )


def embed_snippets(snippets: List[str]) -> List[List[float]]:
    """Embed snippets using OpenAI API."""
    client = openai.OpenAI(api_key=settings.openai_api_key)
    
    # Batch embed all snippets
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=snippets
    )
    
    return [e.embedding for e in response.data]


def save_segments_v4(
    video_id: str,
    episode_title: str,
    refined_segments: list
):
    """
    Save segments with snippet embeddings to ChromaDB.
    
    Each segment becomes a separate document with:
    - ID: {video_id}_{index}
    - Embedding: snippet embedding
    - Document: snippet text
    - Metadata: timestamps, video_id, title
    """
    collection = _get_segments_collection()
    
    # First, delete any existing segments for this episode
    try:
        existing = collection.get(where={"video_id": video_id})
        if existing['ids']:
            collection.delete(ids=existing['ids'])
            print(f"Deleted {len(existing['ids'])} existing segments for {video_id}")
    except Exception:
        pass  # Collection might be empty
    
    # Extract snippets
    snippets = [seg.snippet for seg in refined_segments]
    
    # Embed all snippets in one batch
    print(f"Embedding {len(snippets)} snippets...")
    embeddings = embed_snippets(snippets)
    
    # Prepare data for ChromaDB
    ids = [f"{video_id}_{i}" for i in range(len(refined_segments))]
    
    metadatas = [
        {
            "video_id": video_id,
            "episode_title": episode_title,
            "segment_index": i,
            "start_ms": seg.start_ms,
            "end_ms": seg.end_ms,
            "duration_s": round((seg.end_ms - seg.start_ms) / 1000, 1),
            "original_start_ms": seg.original_start_ms,
            "boundary_refined": seg.start_ms != seg.original_start_ms
        }
        for i, seg in enumerate(refined_segments)
    ]
    
    # Add to collection
    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=snippets,
        metadatas=metadatas
    )
    
    print(f"Saved {len(refined_segments)} segments for {video_id} to v4_segments")


def search_segments_v4(
    query: str,
    top_k: int = 10,
    video_id: Optional[str] = None
) -> List[dict]:
    """
    Search segments using snippet embeddings.
    
    Args:
        query: Search query
        top_k: Number of results to return
        video_id: Optional - limit to specific episode
    
    Returns:
        List of matching segments with scores
    """
    collection = _get_segments_collection()
    
    # Embed query
    client = openai.OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    query_embedding = response.data[0].embedding
    
    # Build query params
    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"]
    }
    
    if video_id:
        query_params["where"] = {"video_id": video_id}
    
    # Query
    results = collection.query(**query_params)
    
    # Format results
    segments = []
    for i in range(len(results['ids'][0])):
        # Convert distance to similarity (ChromaDB returns L2 distance)
        # For normalized embeddings, similarity ≈ 1 - (distance²/2)
        distance = results['distances'][0][i]
        similarity = 1 - (distance ** 2 / 2)
        
        segments.append({
            "id": results['ids'][0][i],
            "snippet": results['documents'][0][i],
            "score": round(similarity, 3),
            **results['metadatas'][0][i]
        })
    
    return segments


# ============================================================================
# API Endpoints
# ============================================================================

class SearchResult(BaseModel):
    query: str
    results: List[dict]
    total_results: int


@router.get("/search", response_model=SearchResult)
async def search_all_episodes(q: str, top_k: int = 10):
    """
    Search across ALL episodes using snippet embeddings.
    
    This is the primary search endpoint for the USP:
    "Find segments about X across all my podcasts"
    
    Example: GET /v4/search?q=Chinese AI infrastructure
    """
    results = search_segments_v4(query=q, top_k=top_k)
    
    # Add clip URLs
    from app.services.clip_extractor import get_clip_url
    
    def format_time(ms: int) -> str:
        total_seconds = ms // 1000
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes}:{seconds:02d}"
    
    for r in results:
        r["clip_url"] = get_clip_url(r["video_id"], r["start_ms"], r["end_ms"])
        r["start_formatted"] = format_time(r["start_ms"])
        r["end_formatted"] = format_time(r["end_ms"])
    
    return SearchResult(
        query=q,
        results=results,
        total_results=len(results)
    )


@router.get("/search/{video_id}", response_model=SearchResult)
async def search_single_episode(video_id: str, q: str, top_k: int = 10):
    """
    Search within a specific episode.
    
    Example: GET /v4/search/BvhFuEp55X0?q=elongated skulls
    """
    results = search_segments_v4(query=q, top_k=top_k, video_id=video_id)
    
    # Add clip URLs
    from app.services.clip_extractor import get_clip_url
    
    def format_time(ms: int) -> str:
        total_seconds = ms // 1000
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes}:{seconds:02d}"
    
    for r in results:
        r["clip_url"] = get_clip_url(r["video_id"], r["start_ms"], r["end_ms"])
        r["start_formatted"] = format_time(r["start_ms"])
        r["end_formatted"] = format_time(r["end_ms"])
    
    return SearchResult(
        query=q,
        results=results,
        total_results=len(results)
    )


@router.get("/stats")
async def get_collection_stats():
    """Get stats about the V4 segments collection."""
    collection = _get_segments_collection()
    
    # Get all unique video_ids
    all_data = collection.get(include=["metadatas"])
    
    video_ids = set()
    for meta in all_data['metadatas']:
        video_ids.add(meta.get('video_id', 'unknown'))
    
    return {
        "total_segments": collection.count(),
        "total_episodes": len(video_ids),
        "episodes": list(video_ids)
    }


@router.delete("/episode/{video_id}")
async def delete_episode(video_id: str):
    """Delete all segments for an episode."""
    collection = _get_segments_collection()
    
    existing = collection.get(where={"video_id": video_id})
    if existing['ids']:
        collection.delete(ids=existing['ids'])
        return {"deleted": len(existing['ids']), "video_id": video_id}
    else:
        return {"deleted": 0, "video_id": video_id, "message": "Episode not found"}
