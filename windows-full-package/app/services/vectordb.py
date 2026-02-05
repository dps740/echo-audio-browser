"""Vector database service using ChromaDB."""

from typing import List, Optional, Dict, Any
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings, get_embedding_function
from app.services.segmentation import SegmentResult


# Global client (lazy initialization)
_client: Optional[chromadb.Client] = None
_collection: Optional[chromadb.Collection] = None


def get_client() -> chromadb.Client:
    """Get or create ChromaDB client."""
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    return _client


def get_collection() -> chromadb.Collection:
    """Get or create the segments collection with custom embedding function."""
    global _collection
    if _collection is None:
        client = get_client()
        embedding_fn = get_embedding_function()
        _collection = client.get_or_create_collection(
            name="segments",
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
    return _collection


async def store_segments(
    segments: List[SegmentResult],
    episode_id: str,
    episode_title: str,
    podcast_title: str,
    audio_url: str,
) -> int:
    """
    Store segments in the vector database.
    
    Args:
        segments: List of segments to store
        episode_id: Episode ID
        episode_title: Episode title
        podcast_title: Podcast title
        audio_url: URL of the audio file
        
    Returns:
        Number of segments stored
    """
    collection = get_collection()
    
    ids = []
    documents = []
    metadatas = []
    
    for i, seg in enumerate(segments):
        segment_id = f"{episode_id}_{i}"
        
        # Document is the summary + transcript for embedding
        doc = f"{seg.summary}\n\n{seg.transcript_text[:2000]}"
        
        metadata = {
            "episode_id": episode_id,
            "episode_title": episode_title,
            "podcast_title": podcast_title,
            "audio_url": audio_url,
            "start_ms": seg.start_ms,
            "end_ms": seg.end_ms,
            "summary": seg.summary,
            "topic_tags": ",".join(seg.topic_tags),
            "density_score": seg.density_score,
        }
        
        ids.append(segment_id)
        documents.append(doc)
        metadatas.append(metadata)
    
    # Add to collection (ChromaDB handles embedding automatically)
    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
    )
    
    return len(segments)


async def search_segments(
    query: str,
    limit: int = 20,
    min_density: float = 0.0,
    topic_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search for segments matching a query.
    
    Args:
        query: Search query
        limit: Max results to return
        min_density: Minimum density score filter
        topic_filter: Filter by topic tag (optional)
        
    Returns:
        List of matching segments with scores
    """
    collection = get_collection()
    
    # Build where filter
    where = None
    if min_density > 0:
        where = {"density_score": {"$gte": min_density}}
    
    # Query
    results = collection.query(
        query_texts=[query],
        n_results=limit * 2,  # Get extra for post-filtering
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    
    # Process results
    segments = []
    
    if results["ids"] and results["ids"][0]:
        for i, segment_id in enumerate(results["ids"][0]):
            metadata = results["metadatas"][0][i]
            distance = results["distances"][0][i]
            
            # Apply topic filter if specified
            if topic_filter:
                tags = metadata.get("topic_tags", "").split(",")
                if topic_filter.lower() not in [t.lower() for t in tags]:
                    continue
            
            # Convert distance to similarity score (cosine)
            relevance_score = 1 - distance
            
            segments.append({
                "segment_id": segment_id,
                "episode_id": metadata["episode_id"],
                "episode_title": metadata["episode_title"],
                "podcast_title": metadata["podcast_title"],
                "audio_url": metadata["audio_url"],
                "start_ms": metadata["start_ms"],
                "end_ms": metadata["end_ms"],
                "summary": metadata["summary"],
                "topic_tags": metadata.get("topic_tags", "").split(","),
                "density_score": metadata["density_score"],
                "relevance_score": relevance_score,
            })
            
            if len(segments) >= limit:
                break
    
    return segments


async def get_segments_by_topic(
    topic: str,
    limit: int = 20,
    min_density: float = 0.3,
    diversity: bool = True,
) -> List[Dict[str, Any]]:
    """
    Get segments for a specific topic.
    
    Args:
        topic: Topic to search for
        limit: Max results
        min_density: Minimum density score
        diversity: If True, ensure results come from different podcasts
        
    Returns:
        List of segments
    """
    segments = await search_segments(
        query=topic,
        limit=limit * 3 if diversity else limit,
        min_density=min_density,
    )
    
    if diversity:
        # Ensure diversity by limiting segments per podcast
        seen_podcasts = {}
        diverse_segments = []
        
        for seg in segments:
            podcast = seg["podcast_title"]
            if podcast not in seen_podcasts:
                seen_podcasts[podcast] = 0
            
            if seen_podcasts[podcast] < 3:  # Max 3 per podcast
                diverse_segments.append(seg)
                seen_podcasts[podcast] += 1
            
            if len(diverse_segments) >= limit:
                break
        
        return diverse_segments
    
    return segments[:limit]
