"""Library browsing endpoints - browse ingested content."""

from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict
import chromadb
from collections import defaultdict

from app.config import get_embedding_function

router = APIRouter(prefix="/library", tags=["library"])


def _get_collection():
    """Get ChromaDB collection."""
    client = chromadb.PersistentClient(path="./chroma_data")
    embedding_fn = get_embedding_function()
    return client.get_or_create_collection("segments", embedding_function=embedding_fn)


@router.get("/overview")
async def get_overview():
    """Get library overview - podcasts, episode counts, segment counts."""
    collection = _get_collection()
    total = collection.count()
    
    if total == 0:
        return {
            "total_segments": 0,
            "total_episodes": 0,
            "total_podcasts": 0,
            "podcasts": [],
        }
    
    # Fetch all metadata (ChromaDB doesn't support GROUP BY)
    results = collection.get(include=["metadatas"])
    
    podcasts = defaultdict(lambda: {"episodes": set(), "segments": 0, "total_duration_ms": 0})
    
    for meta in results["metadatas"]:
        podcast = meta.get("podcast_title", "Unknown")
        episode = meta.get("episode_id", "unknown")
        duration = meta.get("end_ms", 0) - meta.get("start_ms", 0)
        
        podcasts[podcast]["episodes"].add(episode)
        podcasts[podcast]["segments"] += 1
        podcasts[podcast]["total_duration_ms"] += duration
    
    podcast_list = []
    for name, data in sorted(podcasts.items(), key=lambda x: -x[1]["segments"]):
        podcast_list.append({
            "name": name,
            "episode_count": len(data["episodes"]),
            "segment_count": data["segments"],
            "total_duration_ms": data["total_duration_ms"],
        })
    
    return {
        "total_segments": total,
        "total_episodes": sum(p["episode_count"] for p in podcast_list),
        "total_podcasts": len(podcast_list),
        "podcasts": podcast_list,
    }


@router.get("/podcasts/{podcast_name}/episodes")
async def get_podcast_episodes(podcast_name: str):
    """Get episodes for a specific podcast."""
    collection = _get_collection()
    
    # ChromaDB where filter
    results = collection.get(
        where={"podcast_title": podcast_name},
        include=["metadatas"],
    )
    
    if not results["metadatas"]:
        raise HTTPException(status_code=404, detail=f"No episodes found for: {podcast_name}")
    
    episodes = defaultdict(lambda: {
        "title": "",
        "segments": 0,
        "total_duration_ms": 0,
        "source": "",
    })
    
    for meta in results["metadatas"]:
        ep_id = meta.get("episode_id", "unknown")
        episodes[ep_id]["title"] = meta.get("episode_title", "Unknown")
        episodes[ep_id]["segments"] += 1
        episodes[ep_id]["total_duration_ms"] += meta.get("end_ms", 0) - meta.get("start_ms", 0)
        episodes[ep_id]["source"] = meta.get("source", "unknown")
    
    episode_list = []
    for ep_id, data in sorted(episodes.items(), key=lambda x: -x[1]["total_duration_ms"]):
        episode_list.append({
            "episode_id": ep_id,
            "title": data["title"],
            "segment_count": data["segments"],
            "total_duration_ms": data["total_duration_ms"],
            "source": data["source"],
        })
    
    return {
        "podcast": podcast_name,
        "episode_count": len(episode_list),
        "episodes": episode_list,
    }


@router.get("/episodes/{episode_id}/segments")
async def get_episode_segments(episode_id: str):
    """Get all segments for a specific episode, ordered by time."""
    collection = _get_collection()
    
    results = collection.get(
        where={"episode_id": episode_id},
        include=["metadatas", "documents"],
    )
    
    if not results["metadatas"]:
        raise HTTPException(status_code=404, detail=f"No segments found for episode: {episode_id}")
    
    segments = []
    for i, meta in enumerate(results["metadatas"]):
        segments.append({
            "segment_id": results["ids"][i],
            "episode_id": meta.get("episode_id"),
            "episode_title": meta.get("episode_title", "Unknown"),
            "podcast_title": meta.get("podcast_title", "Unknown"),
            "audio_url": meta.get("audio_url", ""),
            "start_ms": meta.get("start_ms", 0),
            "end_ms": meta.get("end_ms", 0),
            "text": results["documents"][i] if results["documents"] else "",
            "summary": meta.get("summary", ""),
        })
    
    # Sort by start time
    segments.sort(key=lambda s: s["start_ms"])
    
    return {
        "episode_id": episode_id,
        "episode_title": segments[0]["episode_title"] if segments else "Unknown",
        "podcast_title": segments[0]["podcast_title"] if segments else "Unknown",
        "segment_count": len(segments),
        "segments": segments,
    }


@router.delete("/episodes/{episode_id}")
async def delete_episode(episode_id: str):
    """Delete all segments for an episode."""
    collection = _get_collection()
    
    results = collection.get(
        where={"episode_id": episode_id},
        include=["metadatas"],
    )
    
    if not results["ids"]:
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_id}")
    
    collection.delete(ids=results["ids"])
    
    return {
        "status": "deleted",
        "episode_id": episode_id,
        "segments_removed": len(results["ids"]),
    }
