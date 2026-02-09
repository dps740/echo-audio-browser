"""Library browsing endpoints - browse ingested content (V4)."""

from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict
import chromadb
from collections import defaultdict

router = APIRouter(prefix="/library", tags=["library"])


def _get_v4_collection():
    """Get V4 segments collection."""
    client = chromadb.PersistentClient(path="./chroma_data")
    return client.get_or_create_collection("v4_segments")


@router.get("/overview")
async def get_overview():
    """Get library overview - podcasts, episode counts, segment counts."""
    collection = _get_v4_collection()
    total = collection.count()
    
    if total == 0:
        return {
            "total_segments": 0,
            "total_episodes": 0,
            "total_podcasts": 0,
            "podcasts": [],
        }
    
    # Fetch all metadata
    results = collection.get(include=["metadatas"])
    
    podcasts = defaultdict(lambda: {"episodes": set(), "segments": 0, "total_duration_ms": 0})
    
    for meta in results["metadatas"]:
        # V4 uses episode_title as podcast proxy for now
        podcast = meta.get("episode_title", "Unknown Podcast")
        episode = meta.get("video_id", "unknown")
        start_ms = meta.get("start_ms", 0)
        end_ms = meta.get("end_ms", 0)
        duration = end_ms - start_ms
        
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
    collection = _get_v4_collection()
    
    # V4 uses episode_title as podcast identifier
    results = collection.get(
        where={"episode_title": podcast_name},
        include=["metadatas"],
    )
    
    if not results["metadatas"]:
        raise HTTPException(status_code=404, detail=f"No episodes found for: {podcast_name}")
    
    episodes = defaultdict(lambda: {
        "title": "",
        "segments": 0,
        "total_duration_ms": 0,
    })
    
    for meta in results["metadatas"]:
        ep_id = meta.get("video_id", "unknown")
        episodes[ep_id]["title"] = meta.get("episode_title", "Unknown")
        episodes[ep_id]["segments"] += 1
        start_ms = meta.get("start_ms", 0)
        end_ms = meta.get("end_ms", 0)
        episodes[ep_id]["total_duration_ms"] += end_ms - start_ms
    
    episode_list = []
    for ep_id, data in sorted(episodes.items(), key=lambda x: -x[1]["total_duration_ms"]):
        episode_list.append({
            "episode_id": ep_id,
            "title": data["title"],
            "segment_count": data["segments"],
            "total_duration_ms": data["total_duration_ms"],
        })
    
    return {
        "podcast": podcast_name,
        "episode_count": len(episode_list),
        "episodes": episode_list,
    }


@router.get("/episodes/{episode_id}/segments")
async def get_episode_segments(episode_id: str):
    """Get all segments for a specific episode, ordered by time."""
    collection = _get_v4_collection()
    
    results = collection.get(
        where={"video_id": episode_id},
        include=["metadatas", "documents"],
    )
    
    if not results["metadatas"]:
        raise HTTPException(status_code=404, detail=f"No segments found for episode: {episode_id}")
    
    segments = []
    for i, meta in enumerate(results["metadatas"]):
        segments.append({
            "segment_id": results["ids"][i],
            "video_id": meta.get("video_id"),
            "episode_title": meta.get("episode_title", "Unknown"),
            "start_ms": meta.get("start_ms", 0),
            "end_ms": meta.get("end_ms", 0),
            "duration_s": meta.get("duration_s", 0),
            "snippet": results["documents"][i] if results["documents"] else "",
        })
    
    # Sort by start time
    segments.sort(key=lambda s: s["start_ms"])
    
    return {
        "episode_id": episode_id,
        "episode_title": segments[0]["episode_title"] if segments else "Unknown",
        "segment_count": len(segments),
        "segments": segments,
    }


@router.delete("/episodes/{episode_id}")
async def delete_episode(episode_id: str):
    """Delete all segments for an episode."""
    collection = _get_v4_collection()
    
    results = collection.get(
        where={"video_id": episode_id},
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


@router.get("/stats")
async def get_stats():
    """Get V4 collection statistics."""
    collection = _get_v4_collection()
    
    results = collection.get(include=["metadatas"])
    
    episodes = set()
    total_duration_ms = 0
    
    for meta in results["metadatas"]:
        episodes.add(meta.get("video_id", "unknown"))
        start_ms = meta.get("start_ms", 0)
        end_ms = meta.get("end_ms", 0)
        total_duration_ms += end_ms - start_ms
    
    return {
        "total_segments": collection.count(),
        "total_episodes": len(episodes),
        "episodes": list(episodes),
        "total_duration_hours": round(total_duration_ms / 1000 / 60 / 60, 1),
    }
