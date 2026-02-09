"""Library browsing endpoints — browse ingested content."""

from fastapi import APIRouter, HTTPException
from collections import defaultdict

from app.services import storage
from app.services.clip_extractor import get_clip_url

router = APIRouter(prefix="/library", tags=["library"])


@router.get("/overview")
async def get_overview():
    """Get library overview — podcasts, episode counts, segment counts."""
    total = storage.get_total_count()

    if total == 0:
        return {
            "total_segments": 0,
            "total_episodes": 0,
            "total_podcasts": 0,
            "podcasts": [],
        }

    all_meta = storage.get_all_metadata()

    podcasts = defaultdict(lambda: {
        "episodes": set(), "segments": 0, "total_duration_ms": 0
    })

    for meta in all_meta:
        podcast = meta.get("podcast_name") or meta.get("episode_title", "Unknown")
        episode = meta.get("video_id", "unknown")
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
    all_meta = storage.get_all_metadata()

    episodes = defaultdict(lambda: {
        "title": "", "segments": 0, "total_duration_ms": 0
    })

    for meta in all_meta:
        podcast = meta.get("podcast_name") or meta.get("episode_title", "Unknown")
        if podcast != podcast_name:
            continue

        ep_id = meta.get("video_id", "unknown")
        episodes[ep_id]["title"] = meta.get("episode_title", "Unknown")
        episodes[ep_id]["segments"] += 1
        episodes[ep_id]["total_duration_ms"] += (
            meta.get("end_ms", 0) - meta.get("start_ms", 0)
        )

    if not episodes:
        raise HTTPException(404, f"No episodes found for: {podcast_name}")

    episode_list = []
    for ep_id, data in sorted(
        episodes.items(), key=lambda x: -x[1]["total_duration_ms"]
    ):
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
    """Get all segments for an episode, ordered by time."""
    segments = storage.get_episode_segments(episode_id)

    if not segments:
        raise HTTPException(404, f"No segments found for episode: {episode_id}")

    # Add clip URLs so frontend can play segments
    for seg in segments:
        clip_url = get_clip_url(
            seg.get("video_id", episode_id),
            seg.get("start_ms", 0),
            seg.get("end_ms", 0),
        )
        seg["clip_url"] = clip_url or f"/audio/{episode_id}.mp3"

    return {
        "episode_id": episode_id,
        "episode_title": segments[0].get("episode_title", "Unknown"),
        "podcast_title": segments[0].get("podcast_name")
                         or segments[0].get("episode_title", "Unknown"),
        "segment_count": len(segments),
        "segments": segments,
    }


@router.delete("/episodes/{episode_id}")
async def delete_episode(episode_id: str):
    """Delete all segments for an episode."""
    count = storage.delete_episode(episode_id)
    if count == 0:
        raise HTTPException(404, f"Episode not found: {episode_id}")

    return {
        "status": "deleted",
        "episode_id": episode_id,
        "segments_removed": count,
    }


@router.get("/stats")
async def get_stats():
    """Get collection statistics."""
    all_meta = storage.get_all_metadata()

    episodes = set()
    total_duration_ms = 0

    for meta in all_meta:
        episodes.add(meta.get("video_id", "unknown"))
        total_duration_ms += meta.get("end_ms", 0) - meta.get("start_ms", 0)

    return {
        "total_segments": storage.get_total_count(),
        "total_episodes": len(episodes),
        "episodes": list(episodes),
        "total_duration_hours": round(total_duration_ms / 3_600_000, 1),
    }
