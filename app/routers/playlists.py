"""Playlist generation endpoints."""

from fastapi import APIRouter, HTTPException
from typing import Optional
import uuid
import logging
from datetime import datetime

from app.models import PlaybackManifest, PlaybackSegment
from app.services import storage
from app.services.clip_extractor import get_clip_url

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/playlists", tags=["playlists"])


@router.get("/topic/{topic}", response_model=PlaybackManifest)
async def get_topic_playlist(
    topic: str,
    limit: int = 15,
    diverse: bool = True,
):
    """
    Generate a playlist of segments for a topic.

    - **topic**: Topic to search for
    - **limit**: Max segments (default 15)
    - **diverse**: Limit segments per episode for variety (default true)
    """
    try:
        segments = storage.search(
            query=topic,
            top_k=limit * 2 if diverse else limit
        )

        if diverse and segments:
            diverse_segments = []
            episode_counts: dict = {}
            for seg in segments:
                ep_id = seg.get("video_id", "")
                if episode_counts.get(ep_id, 0) >= 3:
                    continue
                diverse_segments.append(seg)
                episode_counts[ep_id] = episode_counts.get(ep_id, 0) + 1
                if len(diverse_segments) >= limit:
                    break
            segments = diverse_segments
        else:
            segments = segments[:limit]

    except Exception as e:
        logger.error(f"Search failed for topic '{topic}': {e}")
        raise HTTPException(500, f"Search error: {type(e).__name__}: {str(e)[:200]}")

    if not segments:
        raise HTTPException(404, f"No segments found for topic: {topic}")

    playback_segments = []
    total_duration = 0

    for seg in segments:
        start_ms = seg.get("start_ms", 0)
        end_ms = seg.get("end_ms", 0)
        total_duration += end_ms - start_ms

        video_id = seg.get("video_id", "")
        episode_title = seg.get("episode_title", f"Episode {video_id}")
        podcast_name = seg.get("podcast_name") or episode_title
        snippet = seg.get("snippet", "")

        clip_url = get_clip_url(video_id, start_ms, end_ms) or f"/audio/{video_id}.mp3"
        preroll = f"From {episode_title}: {snippet[:100]}..."

        playback_segments.append(PlaybackSegment(
            segment_id=seg.get("id", f"{video_id}_{start_ms}"),
            source_url=clip_url,
            start_time_ms=start_ms,
            end_time_ms=end_ms,
            preroll_text=preroll,
            topic_tags=[],
            podcast_title=podcast_name,
            episode_title=episode_title,
        ))

    return PlaybackManifest(
        playlist_id=str(uuid.uuid4()),
        title=f"Deep Dive: {topic}",
        description=f"A curated collection of {len(playback_segments)} segments about {topic}.",
        segments=playback_segments,
        total_duration_ms=total_duration,
        created_at=datetime.utcnow(),
    )


@router.get("/search", response_model=PlaybackManifest)
async def search_playlist(q: str, limit: int = 15):
    """Generate a playlist from a search query."""
    segments = storage.search(query=q, top_k=limit)

    if not segments:
        raise HTTPException(404, f"No segments found for: {q}")

    playback_segments = []
    total_duration = 0

    for seg in segments:
        start_ms = seg.get("start_ms", 0)
        end_ms = seg.get("end_ms", 0)
        total_duration += end_ms - start_ms

        video_id = seg.get("video_id", "")
        episode_title = seg.get("episode_title", f"Episode {video_id}")
        podcast_name = seg.get("podcast_name") or episode_title
        snippet = seg.get("snippet", "")

        clip_url = get_clip_url(video_id, start_ms, end_ms) or f"/audio/{video_id}.mp3"
        preroll = f"From {episode_title}: {snippet[:100]}..."

        playback_segments.append(PlaybackSegment(
            segment_id=seg.get("id", f"{video_id}_{start_ms}"),
            source_url=clip_url,
            start_time_ms=start_ms,
            end_time_ms=end_ms,
            preroll_text=preroll,
            topic_tags=[],
            podcast_title=podcast_name,
            episode_title=episode_title,
        ))

    return PlaybackManifest(
        playlist_id=str(uuid.uuid4()),
        title=f"Search: {q}",
        description=f"Results for '{q}' - {len(playback_segments)} segments",
        segments=playback_segments,
        total_duration_ms=total_duration,
        created_at=datetime.utcnow(),
    )


@router.get("/daily-mix", response_model=PlaybackManifest)
async def get_daily_mix(topics: Optional[str] = None, limit: int = 10):
    """Generate a daily mix — optionally scoped to comma-separated topics."""
    if topics:
        topic_list = [t.strip() for t in topics.split(",")]
        per_topic = max(2, limit // len(topic_list))
        all_segments = []
        for topic in topic_list:
            all_segments.extend(storage.search(query=topic, top_k=per_topic))
    else:
        all_segments = storage.search(
            query="interesting insights analysis discussion", top_k=limit * 2
        )
        import random
        random.shuffle(all_segments)
        all_segments = all_segments[:limit]

    if not all_segments:
        raise HTTPException(404, "No segments available for daily mix")

    playback_segments = []
    total_duration = 0

    for seg in all_segments[:limit]:
        start_ms = seg.get("start_ms", 0)
        end_ms = seg.get("end_ms", 0)
        total_duration += end_ms - start_ms

        video_id = seg.get("video_id", "")
        episode_title = seg.get("episode_title", f"Episode {video_id}")
        podcast_name = seg.get("podcast_name") or episode_title
        snippet = seg.get("snippet", "")

        clip_url = get_clip_url(video_id, start_ms, end_ms) or f"/audio/{video_id}.mp3"
        preroll = f"From {episode_title}: {snippet[:100]}..."

        playback_segments.append(PlaybackSegment(
            segment_id=seg.get("id", f"{video_id}_{start_ms}"),
            source_url=clip_url,
            start_time_ms=start_ms,
            end_time_ms=end_ms,
            preroll_text=preroll,
            topic_tags=[],
            podcast_title=podcast_name,
            episode_title=episode_title,
        ))

    return PlaybackManifest(
        playlist_id=str(uuid.uuid4()),
        title="Your Daily Mix",
        description=f"Today's personalized mix - {len(playback_segments)} segments",
        segments=playback_segments,
        total_duration_ms=total_duration,
        created_at=datetime.utcnow(),
    )
