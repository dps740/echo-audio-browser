"""Playlist generation endpoints (V4)."""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
import uuid
import logging
from datetime import datetime

from app.models import PlaybackManifest, PlaybackSegment
from app.routers.v4_segments import search_segments_v4
from app.services.clip_extractor import get_clip_url

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/playlists", tags=["playlists"])


def format_time(ms: int) -> str:
    """Format milliseconds as MM:SS."""
    total_seconds = ms // 1000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:02d}"


@router.get("/topic/{topic}", response_model=PlaybackManifest)
async def get_topic_playlist(
    topic: str,
    limit: int = 15,
    diverse: bool = True,
):
    """
    Generate a playlist of segments for a topic using V4 snippet search.
    
    - **topic**: Topic to search for (e.g., "AI", "consciousness")
    - **limit**: Max number of segments (default 15)
    - **diverse**: Ensure segments from different episodes (default true)
    """
    try:
        # Use V4 search (snippet embeddings)
        segments = search_segments_v4(query=topic, top_k=limit * 2 if diverse else limit)
        
        # Apply diversity filter if requested
        if diverse and segments:
            diverse_segments = []
            episode_counts = {}
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
        logger.error(f"V4 search failed for topic '{topic}': {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {type(e).__name__}: {str(e)[:200]}")
    
    if not segments:
        raise HTTPException(status_code=404, detail=f"No segments found for topic: {topic}")
    
    # Build manifest
    playback_segments = []
    total_duration = 0
    
    for seg in segments:
        start_ms = seg.get("start_ms", 0)
        end_ms = seg.get("end_ms", 0)
        duration = end_ms - start_ms
        total_duration += duration
        
        video_id = seg.get("video_id", "")
        episode_title = seg.get("episode_title", f"Episode {video_id}")
        snippet = seg.get("snippet", "")
        
        # Generate clip URL
        clip_url = get_clip_url(video_id, start_ms, end_ms) or f"/audio/{video_id}.mp3"
        
        # Generate pre-roll text
        preroll = f"From {episode_title}: {snippet[:100]}..."
        
        playback_segments.append(PlaybackSegment(
            segment_id=seg.get("id", f"{video_id}_{start_ms}"),
            source_url=clip_url,
            start_time_ms=start_ms,
            end_time_ms=end_ms,
            preroll_text=preroll,
            topic_tags=[],  # V4 doesn't have tags, could extract from snippet
            podcast_title=episode_title,
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
async def search_playlist(
    q: str,
    limit: int = 15,
):
    """
    Generate a playlist from a search query using V4.
    
    - **q**: Search query
    - **limit**: Max segments
    """
    segments = search_segments_v4(query=q, top_k=limit)
    
    if not segments:
        raise HTTPException(status_code=404, detail=f"No segments found for: {q}")
    
    # Build manifest
    playback_segments = []
    total_duration = 0
    
    for seg in segments:
        start_ms = seg.get("start_ms", 0)
        end_ms = seg.get("end_ms", 0)
        duration = end_ms - start_ms
        total_duration += duration
        
        video_id = seg.get("video_id", "")
        episode_title = seg.get("episode_title", f"Episode {video_id}")
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
            podcast_title=episode_title,
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
async def get_daily_mix(
    topics: Optional[str] = None,
    limit: int = 10,
):
    """
    Generate a personalized daily mix using V4.
    
    - **topics**: Comma-separated list of topics (optional)
    - **limit**: Total segments in mix
    """
    if topics:
        topic_list = [t.strip() for t in topics.split(",")]
        segments_per_topic = max(2, limit // len(topic_list))
        
        all_segments = []
        for topic in topic_list:
            topic_segs = search_segments_v4(query=topic, top_k=segments_per_topic)
            all_segments.extend(topic_segs)
    else:
        # Get broad mix
        all_segments = search_segments_v4(query="interesting insights analysis discussion", top_k=limit * 2)
        # Shuffle for variety
        import random
        random.shuffle(all_segments)
        all_segments = all_segments[:limit]
    
    if not all_segments:
        raise HTTPException(status_code=404, detail="No segments available for daily mix")
    
    # Build manifest
    playback_segments = []
    total_duration = 0
    
    for seg in all_segments[:limit]:
        start_ms = seg.get("start_ms", 0)
        end_ms = seg.get("end_ms", 0)
        duration = end_ms - start_ms
        total_duration += duration
        
        video_id = seg.get("video_id", "")
        episode_title = seg.get("episode_title", f"Episode {video_id}")
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
            podcast_title=episode_title,
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
