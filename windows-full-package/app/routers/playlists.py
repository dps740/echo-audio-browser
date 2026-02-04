"""Playlist generation endpoints."""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
import uuid
from datetime import datetime

from app.models import PlaybackManifest, PlaybackSegment
from app.services.vectordb import get_segments_by_topic, search_segments


router = APIRouter(prefix="/playlists", tags=["playlists"])


@router.get("/topic/{topic}", response_model=PlaybackManifest)
async def get_topic_playlist(
    topic: str,
    limit: int = 10,
    min_density: float = 0.3,
    diverse: bool = True,
):
    """
    Generate a playlist of segments for a topic.
    
    - **topic**: Topic to search for (e.g., "AI Safety", "Stoicism")
    - **limit**: Max number of segments (default 10)
    - **min_density**: Minimum density score (default 0.3)
    - **diverse**: Ensure segments from different podcasts (default true)
    """
    segments = await get_segments_by_topic(
        topic=topic,
        limit=limit,
        min_density=min_density,
        diversity=diverse,
    )
    
    if not segments:
        raise HTTPException(status_code=404, detail=f"No segments found for topic: {topic}")
    
    # Build manifest
    playback_segments = []
    total_duration = 0
    
    for seg in segments:
        duration = seg["end_ms"] - seg["start_ms"]
        total_duration += duration
        
        # Generate pre-roll text
        preroll = f"From {seg['podcast_title']}: {seg['summary'][:100]}..."
        
        playback_segments.append(PlaybackSegment(
            segment_id=seg["segment_id"],
            source_url=seg["audio_url"],
            start_time_ms=seg["start_ms"],
            end_time_ms=seg["end_ms"],
            preroll_text=preroll,
            topic_tags=seg["topic_tags"],
            podcast_title=seg["podcast_title"],
            episode_title=seg["episode_title"],
        ))
    
    return PlaybackManifest(
        playlist_id=str(uuid.uuid4()),
        title=f"Deep Dive: {topic}",
        description=f"A curated collection of {len(playback_segments)} segments about {topic} from various podcasts.",
        segments=playback_segments,
        total_duration_ms=total_duration,
        created_at=datetime.utcnow(),
    )


@router.get("/search", response_model=PlaybackManifest)
async def search_playlist(
    q: str,
    limit: int = 10,
    min_density: float = 0.0,
):
    """
    Generate a playlist from a search query.
    
    - **q**: Search query
    - **limit**: Max segments
    - **min_density**: Minimum density score
    """
    segments = await search_segments(
        query=q,
        limit=limit,
        min_density=min_density,
    )
    
    if not segments:
        raise HTTPException(status_code=404, detail=f"No segments found for: {q}")
    
    # Build manifest
    playback_segments = []
    total_duration = 0
    
    for seg in segments:
        duration = seg["end_ms"] - seg["start_ms"]
        total_duration += duration
        
        preroll = f"From {seg['podcast_title']}: {seg['summary'][:100]}..."
        
        playback_segments.append(PlaybackSegment(
            segment_id=seg["segment_id"],
            source_url=seg["audio_url"],
            start_time_ms=seg["start_ms"],
            end_time_ms=seg["end_ms"],
            preroll_text=preroll,
            topic_tags=seg["topic_tags"],
            podcast_title=seg["podcast_title"],
            episode_title=seg["episode_title"],
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
    Generate a personalized daily mix.
    
    - **topics**: Comma-separated list of topics (optional)
    - **limit**: Total segments in mix
    
    If no topics provided, returns a mix of high-density segments.
    """
    if topics:
        topic_list = [t.strip() for t in topics.split(",")]
        segments_per_topic = max(1, limit // len(topic_list))
        
        all_segments = []
        for topic in topic_list:
            topic_segs = await get_segments_by_topic(
                topic=topic,
                limit=segments_per_topic,
                min_density=0.5,
                diversity=True,
            )
            all_segments.extend(topic_segs)
    else:
        # Get high-density segments across all topics
        all_segments = await search_segments(
            query="interesting insights analysis",
            limit=limit * 2,
            min_density=0.6,
        )
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
        duration = seg["end_ms"] - seg["start_ms"]
        total_duration += duration
        
        preroll = f"From {seg['podcast_title']}: {seg['summary'][:100]}..."
        
        playback_segments.append(PlaybackSegment(
            segment_id=seg["segment_id"],
            source_url=seg["audio_url"],
            start_time_ms=seg["start_ms"],
            end_time_ms=seg["end_ms"],
            preroll_text=preroll,
            topic_tags=seg["topic_tags"],
            podcast_title=seg["podcast_title"],
            episode_title=seg["episode_title"],
        ))
    
    return PlaybackManifest(
        playlist_id=str(uuid.uuid4()),
        title="Your Daily Mix",
        description=f"Today's personalized mix - {len(playback_segments)} segments",
        segments=playback_segments,
        total_duration_ms=total_duration,
        created_at=datetime.utcnow(),
    )
