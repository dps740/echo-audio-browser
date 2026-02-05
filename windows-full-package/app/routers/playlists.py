"""Playlist generation endpoints."""

from fastapi import APIRouter, HTTPException
from typing import List, Optional
import uuid
import logging
from datetime import datetime

from app.models import PlaybackManifest, PlaybackSegment
from app.services.hybrid_search import hybrid_search
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/playlists", tags=["playlists"])


@router.get("/topic/{topic}", response_model=PlaybackManifest)
async def get_topic_playlist(
    topic: str,
    limit: int = 10,
    min_density: float = 0.0,
    diverse: bool = True,
):
    """
    Generate a playlist of segments for a topic.
    
    - **topic**: Topic to search for (e.g., "AI Safety", "Stoicism")
    - **limit**: Max number of segments (default 10)
    - **min_density**: Minimum density score (default 0.0)
    - **diverse**: Ensure segments from different podcasts (default true)
    """
    try:
        # Use hybrid search instead of pure vector search
        segments = hybrid_search(
            query=topic,
            limit=limit,
            chroma_path=settings.chroma_persist_dir,
            min_density=min_density,
            diversity=diverse,
        )
    except Exception as e:
        logger.error(f"Search failed for topic '{topic}': {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"Search error: {type(e).__name__}: {str(e)[:200]}")
    
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
    # Use hybrid search for better results
    segments = hybrid_search(
        query=q,
        limit=limit,
        chroma_path=settings.chroma_persist_dir,
        min_density=min_density,
        diversity=False,  # Search doesn't enforce diversity by default
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
            topic_segs = hybrid_search(
                query=topic,
                limit=segments_per_topic,
                chroma_path=settings.chroma_persist_dir,
                min_density=0.5,
                diversity=True,
            )
            all_segments.extend(topic_segs)
    else:
        # Get high-density segments across all topics
        all_segments = hybrid_search(
            query="interesting insights analysis",
            limit=limit * 2,
            chroma_path=settings.chroma_persist_dir,
            min_density=0.6,
            diversity=True,
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
