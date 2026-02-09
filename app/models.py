"""Data models for Echo — playback manifest types."""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel


class PlaybackSegment(BaseModel):
    """A segment in a playback manifest."""
    segment_id: str
    source_url: str
    start_time_ms: int
    end_time_ms: int
    preroll_text: str
    topic_tags: List[str]
    podcast_title: str
    episode_title: str


class PlaybackManifest(BaseModel):
    """A playlist of segments for client playback."""
    playlist_id: str
    title: str
    description: Optional[str] = None
    segments: List[PlaybackSegment]
    total_duration_ms: int
    created_at: datetime
