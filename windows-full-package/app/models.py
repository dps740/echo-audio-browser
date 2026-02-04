"""Data models for Echo."""

from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, HttpUrl
from enum import Enum


# Enums
class TranscriptStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETE = "complete"
    FAILED = "failed"


# Database Models (SQLAlchemy style, but using Pydantic for now)
class PodcastFeed(BaseModel):
    """A subscribed podcast RSS feed."""
    id: str
    title: str
    rss_url: HttpUrl
    description: Optional[str] = None
    author: Optional[str] = None
    image_url: Optional[HttpUrl] = None
    created_at: datetime
    last_checked: Optional[datetime] = None


class Episode(BaseModel):
    """A podcast episode."""
    id: str
    feed_id: str
    title: str
    audio_url: HttpUrl
    description: Optional[str] = None
    published_at: Optional[datetime] = None
    duration_sec: Optional[int] = None
    transcript_status: TranscriptStatus = TranscriptStatus.PENDING
    created_at: datetime


class Segment(BaseModel):
    """An atomic segment within an episode."""
    id: str
    episode_id: str
    start_ms: int
    end_ms: int
    transcript_text: str
    summary: str
    topic_tags: List[str]
    density_score: float  # 0-1, higher = more information-dense
    speaker_list: List[str] = []
    created_at: datetime


# API Request/Response Models
class FeedCreate(BaseModel):
    """Request to add a new podcast feed."""
    rss_url: HttpUrl


class FeedResponse(BaseModel):
    """Response after adding a feed."""
    id: str
    title: str
    rss_url: HttpUrl
    episode_count: int


class SegmentSearchResult(BaseModel):
    """A segment returned from search."""
    segment_id: str
    episode_id: str
    episode_title: str
    podcast_title: str
    audio_url: HttpUrl
    start_ms: int
    end_ms: int
    summary: str
    topic_tags: List[str]
    density_score: float
    relevance_score: float  # Search relevance


class PlaybackSegment(BaseModel):
    """A segment in a playback manifest."""
    segment_id: str
    source_url: HttpUrl
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


class IngestRequest(BaseModel):
    """Request to ingest (transcribe + segment) an episode."""
    episode_id: str
    force: bool = False  # Re-process even if already done


class IngestResponse(BaseModel):
    """Response after ingestion."""
    episode_id: str
    status: TranscriptStatus
    segment_count: int
    message: str
