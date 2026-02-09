"""Indexing endpoint — run the segmentation pipeline on a VTT file."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path

from app.services.segmentation import index_episode
from app.services import storage

router = APIRouter(tags=["indexing"])


class IndexRequest(BaseModel):
    episode_title: Optional[str] = None
    podcast_name: Optional[str] = None


class SegmentInfo(BaseModel):
    start_ms: int
    end_ms: int
    duration_s: float
    snippet: str


class IndexResponse(BaseModel):
    video_id: str
    episode_title: str
    podcast_name: str
    total_segments: int
    segments: List[SegmentInfo]


@router.post("/index/{video_id}", response_model=IndexResponse)
async def index_video(video_id: str, request: IndexRequest = None):
    """
    Index a VTT transcript: parse → segment → generate snippets → store.

    Expects audio/{video_id}.en.vtt to exist on disk.
    Optionally accepts episode_title and podcast_name in the request body.
    """
    if request is None:
        request = IndexRequest()

    # Find VTT file
    vtt_path = Path(f"audio/{video_id}.en.vtt")
    if not vtt_path.exists():
        raise HTTPException(404, f"VTT file not found: {vtt_path}")

    vtt_content = vtt_path.read_text()

    episode_title = request.episode_title or f"Episode {video_id}"
    podcast_name = request.podcast_name or "Unknown Podcast"

    # Run pipeline
    segments = index_episode(vtt_content, episode_title)

    if not segments:
        raise HTTPException(400, "No segments produced — VTT may be empty or unparseable")

    # Store in ChromaDB
    storage.save_episode(video_id, episode_title, podcast_name, segments)

    return IndexResponse(
        video_id=video_id,
        episode_title=episode_title,
        podcast_name=podcast_name,
        total_segments=len(segments),
        segments=[
            SegmentInfo(
                start_ms=s.start_ms,
                end_ms=s.end_ms,
                duration_s=round((s.end_ms - s.start_ms) / 1000, 1),
                snippet=s.snippet
            )
            for s in segments
        ]
    )
