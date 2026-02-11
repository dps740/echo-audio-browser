"""Search endpoint — semantic search across all indexed episodes."""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional

from app.services import storage
from app.services.clip_extractor import get_clip_url

router = APIRouter(tags=["search"])


def _format_time(ms: int) -> str:
    m, s = divmod(ms // 1000, 60)
    return f"{m}:{s:02d}"


class SearchResult(BaseModel):
    query: str
    results: List[dict]
    total_results: int


@router.get("/search", response_model=SearchResult)
async def search_all(q: str, top_k: int = 10, video_id: Optional[str] = None):
    """
    Search across all episodes (or a specific episode) by topic.

    This is the primary search endpoint:
    "Find segments about X across all my podcasts"

    Examples:
      GET /search?q=Chinese AI infrastructure
      GET /search?q=consciousness&video_id=abc123
    """
    results = storage.search(query=q, top_k=top_k, video_id=video_id)

    for r in results:
        r["clip_url"] = get_clip_url(r["video_id"], r["start_ms"], r["end_ms"])
        r["start_formatted"] = _format_time(r["start_ms"])
        r["end_formatted"] = _format_time(r["end_ms"])

    return SearchResult(
        query=q,
        results=results,
        total_results=len(results)
    )


@router.get("/clip/{video_id}")
async def get_clip(video_id: str, start_ms: int, end_ms: int):
    """Generate a clip and redirect to it for direct audio playback."""
    from fastapi.responses import RedirectResponse
    clip_url = get_clip_url(video_id, start_ms, end_ms)

    if clip_url:
        # Redirect to the actual audio file for direct playback
        return RedirectResponse(url=clip_url, status_code=302)
    else:
        # Fallback to full MP3 with time range (less accurate but works)
        from fastapi import HTTPException
        raise HTTPException(500, f"Failed to generate clip for {video_id}")
