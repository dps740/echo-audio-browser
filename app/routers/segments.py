"""Segment search and ingestion endpoints."""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import List, Optional

from app.models import (
    SegmentSearchResult, IngestRequest, IngestResponse, 
    TranscriptStatus, Segment
)
from app.services.transcription import transcribe_audio
from app.services.segmentation import segment_transcript, compare_models, ModelComparisonResult
from app.services.vectordb import store_segments, search_segments
from app.services.hybrid_search import hybrid_search
from app.routers.feeds import get_feeds_storage

import uuid
from datetime import datetime


router = APIRouter(prefix="/segments", tags=["segments"])

# Track ingestion status
_ingestion_status = {}


@router.get("/search", response_model=List[SegmentSearchResult])
async def search(
    q: str,
    limit: int = 20,
    min_density: float = 0.0,
    min_relevance: float = 0.3,
    topic: Optional[str] = None,
):
    """
    Search for segments by topic or query.
    
    - **q**: Search query
    - **limit**: Max results (default 20)
    - **min_density**: Minimum density score filter (0-1)
    - **min_relevance**: Minimum relevance score (0-1). Results below this threshold are discarded.
                        Default 0.3 filters out poor semantic matches.
    - **topic**: Filter by specific topic tag
    """
    results = hybrid_search(query=q, limit=limit, min_density=min_density, min_relevance=min_relevance) if True else await search_segments(
        query=q,
        limit=limit,
        min_density=min_density,
        topic_filter=topic,
    )
    
    return [
        SegmentSearchResult(
            segment_id=r["segment_id"],
            episode_id=r["episode_id"],
            episode_title=r["episode_title"],
            podcast_title=r["podcast_title"],
            audio_url=r["audio_url"],
            start_ms=r["start_ms"],
            end_ms=r["end_ms"],
            summary=r["summary"],
            topic_tags=r["topic_tags"],
            density_score=r["density_score"],
            relevance_score=r["relevance_score"],
        )
        for r in results
    ]


@router.post("/ingest", response_model=IngestResponse)
async def ingest_episode(
    request: IngestRequest,
    background_tasks: BackgroundTasks,
):
    """
    Ingest an episode: transcribe and segment it.
    
    This operation runs in the background. Check status with GET /segments/ingest/{episode_id}
    """
    _, episodes = get_feeds_storage()
    
    if request.episode_id not in episodes:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    episode = episodes[request.episode_id]
    
    # Check if already processing
    if request.episode_id in _ingestion_status:
        status = _ingestion_status[request.episode_id]
        if status["status"] == TranscriptStatus.PROCESSING and not request.force:
            return IngestResponse(
                episode_id=request.episode_id,
                status=TranscriptStatus.PROCESSING,
                segment_count=0,
                message="Already processing",
            )
    
    # Start background ingestion
    _ingestion_status[request.episode_id] = {
        "status": TranscriptStatus.PROCESSING,
        "segment_count": 0,
        "message": "Starting ingestion",
    }
    
    background_tasks.add_task(
        _ingest_episode_task,
        request.episode_id,
        episode,
    )
    
    return IngestResponse(
        episode_id=request.episode_id,
        status=TranscriptStatus.PROCESSING,
        segment_count=0,
        message="Ingestion started",
    )


@router.get("/ingest/{episode_id}", response_model=IngestResponse)
async def get_ingest_status(episode_id: str):
    """Get the status of an episode ingestion."""
    if episode_id not in _ingestion_status:
        _, episodes = get_feeds_storage()
        if episode_id in episodes:
            return IngestResponse(
                episode_id=episode_id,
                status=TranscriptStatus.PENDING,
                segment_count=0,
                message="Not yet ingested",
            )
        raise HTTPException(status_code=404, detail="Episode not found")
    
    status = _ingestion_status[episode_id]
    return IngestResponse(
        episode_id=episode_id,
        status=status["status"],
        segment_count=status["segment_count"],
        message=status["message"],
    )


async def _ingest_episode_task(episode_id: str, episode):
    """Background task to ingest an episode."""
    feeds, _ = get_feeds_storage()
    
    try:
        # Get podcast info
        podcast = feeds.get(episode.feed_id)
        podcast_title = podcast.title if podcast else "Unknown Podcast"
        
        # Step 1: Transcribe
        _ingestion_status[episode_id]["message"] = "Transcribing..."
        transcript = await transcribe_audio(str(episode.audio_url))
        
        # Step 2: Segment
        _ingestion_status[episode_id]["message"] = "Segmenting..."
        segments = await segment_transcript(transcript, episode.title)
        
        # Step 3: Store in vector DB
        _ingestion_status[episode_id]["message"] = "Storing segments..."
        count = await store_segments(
            segments=segments,
            episode_id=episode_id,
            episode_title=episode.title,
            podcast_title=podcast_title,
            audio_url=str(episode.audio_url),
        )
        
        # Done
        _ingestion_status[episode_id] = {
            "status": TranscriptStatus.COMPLETE,
            "segment_count": count,
            "message": f"Ingested {count} segments",
        }
        
    except Exception as e:
        _ingestion_status[episode_id] = {
            "status": TranscriptStatus.FAILED,
            "segment_count": 0,
            "message": f"Error: {str(e)}",
        }


@router.post("/compare-models")
async def compare_segmentation_models(
    episode_id: str,
    model_a: str = "gpt-4o",
    model_b: str = "gpt-4o-mini",
):
    """
    A/B test two models on the same episode to compare output quality.
    
    Returns segments from both models with cost estimates so you can
    decide if the better model is worth the extra cost.
    
    - **episode_id**: Episode to analyze
    - **model_a**: First model (default: gpt-4o)
    - **model_b**: Second model (default: gpt-4o-mini)
    """
    _, episodes = get_feeds_storage()
    
    if episode_id not in episodes:
        raise HTTPException(status_code=404, detail="Episode not found")
    
    episode = episodes[episode_id]
    
    # Need transcript first - check if we have it
    if episode_id not in _ingestion_status or _ingestion_status[episode_id]["status"] != TranscriptStatus.COMPLETE:
        # Need to transcribe first
        transcript = await transcribe_audio(str(episode.audio_url))
    else:
        # Re-transcribe for comparison (could cache this later)
        transcript = await transcribe_audio(str(episode.audio_url))
    
    # Run comparison
    comparison = await compare_models(
        transcript=transcript,
        model_a=model_a,
        model_b=model_b,
    )
    
    def metrics_to_dict(m):
        if not m:
            return None
        return {
            "segment_count": m.segment_count,
            "avg_duration_min": m.avg_duration_min,
            "duration_std_min": m.duration_std_min,
            "coverage_pct": m.coverage_pct,
            "avg_summary_length": m.avg_summary_length,
            "avg_tags_per_segment": m.avg_tags_per_segment,
            "unique_tags": m.unique_tags,
            "density_mean": m.density_mean,
            "density_std": m.density_std,
            "specificity_score": m.specificity_score,
        }
    
    return {
        "episode_id": episode_id,
        "episode_title": episode.title,
        "model_a": {
            "name": comparison.model_a,
            "cost_usd": round(comparison.cost_a, 4),
            "segment_count": len(comparison.segments_a),
            "metrics": metrics_to_dict(comparison.metrics_a),
            "segments": [
                {
                    "start_ms": s.start_ms,
                    "end_ms": s.end_ms,
                    "summary": s.summary,
                    "topic_tags": s.topic_tags,
                    "density_score": s.density_score,
                }
                for s in comparison.segments_a
            ],
        },
        "model_b": {
            "name": comparison.model_b,
            "cost_usd": round(comparison.cost_b, 4),
            "segment_count": len(comparison.segments_b),
            "metrics": metrics_to_dict(comparison.metrics_b),
            "segments": [
                {
                    "start_ms": s.start_ms,
                    "end_ms": s.end_ms,
                    "summary": s.summary,
                    "topic_tags": s.topic_tags,
                    "density_score": s.density_score,
                }
                for s in comparison.segments_b
            ],
        },
        "cost_difference": f"{comparison.cost_a / comparison.cost_b:.1f}x" if comparison.cost_b > 0 else "N/A",
        "boundary_agreement": comparison.boundary_agreement,
        "interpretation": _interpret_comparison(comparison),
    }


def _interpret_comparison(comparison) -> dict:
    """Generate human-readable interpretation of the comparison."""
    ma = comparison.metrics_a
    mb = comparison.metrics_b
    
    if not ma or not mb:
        return {"summary": "Unable to compare - missing data"}
    
    insights = []
    winner_points = {"a": 0, "b": 0}
    
    # Compare specificity
    if ma.specificity_score > mb.specificity_score + 0.1:
        insights.append(f"{comparison.model_a} has more specific/detailed analysis")
        winner_points["a"] += 1
    elif mb.specificity_score > ma.specificity_score + 0.1:
        insights.append(f"{comparison.model_b} has more specific/detailed analysis")
        winner_points["b"] += 1
    
    # Compare coverage
    if ma.coverage_pct > mb.coverage_pct + 5:
        insights.append(f"{comparison.model_a} covers more of the episode ({ma.coverage_pct}% vs {mb.coverage_pct}%)")
        winner_points["a"] += 1
    elif mb.coverage_pct > ma.coverage_pct + 5:
        insights.append(f"{comparison.model_b} covers more of the episode ({mb.coverage_pct}% vs {ma.coverage_pct}%)")
        winner_points["b"] += 1
    
    # Compare tag diversity
    if ma.unique_tags > mb.unique_tags + 2:
        insights.append(f"{comparison.model_a} identifies more diverse topics ({ma.unique_tags} vs {mb.unique_tags} unique tags)")
        winner_points["a"] += 1
    elif mb.unique_tags > ma.unique_tags + 2:
        insights.append(f"{comparison.model_b} identifies more diverse topics ({mb.unique_tags} vs {ma.unique_tags} unique tags)")
        winner_points["b"] += 1
    
    # Summary length (prefer 15-40 words)
    ideal_summary = 25
    a_summary_quality = 1 - abs(ma.avg_summary_length - ideal_summary) / ideal_summary
    b_summary_quality = 1 - abs(mb.avg_summary_length - ideal_summary) / ideal_summary
    if a_summary_quality > b_summary_quality + 0.2:
        insights.append(f"{comparison.model_a} has better summary length ({ma.avg_summary_length:.0f} words avg)")
        winner_points["a"] += 1
    elif b_summary_quality > a_summary_quality + 0.2:
        insights.append(f"{comparison.model_b} has better summary length ({mb.avg_summary_length:.0f} words avg)")
        winner_points["b"] += 1
    
    # Cost efficiency
    cost_ratio = comparison.cost_a / comparison.cost_b if comparison.cost_b > 0 else 1
    
    # Determine winner
    if winner_points["a"] > winner_points["b"]:
        winner = comparison.model_a
        winner_score = winner_points["a"]
    elif winner_points["b"] > winner_points["a"]:
        winner = comparison.model_b
        winner_score = winner_points["b"]
    else:
        winner = "tie"
        winner_score = 0
    
    return {
        "winner": winner,
        "winner_score": f"{winner_score}/{winner_points['a'] + winner_points['b']} metrics",
        "cost_ratio": f"{cost_ratio:.1f}x",
        "worth_premium": winner == comparison.model_a and cost_ratio > 1,
        "insights": insights,
        "recommendation": (
            f"Use {comparison.model_a} - quality improvement justifies {cost_ratio:.0f}x cost"
            if winner == comparison.model_a and cost_ratio > 1 else
            f"Use {comparison.model_b} - similar quality at {cost_ratio:.0f}x lower cost"
            if winner == comparison.model_b or winner == "tie" else
            "Results are comparable - use cheaper model"
        ),
    }


# === SMART SEARCH ENDPOINT ===
from app.services.smart_search import smart_search as do_smart_search

@router.get("/smart-search")
async def smart_search_endpoint(
    q: str,
    limit: int = 10,
    debug: bool = False,
):
    """
    Smart search with LLM-powered query expansion and relevance filtering.
    
    Higher quality results but slower (~2-5s) than regular search.
    
    - **q**: Search query (e.g., "AGI", "Tesla", "cryptocurrency")
    - **limit**: Max results (default 10)
    - **debug**: Include debug info in response
    """
    result = do_smart_search(query=q, limit=limit, debug=debug)
    return result
