"""
Search API endpoints using Typesense.
"""

from fastapi import APIRouter, Query
from typing import Optional, List

from app.services.typesense_search import search_simple
from app.services.typesense_search_v3 import search_simple_v3, search_categorized
from app.services.typesense_client import get_collections_info, health_check
from app.services.typesense_indexer import get_index_stats
from app.services.search_l3 import search_l3

router = APIRouter(prefix="/v2", tags=["search-v2"])


@router.get("/search")
async def search_topics(
    q: str = Query(..., description="Search query"),
    episode_id: Optional[str] = Query(None, description="Filter by episode ID"),
    limit: int = Query(10, ge=1, le=50, description="Max results")
):
    """
    Search for podcast topics matching the query.
    
    Returns topics ranked by relevance, with the best matching sentence highlighted.
    """
    results = search_simple(q, episode_id, limit)
    
    return {
        "query": q,
        "episode_id": episode_id,
        "count": len(results),
        "results": results
    }


@router.get("/search/v3")
async def search_topics_v3(
    q: str = Query(..., description="Search query"),
    episode_id: Optional[str] = Query(None, description="Filter by episode ID"),
    limit: int = Query(10, ge=1, le=50, description="Max results")
):
    """
    Search for podcast topics using two-stage topic-first approach.
    
    Stage 1: Search topics by summary/keywords/companies (finds aboutness)
    Stage 2: Search sentences within matched topics (finds precise timestamp)
    
    Returns topics that are ABOUT the query, not just mentioning it.
    """
    results = search_simple_v3(q, episode_id, limit)
    
    return {
        "query": q,
        "episode_id": episode_id,
        "count": len(results),
        "results": results,
        "search_mode": "topic-first"
    }


@router.get("/search/v4")
async def search_topics_categorized(
    q: str = Query(..., description="Search query"),
    episode_id: Optional[str] = Query(None, description="Filter by episode ID"),
    limit: int = Query(10, ge=1, le=50, description="Max results per category")
):
    """
    Search with results split into two categories:
    
    - ABOUT: Topics where the query is a main subject (in summary)
             → These are the primary autoplay results
    
    - MENTIONS: Topics that mention the query but aren't primarily about it
               → Optional secondary list for deeper exploration
    
    This prevents tangential mentions from cluttering the main results.
    """
    results = search_categorized(q, episode_id, limit)
    results["search_mode"] = "categorized"
    return results


@router.get("/search/l3")
async def search_level3(
    q: str = Query(..., description="Search query"),
    limit: int = Query(10, ge=1, le=20, description="Max results"),
    min_confidence: str = Query("low", description="Minimum confidence: high, medium, low")
):
    """
    Level 3 Commercial-Grade Search.
    
    Full pipeline:
    1. Query Understanding - LLM expands query with synonyms + identifies intent
    2. Hybrid Retrieval - BM25 + vector similarity (RRF fusion)
    3. Cross-Encoder Reranking - LLM judges relevance with explanations
    4. Confidence Scoring - High/Medium/Low per result
    
    Returns results categorized as:
    - ABOUT: Topics that are primarily about the query
    - MENTIONS: Topics that mention the query
    - RELATED: Semantically related topics
    
    Each result includes:
    - relevance_score (0-1)
    - confidence (high/medium/low)
    - explanation (why it matched)
    - match_type (about/mentions/related)
    """
    results = search_l3(q, limit=limit, min_confidence=min_confidence)
    results["search_mode"] = "level3"
    return results


@router.get("/health")
async def typesense_health():
    """Check Typesense health and index stats."""
    health = health_check()
    stats = get_index_stats()
    
    return {
        "typesense": health,
        "index": stats
    }


@router.get("/stats")
async def index_stats():
    """Get detailed index statistics."""
    return get_index_stats()
