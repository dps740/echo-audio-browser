"""Feed management endpoints."""

from fastapi import APIRouter, HTTPException
from typing import List, Dict
import uuid
from datetime import datetime

from app.models import FeedCreate, FeedResponse, PodcastFeed, Episode
from app.services.rss import fetch_feed


router = APIRouter(prefix="/feeds", tags=["feeds"])

# In-memory storage for MVP (would be DB in production)
_feeds: Dict[str, PodcastFeed] = {}
_episodes: Dict[str, Episode] = {}


@router.post("", response_model=FeedResponse)
async def add_feed(feed_create: FeedCreate):
    """Add a new podcast RSS feed."""
    try:
        podcast, episodes = await fetch_feed(str(feed_create.rss_url))
        
        # Store feed and episodes
        _feeds[podcast.id] = podcast
        for ep in episodes:
            _episodes[ep.id] = ep
        
        return FeedResponse(
            id=podcast.id,
            title=podcast.title,
            rss_url=feed_create.rss_url,
            episode_count=len(episodes),
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("", response_model=List[FeedResponse])
async def list_feeds():
    """List all subscribed feeds."""
    return [
        FeedResponse(
            id=feed.id,
            title=feed.title,
            rss_url=feed.rss_url,
            episode_count=len([e for e in _episodes.values() if e.feed_id == feed.id]),
        )
        for feed in _feeds.values()
    ]


@router.get("/{feed_id}")
async def get_feed(feed_id: str):
    """Get feed details with episodes."""
    if feed_id not in _feeds:
        raise HTTPException(status_code=404, detail="Feed not found")
    
    feed = _feeds[feed_id]
    episodes = [e for e in _episodes.values() if e.feed_id == feed_id]
    
    return {
        "feed": feed,
        "episodes": episodes,
    }


@router.delete("/{feed_id}")
async def delete_feed(feed_id: str):
    """Remove a feed and its episodes."""
    if feed_id not in _feeds:
        raise HTTPException(status_code=404, detail="Feed not found")
    
    # Remove episodes
    episode_ids_to_remove = [
        ep_id for ep_id, ep in _episodes.items() if ep.feed_id == feed_id
    ]
    for ep_id in episode_ids_to_remove:
        del _episodes[ep_id]
    
    # Remove feed
    del _feeds[feed_id]
    
    return {"status": "deleted", "episodes_removed": len(episode_ids_to_remove)}


# Export storage for other routers
def get_feeds_storage():
    return _feeds, _episodes
