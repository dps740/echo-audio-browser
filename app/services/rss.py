"""RSS feed parsing service."""

import feedparser
import httpx
from typing import List, Optional, Tuple
from datetime import datetime
import uuid

from app.models import PodcastFeed, Episode


async def fetch_feed(rss_url: str) -> Tuple[PodcastFeed, List[Episode]]:
    """
    Fetch and parse an RSS feed.
    
    Returns:
        Tuple of (PodcastFeed, List[Episode])
    """
    # Fetch the feed
    async with httpx.AsyncClient() as client:
        response = await client.get(str(rss_url), follow_redirects=True)
        response.raise_for_status()
        content = response.text
    
    # Parse with feedparser
    feed = feedparser.parse(content)
    
    if not feed.feed:
        raise ValueError(f"Invalid RSS feed: {rss_url}")
    
    # Create PodcastFeed
    feed_id = str(uuid.uuid4())
    podcast = PodcastFeed(
        id=feed_id,
        title=feed.feed.get('title', 'Unknown Podcast'),
        rss_url=rss_url,
        description=feed.feed.get('description'),
        author=feed.feed.get('author'),
        image_url=_get_image_url(feed),
        created_at=datetime.utcnow(),
    )
    
    # Parse episodes
    episodes = []
    for entry in feed.entries:
        audio_url = _get_audio_url(entry)
        if not audio_url:
            continue  # Skip entries without audio
        
        episode = Episode(
            id=str(uuid.uuid4()),
            feed_id=feed_id,
            title=entry.get('title', 'Untitled Episode'),
            audio_url=audio_url,
            description=entry.get('summary'),
            published_at=_parse_date(entry.get('published')),
            duration_sec=_parse_duration(entry),
            created_at=datetime.utcnow(),
        )
        episodes.append(episode)
    
    return podcast, episodes


def _get_image_url(feed) -> Optional[str]:
    """Extract podcast image URL from feed."""
    # Try itunes:image first
    if hasattr(feed.feed, 'image') and feed.feed.image:
        if hasattr(feed.feed.image, 'href'):
            return feed.feed.image.href
        if hasattr(feed.feed.image, 'url'):
            return feed.feed.image.url
    return None


def _get_audio_url(entry) -> Optional[str]:
    """Extract audio URL from feed entry."""
    # Check enclosures (standard podcast format)
    if hasattr(entry, 'enclosures'):
        for enclosure in entry.enclosures:
            if enclosure.get('type', '').startswith('audio/'):
                return enclosure.get('href') or enclosure.get('url')
    
    # Check links
    if hasattr(entry, 'links'):
        for link in entry.links:
            if link.get('type', '').startswith('audio/'):
                return link.get('href')
    
    return None


def _parse_date(date_str: Optional[str]) -> Optional[datetime]:
    """Parse a date string from RSS feed."""
    if not date_str:
        return None
    try:
        import email.utils
        parsed = email.utils.parsedate_to_datetime(date_str)
        return parsed
    except:
        return None


def _parse_duration(entry) -> Optional[int]:
    """Parse episode duration in seconds."""
    # Try itunes:duration
    duration = entry.get('itunes_duration')
    if duration:
        try:
            # Format: HH:MM:SS or MM:SS or just seconds
            parts = str(duration).split(':')
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            elif len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
            else:
                return int(duration)
        except:
            pass
    return None
