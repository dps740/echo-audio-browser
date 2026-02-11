"""
Search implementation using Typesense.
Searches sentences, groups by topic, returns ranked results.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field

from app.services.typesense_client import client


@dataclass
class SearchMatch:
    """A matching sentence within a topic."""
    text: str
    start_ms: int
    end_ms: int
    score: float


@dataclass 
class TopicResult:
    """A topic result with matching sentences."""
    topic_id: str
    episode_id: str
    summary: str
    start_ms: int
    end_ms: int
    people: List[str] = field(default_factory=list)
    companies: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    matches: List[SearchMatch] = field(default_factory=list)
    best_match: Optional[SearchMatch] = None
    score: float = 0.0
    match_count: int = 0


def search(
    query: str,
    episode_id: Optional[str] = None,
    limit: int = 10
) -> List[TopicResult]:
    """
    Search for topics matching the query.
    
    1. Searches sentences using BM25
    2. Groups matched sentences by topic
    3. Scores and ranks topics
    4. Returns topics with highlighted matches
    
    Args:
        query: Search query
        episode_id: Optional filter by episode
        limit: Max topics to return
        
    Returns:
        List of TopicResult, ranked by relevance
    """
    # Build filter
    filter_by = f"episode_id:={episode_id}" if episode_id else ""
    
    # Search sentences
    search_params = {
        "q": query,
        "query_by": "text,people,companies,topics",
        "query_by_weights": "4,2,2,1",  # Prioritize text matches
        "per_page": 250,  # Get many matches to group
        "highlight_full_fields": "text",
    }
    
    if filter_by:
        search_params["filter_by"] = filter_by
    
    try:
        results = client.collections['sentences'].documents.search(search_params)
    except Exception as e:
        print(f"Search error: {e}")
        return []
    
    # Group by topic
    topic_matches: Dict[str, List[Dict]] = {}
    
    for hit in results.get('hits', []):
        doc = hit['document']
        topic_id = doc.get('topic_id', '')
        
        if topic_id not in topic_matches:
            topic_matches[topic_id] = []
        
        # Get highlighted text if available
        highlight = hit.get('highlight', {})
        highlighted_text = highlight.get('text', {}).get('snippet', doc.get('text', ''))
        
        topic_matches[topic_id].append({
            'text': highlighted_text,
            'original_text': doc.get('text', ''),
            'start_ms': doc.get('start_ms', 0),
            'end_ms': doc.get('end_ms', 0),
            'score': hit.get('text_match', 0),
            'episode_id': doc.get('episode_id', '')
        })
    
    if not topic_matches:
        return []
    
    # Fetch topic details
    topic_ids = list(topic_matches.keys())
    topic_details = _fetch_topics(topic_ids)
    
    # Build and rank results
    topic_results = []
    
    for topic_id, matches in topic_matches.items():
        if topic_id not in topic_details:
            continue
        
        topic = topic_details[topic_id]
        
        # Sort matches by score
        matches.sort(key=lambda m: m['score'], reverse=True)
        
        # Create SearchMatch objects
        search_matches = [
            SearchMatch(
                text=m['text'],
                start_ms=m['start_ms'],
                end_ms=m['end_ms'],
                score=m['score']
            )
            for m in matches[:5]  # Top 5 matches per topic
        ]
        
        # Calculate topic score: best match score * sqrt(match count) for density bonus
        best_score = matches[0]['score'] if matches else 0
        density_bonus = len(matches) ** 0.5
        topic_score = best_score * density_bonus
        
        topic_results.append(TopicResult(
            topic_id=topic_id,
            episode_id=topic.get('episode_id', ''),
            summary=topic.get('summary', ''),
            start_ms=topic.get('start_ms', 0),
            end_ms=topic.get('end_ms', 0),
            people=topic.get('people', []),
            companies=topic.get('companies', []),
            keywords=topic.get('keywords', []),
            matches=search_matches,
            best_match=search_matches[0] if search_matches else None,
            score=topic_score,
            match_count=len(matches)
        ))
    
    # Sort by score and limit
    topic_results.sort(key=lambda t: t.score, reverse=True)
    
    return topic_results[:limit]


def _fetch_topics(topic_ids: List[str]) -> Dict[str, Dict]:
    """Fetch topic details by IDs."""
    topics = {}
    
    # Batch fetch by ID
    for topic_id in topic_ids:
        try:
            doc = client.collections['topics'].documents[topic_id].retrieve()
            topics[topic_id] = doc
        except Exception:
            continue
    
    return topics


def search_simple(query: str, episode_id: Optional[str] = None, limit: int = 10) -> List[Dict]:
    """
    Simplified search that returns dicts instead of dataclasses.
    Useful for API responses.
    """
    results = search(query, episode_id, limit)
    
    return [
        {
            "topic_id": r.topic_id,
            "episode_id": r.episode_id,
            "summary": r.summary,
            "start_ms": r.start_ms,
            "end_ms": r.end_ms,
            "duration_seconds": (r.end_ms - r.start_ms) / 1000,
            "people": r.people,
            "companies": r.companies,
            "keywords": r.keywords,
            "score": r.score,
            "match_count": r.match_count,
            "best_match": {
                "text": r.best_match.text if r.best_match else "",
                "start_ms": r.best_match.start_ms if r.best_match else 0,
                "timestamp": _format_time(r.best_match.start_ms) if r.best_match else ""
            } if r.best_match else None
        }
        for r in results
    ]


def _format_time(ms: int) -> str:
    """Format milliseconds as MM:SS."""
    total_seconds = ms // 1000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:02d}"


if __name__ == "__main__":
    import sys
    
    query = sys.argv[1] if len(sys.argv) > 1 else "AI"
    
    print(f"Searching for: '{query}'")
    print("=" * 60)
    
    results = search(query)
    
    if not results:
        print("No results found")
    else:
        for i, r in enumerate(results, 1):
            print(f"\n{i}. {r.summary}")
            print(f"   Score: {r.score:.1f} ({r.match_count} matches)")
            print(f"   Time: {_format_time(r.start_ms)} - {_format_time(r.end_ms)}")
            if r.best_match:
                text = r.best_match.text[:100] + "..." if len(r.best_match.text) > 100 else r.best_match.text
                print(f"   Best match at {_format_time(r.best_match.start_ms)}: \"{text}\"")
            if r.people:
                print(f"   People: {r.people[:3]}")
