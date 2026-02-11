"""
Search v3: Two-stage search implementation.

Stage 1: Search TOPICS to find content that is ABOUT the query
Stage 2: Search SENTENCES within matched topics for precise timestamps
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
    topic_score: float = 0.0  # Score from topic search
    sentence_score: float = 0.0  # Score from sentence search
    match_count: int = 0


def search_topics_first(
    query: str,
    episode_id: Optional[str] = None,
    limit: int = 10
) -> List[TopicResult]:
    """
    Two-stage search with summary-first approach:
    
    Stage 1a: Search TOPICS by summary (finds topics ABOUT the query)
    Stage 1b: If few results, expand to keywords/companies (topics MENTIONING the query)
    Stage 2: Search SENTENCES within matched topics (finds precise timestamp)
    
    Args:
        query: Search query
        episode_id: Optional filter by episode
        limit: Max topics to return
        
    Returns:
        List of TopicResult, ranked by topic relevance
    """
    
    topic_filter = f"episode_id:={episode_id}" if episode_id else ""
    
    # === STAGE 1a: Search topics by SUMMARY first (highest relevance) ===
    summary_search_params = {
        "q": query,
        "query_by": "summary",  # Only summary - topics ABOUT the query
        "per_page": limit,
        "highlight_full_fields": "summary",
    }
    
    if topic_filter:
        summary_search_params["filter_by"] = topic_filter
    
    try:
        summary_results = client.collections['topics'].documents.search(summary_search_params)
    except Exception as e:
        print(f"Summary search error: {e}")
        summary_results = {'hits': [], 'found': 0}
    
    # === STAGE 1b: If few summary matches, expand to keywords/companies ===
    topic_hits = summary_results.get('hits', [])
    summary_ids = {h['document']['id'] for h in topic_hits}
    
    if len(topic_hits) < limit:
        # Search keywords/companies for additional results
        expanded_search_params = {
            "q": query,
            "query_by": "keywords,companies,people",
            "per_page": limit * 2,
            "highlight_full_fields": "summary",
        }
        
        if topic_filter:
            expanded_search_params["filter_by"] = topic_filter
        
        try:
            expanded_results = client.collections['topics'].documents.search(expanded_search_params)
            # Add results not already in summary matches
            for hit in expanded_results.get('hits', []):
                if hit['document']['id'] not in summary_ids and len(topic_hits) < limit:
                    topic_hits.append(hit)
                    summary_ids.add(hit['document']['id'])
        except Exception as e:
            print(f"Expanded search error: {e}")
    
    if not topic_hits:
        return []
    
    # === STAGE 2: For each topic, find best sentence match ===
    results = []
    
    for hit in topic_hits[:limit]:
        topic_doc = hit['document']
        topic_id = topic_doc['id']
        topic_score = hit.get('text_match', 0)
        
        # Search sentences within this topic
        sentence_search_params = {
            "q": query,
            "query_by": "text",
            "filter_by": f"topic_id:={topic_id}",
            "per_page": 5,
            "highlight_full_fields": "text",
        }
        
        try:
            sentence_results = client.collections['sentences'].documents.search(sentence_search_params)
        except Exception as e:
            print(f"Sentence search error for topic {topic_id}: {e}")
            sentence_results = {'hits': []}
        
        # Build matches from sentences
        matches = []
        for s_hit in sentence_results.get('hits', []):
            s_doc = s_hit['document']
            highlight = s_hit.get('highlight', {})
            highlighted_text = highlight.get('text', {}).get('snippet', s_doc.get('text', ''))
            
            matches.append(SearchMatch(
                text=highlighted_text,
                start_ms=s_doc.get('start_ms', 0),
                end_ms=s_doc.get('end_ms', 0),
                score=s_hit.get('text_match', 0)
            ))
        
        # Use topic start if no sentence matches (query might match summary/keywords only)
        best_match = matches[0] if matches else SearchMatch(
            text=topic_doc.get('summary', ''),
            start_ms=topic_doc.get('start_ms', 0),
            end_ms=topic_doc.get('start_ms', 0) + 5000,  # 5 sec
            score=0
        )
        
        results.append(TopicResult(
            topic_id=topic_id,
            episode_id=topic_doc.get('episode_id', ''),
            summary=topic_doc.get('summary', ''),
            start_ms=topic_doc.get('start_ms', 0),
            end_ms=topic_doc.get('end_ms', 0),
            people=topic_doc.get('people', []),
            companies=topic_doc.get('companies', []),
            keywords=topic_doc.get('keywords', []),
            matches=matches[:5],
            best_match=best_match,
            topic_score=topic_score,
            sentence_score=matches[0].score if matches else 0,
            match_count=len(matches)
        ))
    
    return results


def search_simple_v3(query: str, episode_id: Optional[str] = None, limit: int = 10) -> List[Dict]:
    """
    Simplified search that returns dicts instead of dataclasses.
    Uses two-stage topic-first search.
    """
    results = search_topics_first(query, episode_id, limit)
    
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
            "topic_score": r.topic_score,
            "match_count": r.match_count,
            "best_match": {
                "text": r.best_match.text if r.best_match else "",
                "start_ms": r.best_match.start_ms if r.best_match else r.start_ms,
                "timestamp": _format_time(r.best_match.start_ms if r.best_match else r.start_ms)
            }
        }
        for r in results
    ]


def search_categorized(query: str, episode_id: Optional[str] = None, limit: int = 10) -> Dict:
    """
    Search with results categorized into ABOUT vs MENTIONS.
    
    Returns:
        {
            "about": [...],      # Topics ABOUT the query (for autoplay)
            "mentions": [...],   # Topics that MENTION the query (optional)
            "query": str,
            "total": int
        }
    """
    results = search_topics_first(query, episode_id, limit * 2)  # Get more to split
    
    query_lower = query.lower()
    
    about_results = []
    mention_results = []
    
    for r in results:
        result_dict = {
            "topic_id": r.topic_id,
            "episode_id": r.episode_id,
            "summary": r.summary,
            "start_ms": r.start_ms,
            "end_ms": r.end_ms,
            "duration_seconds": (r.end_ms - r.start_ms) / 1000,
            "people": r.people,
            "companies": r.companies,
            "keywords": r.keywords,
            "topic_score": r.topic_score,
            "match_count": r.match_count,
            "best_match": {
                "text": r.best_match.text if r.best_match else "",
                "start_ms": r.best_match.start_ms if r.best_match else r.start_ms,
                "timestamp": _format_time(r.best_match.start_ms if r.best_match else r.start_ms)
            }
        }
        
        # Categorize: ABOUT if query is in summary, otherwise MENTIONS
        if query_lower in r.summary.lower():
            if len(about_results) < limit:
                about_results.append(result_dict)
        else:
            if len(mention_results) < limit:
                mention_results.append(result_dict)
    
    return {
        "query": query,
        "about": about_results,
        "mentions": mention_results,
        "about_count": len(about_results),
        "mentions_count": len(mention_results),
        "total": len(about_results) + len(mention_results)
    }


def _format_time(ms: int) -> str:
    """Format milliseconds as MM:SS."""
    total_seconds = ms // 1000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:02d}"


if __name__ == "__main__":
    import sys
    import json
    
    query = sys.argv[1] if len(sys.argv) > 1 else "Tesla"
    
    print(f"Searching for: '{query}' (topic-first approach)")
    print("=" * 60)
    
    results = search_topics_first(query)
    
    if not results:
        print("No results found")
    else:
        for i, r in enumerate(results, 1):
            print(f"\n{i}. {r.summary}")
            print(f"   Topic score: {r.topic_score}")
            print(f"   Sentence matches: {r.match_count}")
            print(f"   Keywords: {r.keywords}")
            print(f"   Companies: {r.companies}")
            if r.best_match:
                text = r.best_match.text[:80] + "..." if len(r.best_match.text) > 80 else r.best_match.text
                print(f"   Best match at {_format_time(r.best_match.start_ms)}: \"{text}\"")
