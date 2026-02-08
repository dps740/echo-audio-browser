"""
Search V3: Cluster-based adaptive clip length.

Key insight: The matches themselves define the clip boundaries.
- Broad query matches many sentences across time → multiple clips or long clip
- Specific query matches few sentences clustered together → short clip
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
import openai
from app.config import settings


@dataclass
class SearchMatch:
    """A single sentence match."""
    sentence_idx: int
    text: str
    start_ms: int
    end_ms: int
    score: float


@dataclass  
class MatchCluster:
    """A group of matches close in time = one result clip."""
    matches: List[SearchMatch]
    start_ms: int
    end_ms: int
    best_score: float
    
    @property
    def duration_s(self) -> float:
        return (self.end_ms - self.start_ms) / 1000
    
    @property
    def best_match(self) -> SearchMatch:
        return max(self.matches, key=lambda m: m.score)
    
    @property
    def snippet(self) -> str:
        best = self.best_match
        return best.text[:200]


def get_query_embedding(query: str) -> np.ndarray:
    """Get embedding for search query."""
    client = openai.OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=[query]
    )
    return np.array(response.data[0].embedding)


def search_sentences(
    query: str,
    sentences: List,  # List of Sentence objects with .embedding
    top_k: int = 50
) -> List[SearchMatch]:
    """
    Find top matching sentences for a query.
    
    Combines:
    - Semantic similarity (embedding cosine)
    - Keyword boost (literal match)
    """
    query_embedding = get_query_embedding(query)
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    
    matches = []
    query_lower = query.lower()
    
    for i, sent in enumerate(sentences):
        if sent.embedding is None:
            continue
        
        # Semantic score
        sent_norm = sent.embedding / np.linalg.norm(sent.embedding)
        semantic_score = float(np.dot(query_norm, sent_norm))
        
        # Keyword boost
        keyword_boost = 0.15 if query_lower in sent.text.lower() else 0
        
        total_score = semantic_score + keyword_boost
        
        matches.append(SearchMatch(
            sentence_idx=i,
            text=sent.text,
            start_ms=sent.start_ms,
            end_ms=sent.end_ms,
            score=total_score
        ))
    
    # Sort by score, return top_k
    matches.sort(key=lambda m: -m.score)
    return matches[:top_k]


def cluster_matches(
    matches: List[SearchMatch],
    gap_threshold_ms: int = 60_000,  # 60 seconds = new cluster
    padding_ms: int = 5_000  # 5 second padding on clips
) -> List[MatchCluster]:
    """
    Group matches into clusters based on time proximity.
    
    If gap between consecutive matches > threshold, start new cluster.
    """
    if not matches:
        return []
    
    # Sort by time
    sorted_matches = sorted(matches, key=lambda m: m.start_ms)
    
    clusters = []
    current_cluster = [sorted_matches[0]]
    
    for match in sorted_matches[1:]:
        prev_match = current_cluster[-1]
        gap = match.start_ms - prev_match.end_ms
        
        if gap > gap_threshold_ms:
            # Start new cluster
            clusters.append(_make_cluster(current_cluster, padding_ms))
            current_cluster = [match]
        else:
            current_cluster.append(match)
    
    # Don't forget last cluster
    if current_cluster:
        clusters.append(_make_cluster(current_cluster, padding_ms))
    
    # Sort clusters by best score
    clusters.sort(key=lambda c: -c.best_score)
    
    return clusters


def _make_cluster(matches: List[SearchMatch], padding_ms: int) -> MatchCluster:
    """Create a cluster from a list of matches."""
    start_ms = max(0, min(m.start_ms for m in matches) - padding_ms)
    end_ms = max(m.end_ms for m in matches) + padding_ms
    
    # Cluster score combines:
    # - Best individual match score (weight: 0.4)
    # - Number of matches / density (weight: 0.4) 
    # - Average score of all matches (weight: 0.2)
    best_individual = max(m.score for m in matches)
    avg_score = sum(m.score for m in matches) / len(matches)
    match_count_factor = min(len(matches) / 10, 1.0)  # Caps at 10 matches
    
    cluster_score = (
        0.4 * best_individual +
        0.4 * match_count_factor +
        0.2 * avg_score
    )
    
    return MatchCluster(
        matches=matches,
        start_ms=start_ms,
        end_ms=end_ms,
        best_score=cluster_score  # Now a composite score
    )


def search_with_clusters(
    query: str,
    sentences: List,
    top_k_sentences: int = 50,
    gap_threshold_ms: int = 60_000,
    max_clusters: int = 5
) -> Tuple[List[MatchCluster], List[SearchMatch]]:
    """
    Full search pipeline:
    1. Find top matching sentences
    2. Cluster by time proximity
    3. Return clusters (clips) ranked by best match score
    
    Returns:
        (clusters, all_matches) - Clusters for display, raw matches for debugging
    """
    # Step 1: Find matches
    matches = search_sentences(query, sentences, top_k=top_k_sentences)
    
    if not matches:
        return [], []
    
    # Step 2: Cluster
    clusters = cluster_matches(matches, gap_threshold_ms=gap_threshold_ms)
    
    # Step 3: Limit results
    return clusters[:max_clusters], matches


def format_cluster_for_display(cluster: MatchCluster, episode_duration_ms: int) -> dict:
    """Format a cluster for API response."""
    return {
        "start_ms": cluster.start_ms,
        "end_ms": cluster.end_ms,
        "duration_s": round(cluster.duration_s, 1),
        "score": round(cluster.best_score, 3),
        "match_count": len(cluster.matches),
        "snippet": cluster.snippet,
        "start_formatted": _format_time(cluster.start_ms),
        "end_formatted": _format_time(cluster.end_ms),
        "pct_of_episode": round(cluster.duration_s / (episode_duration_ms / 1000) * 100, 1)
    }


def _format_time(ms: int) -> str:
    """Format milliseconds as MM:SS."""
    total_seconds = ms // 1000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:02d}"


# Test function
if __name__ == "__main__":
    from pathlib import Path
    from app.services.segmentation_v3 import segment_transcript_v3
    
    # Load and segment
    vtt_path = Path("audio/gXY1kx7zlkk.en.vtt")
    vtt_content = vtt_path.read_text()
    segments, sentences = segment_transcript_v3(vtt_content)
    
    print(f"Loaded {len(sentences)} sentences")
    print()
    
    # Test queries
    test_queries = ["AI", "Chinese AI", "Davos", "Trump", "immigration"]
    
    for query in test_queries:
        print(f"=" * 60)
        print(f"Query: '{query}'")
        print("=" * 60)
        
        clusters, matches = search_with_clusters(query, sentences)
        
        print(f"Top matches: {len(matches)}")
        print(f"Clusters: {len(clusters)}")
        print()
        
        for i, cluster in enumerate(clusters[:3]):
            print(f"Cluster {i+1}:")
            print(f"  Time: {_format_time(cluster.start_ms)} - {_format_time(cluster.end_ms)} ({cluster.duration_s:.0f}s)")
            print(f"  Matches: {len(cluster.matches)}")
            print(f"  Score: {cluster.best_score:.3f}")
            print(f"  Snippet: {cluster.snippet[:100]}...")
            print()
