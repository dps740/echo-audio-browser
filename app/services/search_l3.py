"""
Level 3 Commercial-Grade Search Pipeline

Architecture:
1. Query Understanding - LLM expands query with synonyms + intent
2. Hybrid Retrieval - BM25 (Typesense) + Vector similarity
3. Cross-Encoder Reranking - LLM compares candidates pairwise
4. Confidence + Explanation - Why did this result match?

This is the production-grade search stack used by Perplexity, Cohere, etc.
"""

import os
import json
import asyncio
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from openai import OpenAI

from app.services.typesense_client import client as ts_client
from app.services.clip_extractor import get_clip_url


# Initialize OpenAI client
openai_client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Embedding model
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536

# LLM model for query understanding and reranking
LLM_MODEL = "gpt-4o-mini"


@dataclass
class SearchCandidate:
    """A candidate result from hybrid retrieval."""
    topic_id: str
    episode_id: str
    summary: str
    full_text: str  # Actual transcript text
    start_ms: int
    end_ms: int
    people: List[str] = field(default_factory=list)
    companies: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    bm25_score: float = 0.0
    vector_score: float = 0.0
    combined_score: float = 0.0
    source: str = ""  # "bm25", "vector", or "both"


@dataclass
class RankedResult:
    """A final result after reranking."""
    topic_id: str
    episode_id: str
    summary: str
    start_ms: int
    end_ms: int
    people: List[str] = field(default_factory=list)
    companies: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    relevance_score: float = 0.0
    confidence: str = ""  # "high", "medium", "low"
    explanation: str = ""  # Why this matched
    match_type: str = ""  # "about", "mentions", "related"


@dataclass
class QueryUnderstanding:
    """Result of query understanding phase."""
    original_query: str
    expanded_terms: List[str]
    intent: str  # What the user is looking for
    entity_types: List[str]  # PERSON, COMPANY, TOPIC, etc.


# =============================================================================
# PHASE 1: Query Understanding
# =============================================================================

def understand_query(query: str) -> QueryUnderstanding:
    """
    Use LLM to understand and expand the query.
    
    - Identifies synonyms and related terms
    - Understands user intent
    - Identifies entity types being searched
    """
    
    prompt = f"""Analyze this search query for a podcast search engine:

Query: "{query}"

Return a JSON object with:
1. "expanded_terms": List of 3-7 search terms including synonyms and related concepts. Include the original query term.
   - For "currency" include: ["currency", "money", "fiat", "stablecoin", "monetary", "dollar", "crypto"]
   - For "space" (if about aerospace): ["space", "SpaceX", "NASA", "rocket", "satellite", "orbit"]
   - Be specific to the likely user intent

2. "intent": One sentence describing what the user is looking for

3. "entity_types": List of entity types relevant to this query from: ["PERSON", "COMPANY", "TECHNOLOGY", "TOPIC", "EVENT", "PLACE"]

4. "disambiguation": If the query is ambiguous (like "space" could mean aerospace or industry jargon), note this and pick the most likely meaning for a tech podcast audience

Return ONLY valid JSON, no other text."""

    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,
        response_format={"type": "json_object"}  # Force JSON output
    )
    
    try:
        raw_content = response.choices[0].message.content
        result = json.loads(raw_content)
        expanded = result.get("expanded_terms", [query])
        # Ensure original query is included
        if query.lower() not in [t.lower() for t in expanded]:
            expanded.insert(0, query)
        return QueryUnderstanding(
            original_query=query,
            expanded_terms=expanded,
            intent=result.get("intent", f"Find content about {query}"),
            entity_types=result.get("entity_types", ["TOPIC"])
        )
    except json.JSONDecodeError as e:
        print(f"Query understanding JSON error: {e}")
        print(f"Raw content: {response.choices[0].message.content[:200]}")
        # Fallback if LLM doesn't return valid JSON
        return QueryUnderstanding(
            original_query=query,
            expanded_terms=[query],
            intent=f"Find content about {query}",
            entity_types=["TOPIC"]
        )


# =============================================================================
# PHASE 2: Hybrid Retrieval
# =============================================================================

def get_embedding(text: str) -> List[float]:
    """Get embedding vector for text."""
    response = openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
        dimensions=EMBEDDING_DIMENSIONS
    )
    return response.data[0].embedding


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def bm25_search(query_terms: List[str], limit: int = 30) -> List[SearchCandidate]:
    """
    Search Typesense using BM25 for all expanded query terms.
    Returns deduplicated candidates.
    """
    seen_ids = set()
    candidates = []
    
    for term in query_terms:
        # Search both summary and full_text fields
        search_params = {
            "q": term,
            "query_by": "summary,full_text,keywords,companies,people",
            "per_page": limit,
        }
        
        try:
            results = ts_client.collections['topics'].documents.search(search_params)
            
            for hit in results.get('hits', []):
                doc = hit['document']
                topic_id = doc['id']
                
                if topic_id in seen_ids:
                    continue
                seen_ids.add(topic_id)
                
                candidates.append(SearchCandidate(
                    topic_id=topic_id,
                    episode_id=doc.get('episode_id', ''),
                    summary=doc.get('summary', ''),
                    full_text=doc.get('full_text', ''),
                    start_ms=doc.get('start_ms', 0),
                    end_ms=doc.get('end_ms', 0),
                    people=doc.get('people', []),
                    companies=doc.get('companies', []),
                    keywords=doc.get('keywords', []),
                    bm25_score=hit.get('text_match', 0),
                    source="bm25"
                ))
        except Exception as e:
            print(f"BM25 search error for term '{term}': {e}")
    
    return candidates


def vector_search(query: str, all_topics: List[Dict], limit: int = 30) -> List[SearchCandidate]:
    """
    Search using vector similarity.
    Computes embedding for query and compares to topic embeddings.
    """
    query_embedding = get_embedding(query)
    
    scored_topics = []
    for topic in all_topics:
        if 'embedding' not in topic:
            continue
            
        similarity = cosine_similarity(query_embedding, topic['embedding'])
        scored_topics.append((topic, similarity))
    
    # Sort by similarity descending
    scored_topics.sort(key=lambda x: x[1], reverse=True)
    
    candidates = []
    for topic, score in scored_topics[:limit]:
        candidates.append(SearchCandidate(
            topic_id=topic['id'],
            episode_id=topic.get('episode_id', ''),
            summary=topic.get('summary', ''),
            full_text=topic.get('full_text', ''),
            start_ms=topic.get('start_ms', 0),
            end_ms=topic.get('end_ms', 0),
            people=topic.get('people', []),
            companies=topic.get('companies', []),
            keywords=topic.get('keywords', []),
            vector_score=score,
            source="vector"
        ))
    
    return candidates


def hybrid_retrieve(
    query_understanding: QueryUnderstanding,
    topic_embeddings: Dict[str, List[float]],
    all_topics: List[Dict],
    bm25_limit: int = 30,
    vector_limit: int = 30
) -> List[SearchCandidate]:
    """
    Combine BM25 and vector search results.
    
    Uses Reciprocal Rank Fusion (RRF) to merge rankings.
    """
    # Get BM25 candidates using expanded terms
    bm25_candidates = bm25_search(query_understanding.expanded_terms, limit=bm25_limit)
    
    # Add embeddings to topics for vector search
    topics_with_embeddings = []
    for topic in all_topics:
        if topic['id'] in topic_embeddings:
            topic_copy = topic.copy()
            topic_copy['embedding'] = topic_embeddings[topic['id']]
            topics_with_embeddings.append(topic_copy)
    
    # Get vector candidates using original query (semantic meaning)
    vector_candidates = vector_search(
        query_understanding.original_query, 
        topics_with_embeddings, 
        limit=vector_limit
    )
    
    # Merge using RRF
    rrf_scores = {}
    k = 60  # RRF constant
    
    # Score BM25 results
    for rank, candidate in enumerate(sorted(bm25_candidates, key=lambda x: x.bm25_score, reverse=True)):
        topic_id = candidate.topic_id
        rrf_scores[topic_id] = rrf_scores.get(topic_id, 0) + 1 / (k + rank + 1)
    
    # Score vector results
    for rank, candidate in enumerate(sorted(vector_candidates, key=lambda x: x.vector_score, reverse=True)):
        topic_id = candidate.topic_id
        rrf_scores[topic_id] = rrf_scores.get(topic_id, 0) + 1 / (k + rank + 1)
    
    # Build final candidate list with combined scores
    all_candidates = {c.topic_id: c for c in bm25_candidates}
    for c in vector_candidates:
        if c.topic_id in all_candidates:
            all_candidates[c.topic_id].vector_score = c.vector_score
            all_candidates[c.topic_id].source = "both"
        else:
            all_candidates[c.topic_id] = c
    
    # Update combined scores
    for topic_id, candidate in all_candidates.items():
        candidate.combined_score = rrf_scores.get(topic_id, 0)
    
    # Sort by combined score
    result = sorted(all_candidates.values(), key=lambda x: x.combined_score, reverse=True)
    return result


# =============================================================================
# PHASE 3: Cross-Encoder Reranking
# =============================================================================

def rerank_candidates(
    query: str,
    intent: str,
    candidates: List[SearchCandidate],
    top_k: int = 10
) -> List[Tuple[SearchCandidate, float, str]]:
    """
    Use LLM to rerank candidates with explanations.
    
    Returns list of (candidate, score, explanation) tuples.
    """
    if not candidates:
        return []
    
    # Limit to top candidates for efficiency
    candidates_to_rank = candidates[:min(30, len(candidates))]
    
    # Build candidate descriptions with topic_id for reliable mapping
    candidate_descriptions = []
    topic_id_to_candidate = {}  # Map for lookup by topic_id
    for i, c in enumerate(candidates_to_rank):
        desc = f"[{c.topic_id}] Summary: {c.summary}\n"
        if c.keywords:
            desc += f"   Keywords: {', '.join(c.keywords[:5])}\n"
        if c.people:
            desc += f"   People: {', '.join(c.people[:3])}\n"
        if c.companies:
            desc += f"   Companies: {', '.join(c.companies[:3])}\n"
        # Include snippet of full_text for grounding
        if c.full_text:
            snippet = c.full_text[:200] + "..." if len(c.full_text) > 200 else c.full_text
            desc += f"   Transcript excerpt: {snippet}\n"
        candidate_descriptions.append(desc)
        topic_id_to_candidate[c.topic_id] = c
    
    prompt = f"""You are a strict search relevance judge for a podcast search engine.

USER QUERY: "{query}"
USER INTENT: {intent}

CANDIDATE TOPICS:
{chr(10).join(candidate_descriptions)}

TASK: Score each topic's relevance to the user's query and intent.

For each topic, provide:
- "score": 0-100 (100 = perfectly relevant, 0 = completely irrelevant)
- "explanation": One sentence explaining why this does or doesn't match
- "match_type": STRICT classification (see rules below)

MATCH_TYPE RULES:
- "about": The query subject is A MAJOR FOCUS of the topic. Use if the topic substantially discusses the query subject, even if other subjects are also covered. For "immigration", a topic about "immigration policies and voter ID" should be "about" since immigration is a major focus.
- "mentions": The query subject is briefly mentioned or is a minor part of the discussion. Use for passing references or when the query is incidental to the main topic.
- "related": Semantically related to the query but the query subject is NOT directly discussed.

SCORING RULES:
1. Score 80-100: Query is THE central subject or one of the main subjects
2. Score 60-79: Query is substantially discussed as part of the topic
3. Score 40-59: Query is discussed but as a secondary element
4. Score 20-39: Query is mentioned briefly or tangentially
5. Score 0-19: Query is not really relevant

KEY PRINCIPLE: If the topic summary prominently features the query subject (e.g., "immigration policies", "immigration enforcement", "immigrants"), it should likely be "about" with a score >= 60.
- "Debate on immigration policies and voter ID" → "about" (immigration is a main subject)
- "Discussion about Minneapolis violence where ICE was involved" → "mentions" (focus is violence, not immigration)
- "Economic analysis that briefly references immigration" → "mentions"

Return a JSON array with objects for each topic. Use the topic_id in brackets (e.g., [gXY1kx7zlkk_t11]) to identify each topic:
[
  {{"topic_id": "gXY1kx7zlkk_t11", "score": 85, "explanation": "...", "match_type": "about"}},
  ...
]

Return ONLY the JSON array, no other text."""

    response = openai_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=2000
    )
    
    try:
        rankings = json.loads(response.choices[0].message.content)
        
        # Map scores back to candidates using topic_id
        results = []
        for ranking in rankings[:top_k]:
            topic_id = ranking.get("topic_id", "")
            if topic_id in topic_id_to_candidate:
                candidate = topic_id_to_candidate[topic_id]
                score = ranking.get("score", 0) / 100.0  # Normalize to 0-1
                explanation = ranking.get("explanation", "")
                match_type = ranking.get("match_type", "related")
                results.append((candidate, score, explanation, match_type))
            else:
                # Fallback: try index-based lookup for backwards compatibility
                idx = ranking.get("index", 0) - 1
                if 0 <= idx < len(candidates_to_rank):
                    candidate = candidates_to_rank[idx]
                    score = ranking.get("score", 0) / 100.0
                    explanation = ranking.get("explanation", "")
                    match_type = ranking.get("match_type", "related")
                    results.append((candidate, score, explanation, match_type))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results
        
    except json.JSONDecodeError as e:
        print(f"Reranking JSON parse error: {e}")
        # Fallback: return candidates sorted by combined score
        return [(c, c.combined_score, "Ranked by retrieval score", "related") 
                for c in candidates_to_rank[:top_k]]


# =============================================================================
# PHASE 4: Confidence and Final Assembly
# =============================================================================

def assign_confidence(score: float) -> str:
    """Assign confidence level based on relevance score."""
    if score >= 0.8:
        return "high"
    elif score >= 0.5:
        return "medium"
    else:
        return "low"


def build_final_results(
    ranked: List[Tuple[SearchCandidate, float, str, str]]
) -> List[RankedResult]:
    """Build final result objects with all metadata."""
    results = []
    
    for candidate, score, explanation, match_type in ranked:
        results.append(RankedResult(
            topic_id=candidate.topic_id,
            episode_id=candidate.episode_id,
            summary=candidate.summary,
            start_ms=candidate.start_ms,
            end_ms=candidate.end_ms,
            people=candidate.people,
            companies=candidate.companies,
            keywords=candidate.keywords,
            relevance_score=score,
            confidence=assign_confidence(score),
            explanation=explanation,
            match_type=match_type
        ))
    
    return results


# =============================================================================
# MAIN SEARCH FUNCTION
# =============================================================================

# Cache for topic embeddings (computed once at startup)
_topic_embeddings_cache: Dict[str, List[float]] = {}
_all_topics_cache: List[Dict] = []


def load_topic_embeddings():
    """
    Load or compute embeddings for all topics.
    Caches results for reuse.
    """
    global _topic_embeddings_cache, _all_topics_cache
    
    if _topic_embeddings_cache:
        return _topic_embeddings_cache, _all_topics_cache
    
    print("Loading topic embeddings...")
    
    # Fetch all topics from Typesense (paginated)
    topics = []
    page = 1
    per_page = 250  # Typesense max is 250
    
    try:
        while True:
            search_params = {
                "q": "*",
                "per_page": per_page,
                "page": page,
            }
            results = ts_client.collections['topics'].documents.search(search_params)
            hits = results.get('hits', [])
            if not hits:
                break
            topics.extend([hit['document'] for hit in hits])
            if len(hits) < per_page:
                break
            page += 1
        print(f"Loaded {len(topics)} topics from Typesense")
    except Exception as e:
        print(f"Error fetching topics: {e}")
        return {}, []
    
    _all_topics_cache = topics
    
    # Check if we have cached embeddings on disk
    cache_file = "data/topic_embeddings.json"
    if os.path.exists(cache_file):
        print(f"Loading cached embeddings from {cache_file}")
        with open(cache_file, 'r') as f:
            _topic_embeddings_cache = json.load(f)
        return _topic_embeddings_cache, _all_topics_cache
    
    # Compute embeddings for all topics
    print(f"Computing embeddings for {len(topics)} topics...")
    
    for i, topic in enumerate(topics):
        # Combine summary + keywords for embedding
        text_to_embed = topic.get('summary', '')
        if topic.get('keywords'):
            text_to_embed += " " + " ".join(topic['keywords'][:10])
        
        if text_to_embed.strip():
            embedding = get_embedding(text_to_embed)
            _topic_embeddings_cache[topic['id']] = embedding
        
        if (i + 1) % 50 == 0:
            print(f"  Computed {i+1}/{len(topics)} embeddings")
    
    # Save cache
    os.makedirs("data", exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(_topic_embeddings_cache, f)
    print(f"Saved embeddings to {cache_file}")
    
    return _topic_embeddings_cache, _all_topics_cache


def search_l3(
    query: str,
    limit: int = 10,
    min_confidence: str = "low"
) -> Dict[str, Any]:
    """
    Level 3 Commercial-Grade Search.
    
    Args:
        query: User's search query
        limit: Maximum results to return
        min_confidence: Minimum confidence level ("high", "medium", "low")
        
    Returns:
        {
            "query": str,
            "intent": str,
            "expanded_terms": [...],
            "results": [...],
            "about": [...],      # High-confidence results ABOUT the query
            "mentions": [...],   # Results that mention the query
            "related": [...],    # Semantically related
            "total": int,
            "search_metadata": {...}
        }
    """
    # Load embeddings (cached after first call)
    topic_embeddings, all_topics = load_topic_embeddings()
    
    # Phase 1: Query Understanding
    print(f"Phase 1: Understanding query '{query}'...")
    query_understanding = understand_query(query)
    print(f"  Intent: {query_understanding.intent}")
    print(f"  Expanded terms: {query_understanding.expanded_terms}")
    
    # Phase 2: Hybrid Retrieval
    print("Phase 2: Hybrid retrieval...")
    candidates = hybrid_retrieve(
        query_understanding,
        topic_embeddings,
        all_topics,
        bm25_limit=30,
        vector_limit=30
    )
    print(f"  Retrieved {len(candidates)} candidates")
    
    if not candidates:
        return {
            "query": query,
            "intent": query_understanding.intent,
            "expanded_terms": query_understanding.expanded_terms,
            "results": [],
            "about": [],
            "mentions": [],
            "related": [],
            "total": 0,
            "search_metadata": {"phases_completed": ["understand", "retrieve"]}
        }
    
    # Phase 3: Cross-Encoder Reranking
    print("Phase 3: Reranking with LLM...")
    ranked = rerank_candidates(
        query,
        query_understanding.intent,
        candidates,
        top_k=limit * 2  # Get more so we can filter by confidence
    )
    print(f"  Reranked {len(ranked)} results")
    
    # Phase 4: Build Final Results
    print("Phase 4: Building final results...")
    results = build_final_results(ranked)
    
    # Filter by minimum confidence
    confidence_order = {"high": 3, "medium": 2, "low": 1}
    min_conf_value = confidence_order.get(min_confidence, 1)
    results = [r for r in results if confidence_order.get(r.confidence, 0) >= min_conf_value]
    
    # Categorize results with balanced filtering
    # ABOUT: HIGH/MEDIUM confidence "about" results, OR "mentions" with high score (>=0.6)
    # The key insight: if the summary prominently features the query, it's probably ABOUT it
    about = [r for r in results 
             if (r.match_type == "about" and r.confidence in ("high", "medium"))
             or (r.match_type == "mentions" and r.confidence in ("high", "medium") and r.relevance_score >= 0.6)][:limit]
    
    # MENTIONS: Lower-scoring items that still discuss the query
    mentions = [r for r in results 
                if r not in about
                and r.match_type in ("about", "mentions")
                and r.confidence != "low"
                and r.relevance_score >= 0.3][:limit]
    
    # RELATED: Everything else - tangentially related or low confidence
    about_and_mentions = set(r.topic_id for r in about + mentions)
    related = [r for r in results 
               if r.topic_id not in about_and_mentions][:limit]
    
    # Convert to dicts for JSON serialization
    def result_to_dict(r: RankedResult) -> Dict:
        # Don't generate clips here - let frontend request via /clip endpoint
        # This avoids blocking search on ffmpeg extraction
        return {
            "topic_id": r.topic_id,
            "episode_id": r.episode_id,
            "summary": r.summary,
            "start_ms": r.start_ms,
            "end_ms": r.end_ms,
            "duration_seconds": (r.end_ms - r.start_ms) / 1000,
            "people": r.people,
            "companies": r.companies,
            "keywords": r.keywords,
            "relevance_score": round(r.relevance_score, 3),
            "confidence": r.confidence,
            "explanation": r.explanation,
            "match_type": r.match_type,
            "timestamp": _format_time(r.start_ms),
            "clip_url": f"/clip/{r.episode_id}?start_ms={r.start_ms}&end_ms={r.end_ms}"
        }
    
    return {
        "query": query,
        "intent": query_understanding.intent,
        "expanded_terms": query_understanding.expanded_terms,
        "results": [result_to_dict(r) for r in results[:limit]],
        "about": [result_to_dict(r) for r in about],
        "mentions": [result_to_dict(r) for r in mentions],
        "related": [result_to_dict(r) for r in related],
        "total": len(results),
        "about_count": len(about),
        "mentions_count": len(mentions),
        "related_count": len(related),
        "search_metadata": {
            "phases_completed": ["understand", "retrieve", "rerank", "assemble"],
            "candidates_retrieved": len(candidates),
            "candidates_reranked": len(ranked)
        }
    }


def _format_time(ms: int) -> str:
    """Format milliseconds as MM:SS."""
    total_seconds = ms // 1000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:02d}"


# =============================================================================
# CLI TESTING
# =============================================================================

if __name__ == "__main__":
    import sys
    
    query = sys.argv[1] if len(sys.argv) > 1 else "currency"
    
    print("=" * 70)
    print(f"LEVEL 3 SEARCH: '{query}'")
    print("=" * 70)
    
    results = search_l3(query, limit=5)
    
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\nIntent: {results['intent']}")
    print(f"Expanded terms: {results['expanded_terms']}")
    print(f"Total results: {results['total']}")
    print(f"  ABOUT: {results['about_count']} | MENTIONS: {results['mentions_count']} | RELATED: {results['related_count']}")
    
    if results['about']:
        print("\n--- ABOUT (topic IS about this) ---")
        for r in results['about']:
            print(f"\n  [{r['confidence'].upper()}] {r['summary'][:70]}...")
            print(f"    Score: {r['relevance_score']} | {r['timestamp']}")
            print(f"    Why: {r['explanation']}")
    
    if results['mentions']:
        print("\n--- MENTIONS (topic mentions this) ---")
        for r in results['mentions']:
            print(f"\n  [{r['confidence'].upper()}] {r['summary'][:70]}...")
            print(f"    Score: {r['relevance_score']} | {r['timestamp']}")
            print(f"    Why: {r['explanation']}")
    
    if results['related']:
        print("\n--- RELATED (semantically related) ---")
        for r in results['related']:
            print(f"\n  [{r['confidence'].upper()}] {r['summary'][:70]}...")
            print(f"    Score: {r['relevance_score']} | {r['timestamp']}")
            print(f"    Why: {r['explanation']}")
