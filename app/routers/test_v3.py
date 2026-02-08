"""
Test endpoints for V3 segmentation.

V3 uses:
- Sentence-level indexing (99% coverage vs 49% with V2)
- Cluster-based adaptive clip length
- No fixed segments - clips sized by query specificity
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from pathlib import Path
import numpy as np

from app.services.segmentation_v3 import (
    segment_transcript_v3,
    get_embeddings,
    SegmentV3,
    Sentence
)
from app.services.search_v3 import (
    search_with_clusters,
    format_cluster_for_display,
    MatchCluster
)
from app.config import settings

router = APIRouter(prefix="/v3", tags=["v3-test"])


class SegmentResponse(BaseModel):
    start_ms: int
    end_ms: int
    duration_s: float
    text: str
    label: str


class SegmentationResult(BaseModel):
    episode: str
    total_sentences: int
    total_segments: int
    coverage_pct: float
    segments: List[SegmentResponse]
    stats: dict


class ClusterHit(BaseModel):
    """A search result cluster - adaptive clip length based on query."""
    start_ms: int
    end_ms: int
    start_formatted: str
    end_formatted: str
    duration_s: float
    score: float
    match_count: int
    snippet: str
    clip_url: Optional[str] = None  # Generated MP3 clip URL


class SearchResult(BaseModel):
    query: str
    clusters: List[ClusterHit]
    total_sentences: int
    total_matches: int
    note: str


# Global cache for testing
_cache = {}


@router.post("/segment/{video_id}", response_model=SegmentationResult)
async def segment_episode(video_id: str):
    """
    Run V3 segmentation on a VTT file.
    
    Example: POST /v3/segment/gXY1kx7zlkk
    """
    # Find VTT file
    vtt_path = Path(f"audio/{video_id}.en.vtt")
    if not vtt_path.exists():
        raise HTTPException(404, f"VTT file not found: {vtt_path}")
    
    vtt_content = vtt_path.read_text()
    
    # Run segmentation
    segments, sentences = segment_transcript_v3(vtt_content)
    
    if not sentences:
        raise HTTPException(400, "No sentences extracted from VTT")
    
    # Cache for search
    _cache[video_id] = {
        'segments': segments,
        'sentences': sentences
    }
    
    # Calculate stats
    total_duration_ms = sentences[-1].end_ms
    covered_ms = sum(s.end_ms - s.start_ms for s in segments)
    durations = [s.duration_s for s in segments]
    
    return SegmentationResult(
        episode=video_id,
        total_sentences=len(sentences),
        total_segments=len(segments),
        coverage_pct=round(covered_ms / total_duration_ms * 100, 1),
        segments=[
            SegmentResponse(
                start_ms=s.start_ms,
                end_ms=s.end_ms,
                duration_s=round(s.duration_s, 1),
                text=s.text[:200] + "..." if len(s.text) > 200 else s.text,
                label=s.label or f"Segment {i+1}"
            )
            for i, s in enumerate(segments)
        ],
        stats={
            "mean_duration_s": round(sum(durations) / len(durations), 1),
            "min_duration_s": round(min(durations), 1),
            "max_duration_s": round(max(durations), 1),
            "too_short": sum(1 for d in durations if d < 15),
            "too_long": sum(1 for d in durations if d > 180)
        }
    )


@router.get("/search/{video_id}", response_model=SearchResult)
async def search_episode(video_id: str, q: str, top_k: int = 5):
    """
    Search within a V3-segmented episode using cluster-based adaptive clips.
    
    The clip length adapts to query specificity:
    - Broad query ("AI") → longer clips where topic is discussed
    - Specific query ("Chinese AI infrastructure") → shorter, focused clips
    
    Example: GET /v3/search/gXY1kx7zlkk?q=AI
    """
    if video_id not in _cache:
        raise HTTPException(400, f"Episode not segmented yet. POST /v3/segment/{video_id} first")
    
    data = _cache[video_id]
    sentences = data['sentences']
    
    # Use cluster-based search
    clusters, all_matches = search_with_clusters(
        query=q,
        sentences=sentences,
        top_k_sentences=50,
        gap_threshold_ms=60_000,  # 60s gap = new cluster
        max_clusters=top_k
    )
    
    # Format results
    def format_time(ms: int) -> str:
        total_seconds = ms // 1000
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes}:{seconds:02d}"
    
    # Generate clips and format results
    from app.services.clip_extractor import get_clip_url
    
    cluster_hits = []
    for c in clusters:
        # Generate clip from WAV source
        clip_url = get_clip_url(video_id, c.start_ms, c.end_ms)
        
        cluster_hits.append(ClusterHit(
            start_ms=c.start_ms,
            end_ms=c.end_ms,
            start_formatted=format_time(c.start_ms),
            end_formatted=format_time(c.end_ms),
            duration_s=round(c.duration_s, 1),
            score=round(c.best_score, 3),
            match_count=len(c.matches),
            snippet=c.snippet[:200] + "..." if len(c.snippet) > 200 else c.snippet,
            clip_url=clip_url
        ))
    
    # Generate helpful note about clip lengths
    if cluster_hits:
        avg_duration = sum(c.duration_s for c in cluster_hits) / len(cluster_hits)
        if avg_duration > 300:
            note = "Broad topic - multiple long discussions found"
        elif avg_duration > 120:
            note = "Medium topic - several focused discussions found"
        else:
            note = "Specific topic - short, focused clips found"
    else:
        note = "No matches found"
    
    return SearchResult(
        query=q,
        clusters=cluster_hits,
        total_sentences=len(sentences),
        total_matches=len(all_matches),
        note=note
    )


@router.get("/compare/{video_id}")
async def compare_approaches(video_id: str):
    """
    Compare V3 segmentation with current V2 approach.
    
    Example: GET /v3/compare/gXY1kx7zlkk
    """
    import chromadb
    
    # Get V3 segments
    if video_id not in _cache:
        # Run segmentation first
        vtt_path = Path(f"audio/{video_id}.en.vtt")
        if not vtt_path.exists():
            raise HTTPException(404, f"VTT file not found")
        
        segments, sentences = segment_transcript_v3(vtt_path.read_text())
        _cache[video_id] = {'segments': segments, 'sentences': sentences}
    
    v3_data = _cache[video_id]
    v3_segments = v3_data['segments']
    v3_sentences = v3_data['sentences']
    
    # Get current segments from ChromaDB
    client = chromadb.PersistentClient(path='chroma_data')
    col = client.get_collection('segments')
    results = col.get(include=['metadatas'])
    
    current_segs = []
    for meta in results['metadatas']:
        ep = meta.get('episode_title', '')
        if 'ICE' in ep and video_id == 'gXY1kx7zlkk':
            current_segs.append(meta)
        elif 'Oz' in ep and video_id == 'b5p40OuTTW4':
            current_segs.append(meta)
    
    total_duration_ms = v3_sentences[-1].end_ms
    
    # V3 stats
    v3_covered = sum(s.end_ms - s.start_ms for s in v3_segments)
    v3_durations = [s.duration_s for s in v3_segments]
    
    # Current stats
    curr_covered = sum((m.get('end_ms', 0) or 0) - (m.get('start_ms', 0) or 0) for m in current_segs)
    curr_durations = [((m.get('end_ms', 0) or 0) - (m.get('start_ms', 0) or 0)) / 1000 for m in current_segs if m.get('end_ms', 0)]
    
    return {
        "video_id": video_id,
        "total_duration_min": round(total_duration_ms / 60000, 1),
        "v3": {
            "segments": len(v3_segments),
            "coverage_pct": round(v3_covered / total_duration_ms * 100, 1),
            "mean_duration_s": round(sum(v3_durations) / len(v3_durations), 1) if v3_durations else 0,
            "min_duration_s": round(min(v3_durations), 1) if v3_durations else 0,
            "max_duration_s": round(max(v3_durations), 1) if v3_durations else 0,
        },
        "current": {
            "segments": len(current_segs),
            "coverage_pct": round(curr_covered / total_duration_ms * 100, 1) if total_duration_ms else 0,
            "mean_duration_s": round(sum(curr_durations) / len(curr_durations), 1) if curr_durations else 0,
            "min_duration_s": round(min(curr_durations), 1) if curr_durations else 0,
            "max_duration_s": round(max(curr_durations), 1) if curr_durations else 0,
        },
        "winner": {
            "coverage": "V3" if v3_covered > curr_covered else "Current",
            "coverage_delta_pct": round((v3_covered - curr_covered) / total_duration_ms * 100, 1)
        }
    }


@router.get("/clip/{video_id}")
async def get_clip(video_id: str, start_ms: int, end_ms: int):
    """
    Generate and return a clip URL.
    
    Extracts segment from WAV source, converts to MP3.
    
    Example: GET /v3/clip/gXY1kx7zlkk?start_ms=60000&end_ms=120000
    """
    from app.services.clip_extractor import get_clip_url
    
    clip_url = get_clip_url(video_id, start_ms, end_ms)
    
    if clip_url:
        return {
            "video_id": video_id,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "duration_s": (end_ms - start_ms) / 1000,
            "clip_url": clip_url
        }
    else:
        raise HTTPException(500, f"Failed to generate clip for {video_id}")


# ============================================================================
# LLM-REFINED SEGMENTATION (V3.1)
# ============================================================================

class RefinedSegmentResponse(BaseModel):
    start_ms: int
    end_ms: int
    original_start_ms: int
    duration_s: float
    snippet: str  # LLM-generated specific summary
    boundary_refined: bool  # Was the start moved?


class RefinedSegmentationResult(BaseModel):
    episode: str
    total_sentences: int
    total_segments: int
    segments: List[RefinedSegmentResponse]
    llm_model: str
    cost_estimate: str


@router.post("/segment-refined/{video_id}", response_model=RefinedSegmentationResult)
async def segment_episode_refined(video_id: str, similarity_threshold: float = 0.5):
    """
    Run V3 segmentation with LLM refinement.
    
    This is the full pipeline:
    1. Parse VTT → sentences with timestamps
    2. Embed sentences
    3. Detect topic boundaries (embedding similarity drops)
    4. LLM refines each boundary (finds true topic start)
    5. LLM generates specific snippet for each segment
    
    Run this ONCE per episode at index time. Results are cached.
    
    Example: POST /v3/segment-refined/gXY1kx7zlkk
    """
    from app.services.segment_refiner import (
        detect_topic_boundaries,
        refine_segments
    )
    
    # Find VTT file
    vtt_path = Path(f"audio/{video_id}.en.vtt")
    if not vtt_path.exists():
        raise HTTPException(404, f"VTT file not found: {vtt_path}")
    
    vtt_content = vtt_path.read_text()
    
    # Step 1-2: Parse VTT and embed sentences
    segments, sentences = segment_transcript_v3(vtt_content)
    
    if not sentences:
        raise HTTPException(400, "No sentences extracted from VTT")
    
    # Step 3: Detect topic boundaries
    initial_boundaries = detect_topic_boundaries(
        sentences, 
        similarity_threshold=similarity_threshold
    )
    
    print(f"Found {len(initial_boundaries)} initial segments")
    
    # Step 4-5: LLM refinement
    refined = refine_segments(sentences, initial_boundaries, model="gpt-4o-mini")
    
    # Cache refined segments
    _cache[video_id] = {
        'segments': segments,
        'sentences': sentences,
        'refined_segments': refined
    }
    
    # Calculate cost estimate
    # ~700 tokens input + 50 output per segment, gpt-4o-mini pricing
    cost_per_segment = 0.00015
    total_cost = len(refined) * cost_per_segment
    
    return RefinedSegmentationResult(
        episode=video_id,
        total_sentences=len(sentences),
        total_segments=len(refined),
        segments=[
            RefinedSegmentResponse(
                start_ms=s.start_ms,
                end_ms=s.end_ms,
                original_start_ms=s.original_start_ms,
                duration_s=round((s.end_ms - s.start_ms) / 1000, 1),
                snippet=s.snippet,
                boundary_refined=(s.start_ms != s.original_start_ms)
            )
            for s in refined
        ],
        llm_model="gpt-4o-mini",
        cost_estimate=f"${total_cost:.4f}"
    )


@router.get("/search-refined/{video_id}")
async def search_refined(video_id: str, q: str, top_k: int = 5):
    """
    Search using LLM-refined segments.
    
    Uses pre-computed snippets - no LLM calls at search time.
    
    Example: GET /v3/search-refined/gXY1kx7zlkk?q=AI
    """
    if video_id not in _cache:
        raise HTTPException(400, f"Episode not segmented yet. POST /v3/segment-refined/{video_id} first")
    
    data = _cache[video_id]
    
    if 'refined_segments' not in data:
        raise HTTPException(400, f"Episode not refined yet. POST /v3/segment-refined/{video_id} first")
    
    refined = data['refined_segments']
    sentences = data['sentences']
    
    # Search within refined segments
    from app.services.search_v3 import get_query_embedding
    import numpy as np
    
    query_emb = get_query_embedding(q)
    query_norm = query_emb / np.linalg.norm(query_emb)
    
    # Score each segment by match count AND best score
    # Requires multiple strong matches, not just one passing mention
    scored_segments = []
    MATCH_THRESHOLD = 0.35  # Minimum score for a sentence to count as a match
    
    for seg in refined:
        best_score = 0
        match_count = 0
        for idx in seg.sentence_indices:
            sent = sentences[idx]
            if sent.embedding is not None:
                sent_norm = sent.embedding / np.linalg.norm(sent.embedding)
                score = float(np.dot(query_norm, sent_norm))
                best_score = max(best_score, score)
                if score >= MATCH_THRESHOLD:
                    match_count += 1
        
        # Combined score: best match + density bonus
        # Segments with multiple matches rank higher
        density_bonus = min(match_count / 5, 0.3)  # Up to 0.3 bonus for 5+ matches
        combined_score = best_score + density_bonus
        
        # Require at least 2 sentence matches to avoid passing mentions
        # Single mentions even with high score are often not the main topic
        if match_count >= 2:
            scored_segments.append((seg, combined_score, match_count))
    
    # Sort by combined score, return top_k
    scored_segments.sort(key=lambda x: -x[1])
    top_segments = scored_segments[:top_k]
    
    # Generate clip URLs
    from app.services.clip_extractor import get_clip_url
    
    def format_time(ms: int) -> str:
        total_seconds = ms // 1000
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes}:{seconds:02d}"
    
    results = []
    for seg, score, match_count in top_segments:
        clip_url = get_clip_url(video_id, seg.start_ms, seg.end_ms)
        
        results.append({
            "start_ms": seg.start_ms,
            "end_ms": seg.end_ms,
            "start_formatted": format_time(seg.start_ms),
            "end_formatted": format_time(seg.end_ms),
            "duration_s": round((seg.end_ms - seg.start_ms) / 1000, 1),
            "score": round(score, 3),
            "match_count": match_count,  # How many sentences matched
            "snippet": seg.snippet,
            "clip_url": clip_url,
            "boundary_refined": seg.start_ms != seg.original_start_ms
        })
    
    return {
        "query": q,
        "results": results,
        "total_segments": len(refined),
        "matches_found": len(results)
    }
