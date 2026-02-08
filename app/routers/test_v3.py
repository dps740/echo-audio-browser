"""
Test endpoints for V3 segmentation.
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


class SearchHit(BaseModel):
    segment_idx: int
    start_ms: int
    end_ms: int
    duration_s: float
    score: float
    snippet: str
    label: str


class SearchResult(BaseModel):
    query: str
    hits: List[SearchHit]
    total_sentences_searched: int


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
    Search within a V3-segmented episode.
    
    Example: GET /v3/search/gXY1kx7zlkk?q=AI
    """
    if video_id not in _cache:
        raise HTTPException(400, f"Episode not segmented yet. POST /v3/segment/{video_id} first")
    
    data = _cache[video_id]
    sentences = data['sentences']
    segments = data['segments']
    
    # Get query embedding
    query_embedding = get_embeddings([q])[0]
    query_norm = query_embedding / np.linalg.norm(query_embedding)
    
    # Score all sentences
    scores = []
    for i, sent in enumerate(sentences):
        if sent.embedding is not None:
            sent_norm = sent.embedding / np.linalg.norm(sent.embedding)
            sim = np.dot(query_norm, sent_norm)
            scores.append((i, sim, sent))
        
        # Also boost for keyword match
        if q.lower() in sent.text.lower():
            # Find existing score and boost it
            for j, (idx, score, s) in enumerate(scores):
                if idx == i:
                    scores[j] = (idx, score + 0.2, s)  # Boost
                    break
    
    # Sort by score
    scores.sort(key=lambda x: -x[1])
    
    # Get top hits and map to segments
    hits = []
    seen_segments = set()
    
    for sent_idx, score, sent in scores[:top_k * 3]:  # Get more candidates
        # Find which segment this sentence belongs to
        for seg_idx, seg in enumerate(segments):
            if seg.start_ms <= sent.start_ms <= seg.end_ms:
                if seg_idx not in seen_segments:
                    seen_segments.add(seg_idx)
                    hits.append(SearchHit(
                        segment_idx=seg_idx,
                        start_ms=seg.start_ms,
                        end_ms=seg.end_ms,
                        duration_s=round(seg.duration_s, 1),
                        score=round(float(score), 3),
                        snippet=sent.text[:150] + "..." if len(sent.text) > 150 else sent.text,
                        label=seg.label or f"Segment {seg_idx + 1}"
                    ))
                break
        
        if len(hits) >= top_k:
            break
    
    return SearchResult(
        query=q,
        hits=hits,
        total_sentences_searched=len(sentences)
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
