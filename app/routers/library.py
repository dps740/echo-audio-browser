"""Library browsing endpoints - browse ingested content."""

from fastapi import APIRouter, HTTPException
from typing import List, Optional, Dict
import chromadb
from collections import defaultdict

from app.config import get_embedding_function

router = APIRouter(prefix="/library", tags=["library"])


def _get_collection():
    """Get ChromaDB collection."""
    client = chromadb.PersistentClient(path="./chroma_data")
    embedding_fn = get_embedding_function()
    return client.get_or_create_collection("segments", embedding_function=embedding_fn)


@router.get("/overview")
async def get_overview():
    """Get library overview - podcasts, episode counts, segment counts."""
    collection = _get_collection()
    total = collection.count()
    
    if total == 0:
        return {
            "total_segments": 0,
            "total_episodes": 0,
            "total_podcasts": 0,
            "podcasts": [],
        }
    
    # Fetch all metadata (ChromaDB doesn't support GROUP BY)
    results = collection.get(include=["metadatas"])
    
    podcasts = defaultdict(lambda: {"episodes": set(), "segments": 0, "total_duration_ms": 0})
    
    for meta in results["metadatas"]:
        podcast = meta.get("podcast_title", "Unknown")
        episode = meta.get("episode_id", "unknown")
        duration = meta.get("end_ms", 0) - meta.get("start_ms", 0)
        
        podcasts[podcast]["episodes"].add(episode)
        podcasts[podcast]["segments"] += 1
        podcasts[podcast]["total_duration_ms"] += duration
    
    podcast_list = []
    for name, data in sorted(podcasts.items(), key=lambda x: -x[1]["segments"]):
        podcast_list.append({
            "name": name,
            "episode_count": len(data["episodes"]),
            "segment_count": data["segments"],
            "total_duration_ms": data["total_duration_ms"],
        })
    
    return {
        "total_segments": total,
        "total_episodes": sum(p["episode_count"] for p in podcast_list),
        "total_podcasts": len(podcast_list),
        "podcasts": podcast_list,
    }


@router.get("/podcasts/{podcast_name}/episodes")
async def get_podcast_episodes(podcast_name: str):
    """Get episodes for a specific podcast."""
    collection = _get_collection()
    
    # ChromaDB where filter
    results = collection.get(
        where={"podcast_title": podcast_name},
        include=["metadatas"],
    )
    
    if not results["metadatas"]:
        raise HTTPException(status_code=404, detail=f"No episodes found for: {podcast_name}")
    
    episodes = defaultdict(lambda: {
        "title": "",
        "segments": 0,
        "total_duration_ms": 0,
        "source": "",
    })
    
    for meta in results["metadatas"]:
        ep_id = meta.get("episode_id", "unknown")
        episodes[ep_id]["title"] = meta.get("episode_title", "Unknown")
        episodes[ep_id]["segments"] += 1
        episodes[ep_id]["total_duration_ms"] += meta.get("end_ms", 0) - meta.get("start_ms", 0)
        episodes[ep_id]["source"] = meta.get("source", "unknown")
    
    episode_list = []
    for ep_id, data in sorted(episodes.items(), key=lambda x: -x[1]["total_duration_ms"]):
        episode_list.append({
            "episode_id": ep_id,
            "title": data["title"],
            "segment_count": data["segments"],
            "total_duration_ms": data["total_duration_ms"],
            "source": data["source"],
        })
    
    return {
        "podcast": podcast_name,
        "episode_count": len(episode_list),
        "episodes": episode_list,
    }


@router.get("/episodes/{episode_id}/segments")
async def get_episode_segments(episode_id: str):
    """Get all segments for a specific episode, ordered by time."""
    collection = _get_collection()
    
    results = collection.get(
        where={"episode_id": episode_id},
        include=["metadatas", "documents"],
    )
    
    if not results["metadatas"]:
        raise HTTPException(status_code=404, detail=f"No segments found for episode: {episode_id}")
    
    segments = []
    for i, meta in enumerate(results["metadatas"]):
        segments.append({
            "segment_id": results["ids"][i],
            "episode_id": meta.get("episode_id"),
            "episode_title": meta.get("episode_title", "Unknown"),
            "podcast_title": meta.get("podcast_title", "Unknown"),
            "audio_url": meta.get("audio_url", ""),
            "start_ms": meta.get("start_ms", 0),
            "end_ms": meta.get("end_ms", 0),
            "text": results["documents"][i] if results["documents"] else "",
            "summary": meta.get("summary", ""),
        })
    
    # Sort by start time
    segments.sort(key=lambda s: s["start_ms"])
    
    return {
        "episode_id": episode_id,
        "episode_title": segments[0]["episode_title"] if segments else "Unknown",
        "podcast_title": segments[0]["podcast_title"] if segments else "Unknown",
        "segment_count": len(segments),
        "segments": segments,
    }


@router.delete("/episodes/{episode_id}")
async def delete_episode(episode_id: str):
    """Delete all segments for an episode."""
    collection = _get_collection()
    
    results = collection.get(
        where={"episode_id": episode_id},
        include=["metadatas"],
    )
    
    if not results["ids"]:
        raise HTTPException(status_code=404, detail=f"Episode not found: {episode_id}")
    
    collection.delete(ids=results["ids"])
    
    return {
        "status": "deleted",
        "episode_id": episode_id,
        "segments_removed": len(results["ids"]),
    }


@router.get("/diagnostics")
async def get_diagnostics():
    """
    Get detailed diagnostics about segment quality.
    
    Use this to verify that LLM segmentation is working properly.
    Healthy segments should have:
    - Summaries > 100 chars (detailed descriptions)
    - 2+ tags per segment
    - Specific tags (not generic like "AI" or "technology")
    - Density scores between 0.3-0.9
    """
    collection = _get_collection()
    total = collection.count()
    
    if total == 0:
        return {
            "status": "empty",
            "message": "No segments in database",
            "total_segments": 0,
        }
    
    # Fetch all metadata
    results = collection.get(include=["metadatas", "documents"])
    
    # Analyze segment quality
    summaries = []
    all_tags = []
    tag_counts = []
    densities = []
    summary_lengths = []
    transcript_lengths = []
    
    generic_tags = {"ai", "technology", "business", "science", "philosophy", 
                   "discussion", "conversation", "interview", "podcast", 
                   "machine learning", "tech", "life", "work", "people"}
    
    episodes_seen = set()
    
    for i, meta in enumerate(results["metadatas"]):
        summary = meta.get("summary", "")
        tags_str = meta.get("topic_tags", "")
        tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
        density = meta.get("density_score", 0)
        transcript = results["documents"][i] if results["documents"] else ""
        
        summaries.append(summary)
        summary_lengths.append(len(summary))
        transcript_lengths.append(len(transcript))
        all_tags.extend(tags)
        tag_counts.append(len(tags))
        densities.append(density)
        episodes_seen.add(meta.get("episode_id", "unknown"))
    
    # Calculate quality metrics
    avg_summary_length = sum(summary_lengths) / len(summary_lengths) if summary_lengths else 0
    avg_tags = sum(tag_counts) / len(tag_counts) if tag_counts else 0
    avg_density = sum(densities) / len(densities) if densities else 0
    avg_transcript_length = sum(transcript_lengths) / len(transcript_lengths) if transcript_lengths else 0
    
    # Count problem segments
    short_summaries = sum(1 for l in summary_lengths if l < 100)
    no_tags = sum(1 for c in tag_counts if c == 0)
    generic_only = sum(1 for i, c in enumerate(tag_counts) if c > 0 and all(
        t.lower() in generic_tags for t in (results["metadatas"][i].get("topic_tags", "").split(",") if results["metadatas"][i].get("topic_tags") else [])
    ))
    
    # Unique vs generic tags
    unique_tags = list(set(all_tags))
    specific_tags = [t for t in unique_tags if t.lower() not in generic_tags and len(t.split()) >= 2]
    generic_found = [t for t in unique_tags if t.lower() in generic_tags]
    
    # Quality score (0-100)
    quality_score = 0
    quality_issues = []
    
    # Check summary quality (40 points)
    if avg_summary_length >= 150:
        quality_score += 40
    elif avg_summary_length >= 100:
        quality_score += 25
        quality_issues.append(f"Summaries could be longer (avg {avg_summary_length:.0f} chars, target 150+)")
    else:
        quality_score += 10
        quality_issues.append(f"⚠️ Summaries too short (avg {avg_summary_length:.0f} chars) - LLM segmentation may not be running")
    
    # Check tag quality (30 points)
    if avg_tags >= 2 and len(specific_tags) >= len(unique_tags) * 0.5:
        quality_score += 30
    elif avg_tags >= 1:
        quality_score += 15
        quality_issues.append(f"Tags could be more specific ({len(specific_tags)} specific vs {len(generic_found)} generic)")
    else:
        quality_issues.append(f"⚠️ Missing or generic tags - LLM segmentation may not be running")
    
    # Check density scores (15 points)
    if 0.3 <= avg_density <= 0.8:
        quality_score += 15
    elif avg_density > 0:
        quality_score += 8
    else:
        quality_issues.append("⚠️ No density scores - basic chunking used instead of LLM")
    
    # Check coverage (15 points)
    if short_summaries < total * 0.2:
        quality_score += 15
    elif short_summaries < total * 0.5:
        quality_score += 8
        quality_issues.append(f"{short_summaries}/{total} segments have short summaries (<100 chars)")
    else:
        quality_issues.append(f"⚠️ {short_summaries}/{total} segments have short summaries - likely basic chunking")
    
    # Determine status
    if quality_score >= 80:
        status = "healthy"
        message = "✅ Segments are well-formed with detailed summaries and specific tags"
    elif quality_score >= 50:
        status = "needs_improvement"
        message = "⚠️ Segments exist but quality could be improved"
    else:
        status = "poor"
        message = "❌ Segment quality is poor - likely using basic chunking instead of LLM segmentation"
    
    return {
        "status": status,
        "quality_score": quality_score,
        "message": message,
        "issues": quality_issues,
        "stats": {
            "total_segments": total,
            "total_episodes": len(episodes_seen),
            "avg_summary_length": round(avg_summary_length, 1),
            "avg_transcript_length": round(avg_transcript_length, 1),
            "avg_tags_per_segment": round(avg_tags, 2),
            "avg_density_score": round(avg_density, 3),
            "segments_with_no_tags": no_tags,
            "segments_with_short_summaries": short_summaries,
            "segments_with_only_generic_tags": generic_only,
        },
        "tags": {
            "total_unique": len(unique_tags),
            "specific_tags": specific_tags[:20],
            "generic_tags_found": generic_found,
        },
        "sample_segments": [
            {
                "summary": results["metadatas"][i].get("summary", "")[:200],
                "tags": results["metadatas"][i].get("topic_tags", ""),
                "density": results["metadatas"][i].get("density_score", 0),
                "transcript_preview": (results["documents"][i] if results["documents"] else "")[:150] + "...",
            }
            for i in range(min(3, total))
        ],
        "recommendation": (
            "Run the nuke_and_reingest.py script to re-process all episodes with improved LLM segmentation"
            if quality_score < 50 else
            "Consider re-ingesting episodes with poor segments"
            if quality_score < 80 else
            "Segments look good!"
        ),
    }
