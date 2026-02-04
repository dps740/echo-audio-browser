"""Ingest API - receive transcripts from local clients."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import chromadb

router = APIRouter(prefix="/ingest", tags=["ingest"])


class TranscriptSegment(BaseModel):
    start_ms: int
    end_ms: int
    text: str


class IngestRequest(BaseModel):
    episode_id: str
    episode_title: str
    podcast_title: str
    audio_url: str
    segments: List[TranscriptSegment]
    source: str = "youtube"  # youtube, whisper, manual


@router.post("/episode")
async def ingest_episode(request: IngestRequest):
    """
    Receive transcript data from local ingestion client.
    Chunks into ~60 sec segments and stores in ChromaDB.
    """
    try:
        client = chromadb.PersistentClient(path="./chroma_data")
        collection = client.get_or_create_collection("segments")
        
        # Group transcript segments into ~60 sec chunks
        chunks = []
        current_chunk = {
            "start_ms": None,
            "texts": [],
            "end_ms": 0
        }
        
        for seg in request.segments:
            if current_chunk["start_ms"] is None:
                current_chunk["start_ms"] = seg.start_ms
            
            current_chunk["texts"].append(seg.text)
            current_chunk["end_ms"] = seg.end_ms
            
            # If chunk is ~60+ seconds, finalize it
            duration = current_chunk["end_ms"] - current_chunk["start_ms"]
            if duration >= 60000:  # 60 seconds
                chunks.append({
                    "start_ms": current_chunk["start_ms"],
                    "end_ms": current_chunk["end_ms"],
                    "text": " ".join(current_chunk["texts"])
                })
                current_chunk = {"start_ms": None, "texts": [], "end_ms": 0}
        
        # Don't forget last chunk
        if current_chunk["texts"]:
            chunks.append({
                "start_ms": current_chunk["start_ms"],
                "end_ms": current_chunk["end_ms"],
                "text": " ".join(current_chunk["texts"])
            })
        
        # Store in ChromaDB
        ids = []
        docs = []
        metas = []
        
        for i, chunk in enumerate(chunks):
            chunk_id = f"{request.episode_id}_{i}"
            ids.append(chunk_id)
            docs.append(chunk["text"])
            metas.append({
                "episode_id": request.episode_id,
                "episode_title": request.episode_title,
                "podcast_title": request.podcast_title,
                "audio_url": request.audio_url,
                "start_ms": chunk["start_ms"],
                "end_ms": chunk["end_ms"],
                "summary": chunk["text"][:150] + "...",
                "topic_tags": "",
                "density_score": 0.7,
                "source": request.source,
            })
        
        # Upsert in batches (handles duplicates safely)
        batch_size = 50
        for j in range(0, len(ids), batch_size):
            collection.upsert(
                ids=ids[j:j+batch_size],
                documents=docs[j:j+batch_size],
                metadatas=metas[j:j+batch_size]
            )
        
        return {
            "status": "success",
            "episode_id": request.episode_id,
            "chunks_created": len(chunks),
            "total_duration_ms": chunks[-1]["end_ms"] if chunks else 0
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_stats():
    """Get ingestion stats."""
    client = chromadb.PersistentClient(path="./chroma_data")
    collection = client.get_or_create_collection("segments")
    
    count = collection.count()
    
    # Get unique episodes
    results = collection.get(include=["metadatas"])
    episodes = set()
    for meta in results["metadatas"]:
        episodes.add(meta.get("episode_id", "unknown"))
    
    return {
        "total_segments": count,
        "total_episodes": len(episodes),
        "episodes": list(episodes)
    }
