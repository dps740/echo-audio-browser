"""YouTube ingestion endpoints - ingest from the web UI."""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional, Dict
import subprocess
import json
import tempfile
import os
import re
import time
import asyncio
from pathlib import Path
from datetime import datetime

router = APIRouter(prefix="/ingest/youtube", tags=["youtube-ingest"])

# In-memory job tracking
_jobs: Dict[str, dict] = {}
_job_counter = 0

# Curated podcast list (same as batch script)
CURATED_PODCASTS = [
    {"name": "Lex Fridman Podcast", "channel": "@lexfridman", "description": "Deep conversations on AI, science, philosophy"},
    {"name": "The Knowledge Project", "channel": "@tkppodcast", "description": "Mental models, decision making, wisdom"},
    {"name": "Joe Rogan Experience", "channel": "@joerogan", "description": "Long-form conversations, varied topics"},
    {"name": "Huberman Lab", "channel": "@hubermanlab", "description": "Neuroscience, health, performance"},
    {"name": "Tim Ferriss Show", "channel": "@timferriss", "description": "World-class performers, tactics, routines"},
    {"name": "All-In Podcast", "channel": "@allin", "description": "Tech, economics, politics from investors"},
    {"name": "Acquired", "channel": "@AcquiredFM", "description": "Deep dives into great companies"},
    {"name": "Dwarkesh Podcast", "channel": "@DwarkeshPatel", "description": "Deep conversations on progress, AI, history"},
    {"name": "EconTalk", "channel": "@EconTalk", "description": "Economics, philosophy, ideas"},
    {"name": "Invest Like the Best", "channel": "@investlikethebest", "description": "Investing, business, decision making"},
    {"name": "Making Sense (Sam Harris)", "channel": "@samharrisorg", "description": "Neuroscience, philosophy, AI, society"},
    {"name": "Conversations with Tyler", "channel": "@MercatusCenter", "description": "Economics, culture, ideas"},
    {"name": "Peter Attia - The Drive", "channel": "@PeterAttiaMD", "description": "Longevity, health, medicine"},
    {"name": "My First Million", "channel": "@MyFirstMillionPod", "description": "Business ideas, entrepreneurship"},
    {"name": "Founders Podcast", "channel": "@FoundersPodcast", "description": "Biographies of entrepreneurs and innovators"},
    {"name": "Sean Carroll Mindscape", "channel": "@seancarroll", "description": "Physics, philosophy, science"},
]


class IngestURLRequest(BaseModel):
    url: str
    podcast_name: Optional[str] = None


class IngestChannelRequest(BaseModel):
    channel: str
    podcast_name: Optional[str] = None
    limit: int = 10


def _get_collection():
    """Get ChromaDB collection."""
    import chromadb
    client = chromadb.PersistentClient(path="./chroma_data")
    return client.get_or_create_collection("segments")


def _extract_video_id(url: str) -> Optional[str]:
    """Extract YouTube video ID from URL."""
    patterns = [
        r'(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})',
        r'(?:embed/)([a-zA-Z0-9_-]{11})',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def _is_already_ingested(video_id: str) -> bool:
    """Check if video is already in ChromaDB."""
    try:
        collection = _get_collection()
        results = collection.get(
            where={"episode_id": video_id},
            include=["metadatas"],
        )
        return len(results["ids"]) > 0
    except:
        return False


def _get_captions(url: str) -> list:
    """Extract captions with timestamps using yt-dlp."""
    with tempfile.TemporaryDirectory() as tmpdir:
        cmd = [
            "yt-dlp",
            "--write-auto-sub", "--write-sub",
            "--sub-lang", "en",
            "--sub-format", "json3",
            "--skip-download",
            "-o", f"{tmpdir}/subs",
            url
        ]
        subprocess.run(cmd, capture_output=True, timeout=120)
        
        sub_files = list(Path(tmpdir).glob("*.json3"))
        
        if not sub_files:
            # Try VTT fallback
            cmd2 = list(cmd)
            cmd2[cmd2.index("json3")] = "vtt"
            subprocess.run(cmd2, capture_output=True, timeout=120)
            sub_files = list(Path(tmpdir).glob("*.vtt"))
        
        if not sub_files:
            return []
        
        sub_file = sub_files[0]
        
        if sub_file.suffix == ".json3":
            with open(sub_file, encoding='utf-8') as f:
                data = json.load(f)
            
            segments = []
            for event in data.get("events", []):
                if "segs" in event:
                    start_ms = event.get("tStartMs", 0)
                    duration_ms = event.get("dDurationMs", 0)
                    text = "".join(s.get("utf8", "") for s in event["segs"]).strip()
                    text = text.replace("\n", " ")
                    if text and text != " ":
                        segments.append({
                            "start_ms": start_ms,
                            "end_ms": start_ms + duration_ms,
                            "text": text
                        })
            return segments
        else:
            return _parse_vtt(sub_file)


def _parse_vtt(vtt_path) -> list:
    """Parse VTT subtitle file."""
    segments = []
    with open(vtt_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3}) --> (\d{2}:\d{2}:\d{2}\.\d{3})\n(.+?)(?=\n\n|\Z)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    def to_ms(t):
        h, m, rest = t.split(':')
        s, ms = rest.split('.')
        return int(h)*3600000 + int(m)*60000 + int(s)*1000 + int(ms)
    
    for start, end, text in matches:
        clean_text = re.sub(r'<[^>]+>', '', text).strip().replace("\n", " ")
        if clean_text:
            segments.append({
                "start_ms": to_ms(start),
                "end_ms": to_ms(end),
                "text": clean_text
            })
    return segments


def _download_audio(url: str, video_id: str) -> str:
    """Download audio file locally. Returns local URL path."""
    audio_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio")
    os.makedirs(audio_dir, exist_ok=True)
    output_path = os.path.join(audio_dir, f"{video_id}.mp3")
    
    if os.path.exists(output_path):
        return f"/audio/{video_id}.mp3"
    
    cmd = [
        "yt-dlp",
        "-f", "bestaudio",
        "-x", "--audio-format", "mp3",
        "--audio-quality", "5",
        "-o", output_path.replace('.mp3', '.%(ext)s'),
        url
    ]
    subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    
    # yt-dlp may save with different extension
    base = output_path.replace('.mp3', '')
    for ext in ['.mp3', '.m4a', '.opus', '.webm', '.wav']:
        if os.path.exists(base + ext):
            if ext != '.mp3':
                os.rename(base + ext, output_path)
            break
    
    if os.path.exists(output_path):
        return f"/audio/{video_id}.mp3"
    
    # Fallback: stream URL (expires)
    cmd = ["yt-dlp", "-f", "bestaudio", "-g", url]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    return result.stdout.strip() if result.returncode == 0 else url


def _ingest_to_chromadb(episode_id, title, podcast, audio_url, segments, source="youtube"):
    """Store segments in ChromaDB using LLM segmentation and enrichment."""
    import chromadb
    from app.config import get_embedding_function
    from app.services.segmentation import segment_transcript
    from app.services.transcription import TranscriptResult, TranscriptWord
    import openai
    import asyncio
    
    client = chromadb.PersistentClient(path="./chroma_data")
    embedding_fn = get_embedding_function()
    collection = client.get_or_create_collection(
        "segments",
        embedding_function=embedding_fn
    )
    
    # Convert caption segments to TranscriptResult for LLM segmentation
    transcript_words = [
        TranscriptWord(
            word=seg["text"],
            start_ms=seg["start_ms"],
            end_ms=seg["end_ms"],
            confidence=1.0
        )
        for seg in segments
    ]
    
    transcript = TranscriptResult(
        text=" ".join([seg["text"] for seg in segments]),
        words=transcript_words,
        language="en"
    )
    
    # Use LLM segmentation for smart chunking
    try:
        llm_segments = asyncio.run(segment_transcript(transcript, title))
    except Exception as e:
        print(f"LLM segmentation failed: {e}, falling back to simple chunking")
        # Fallback to dumb chunking if LLM fails
        llm_segments = _fallback_chunking(segments)
    
    # Extract key terms for each segment
    ids, docs, metas = [], [], []
    
    for i, seg in enumerate(llm_segments):
        # Extract key terms using GPT-4o-mini
        key_terms = _extract_key_terms(seg.transcript_text, seg.summary)
        
        # Build enriched document for embedding
        enriched_doc = f"""TOPIC: {', '.join(seg.topic_tags)}
SUMMARY: {seg.summary}
KEY TERMS: {', '.join(key_terms)}
TRANSCRIPT: {seg.transcript_text[:2000]}"""
        
        ids.append(f"{episode_id}_{i}")
        docs.append(enriched_doc)
        metas.append({
            "episode_id": episode_id,
            "episode_title": title,
            "podcast_title": podcast,
            "audio_url": audio_url,
            "start_ms": seg.start_ms,
            "end_ms": seg.end_ms,
            "summary": seg.summary,
            "topic_tags": ",".join(seg.topic_tags),
            "key_terms": ",".join(key_terms),
            "density_score": seg.density_score,
            "source": source,
        })
    
    # Batch upsert
    batch_size = 50
    for j in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[j:j+batch_size],
            documents=docs[j:j+batch_size],
            metadatas=metas[j:j+batch_size]
        )
    
    return len(llm_segments)


def _extract_key_terms(transcript: str, summary: str) -> list:
    """Extract key terms and entities from segment using GPT-4o-mini."""
    import openai
    from app.config import settings
    
    if not settings.openai_api_key:
        return []
    
    try:
        client = openai.OpenAI(api_key=settings.openai_api_key)
        
        prompt = f"""Extract 3-7 key terms, entities, and concepts from this podcast segment. 
Focus on: named entities, technical terms, topics discussed, important concepts.
Return as comma-separated list.

Summary: {summary}

Transcript excerpt: {transcript[:1000]}

Key terms:"""
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=100,
        )
        
        terms_text = response.choices[0].message.content.strip()
        # Parse comma-separated terms
        terms = [t.strip() for t in terms_text.split(",") if t.strip()]
        return terms[:10]  # Limit to 10 terms max
        
    except Exception as e:
        print(f"Key term extraction failed: {e}")
        return []


def _fallback_chunking(segments):
    """Fallback to simple chunking if LLM segmentation fails."""
    from app.services.segmentation import SegmentResult
    
    chunks = []
    current = {"start_ms": None, "texts": [], "end_ms": 0}
    
    for seg in segments:
        if current["start_ms"] is None:
            current["start_ms"] = seg["start_ms"]
        current["texts"].append(seg["text"])
        current["end_ms"] = seg["end_ms"]
        
        # Chunk at ~120 seconds (2 minutes) instead of 60
        if current["end_ms"] - current["start_ms"] >= 120000:
            text = " ".join(current["texts"])
            chunks.append(SegmentResult(
                start_ms=current["start_ms"],
                end_ms=current["end_ms"],
                summary=text[:200] + "...",
                topic_tags=["general"],
                density_score=0.5,
                transcript_text=text
            ))
            current = {"start_ms": None, "texts": [], "end_ms": 0}
    
    if current["texts"]:
        text = " ".join(current["texts"])
        chunks.append(SegmentResult(
            start_ms=current["start_ms"],
            end_ms=current["end_ms"],
            summary=text[:200] + "...",
            topic_tags=["general"],
            density_score=0.5,
            transcript_text=text
        ))
    
    return chunks


def _ingest_single_video(job_id: str, url: str, podcast_name: Optional[str] = None):
    """Background: ingest a single YouTube video."""
    try:
        _jobs[job_id]["status"] = "fetching_info"
        
        # Get video info
        cmd = ["yt-dlp", "--dump-json", "--no-download", url]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = f"yt-dlp failed: {result.stderr[:200]}"
            return
        
        info = json.loads(result.stdout)
        video_id = info["id"]
        title = info["title"]
        channel = podcast_name or info.get("channel", info.get("uploader", "Unknown"))
        duration = info.get("duration", 0)
        
        _jobs[job_id]["video_id"] = video_id
        _jobs[job_id]["title"] = title
        _jobs[job_id]["channel"] = channel
        _jobs[job_id]["duration"] = duration
        
        # Check if already ingested
        if _is_already_ingested(video_id):
            _jobs[job_id]["status"] = "skipped"
            _jobs[job_id]["message"] = "Already ingested"
            return
        
        # Get captions
        _jobs[job_id]["status"] = "extracting_captions"
        segments = _get_captions(url)
        if not segments:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = "No captions found"
            return
        
        _jobs[job_id]["caption_segments"] = len(segments)
        
        # Download audio
        _jobs[job_id]["status"] = "downloading_audio"
        audio_url = _download_audio(url, video_id)
        
        # Ingest to ChromaDB
        _jobs[job_id]["status"] = "indexing"
        chunks = _ingest_to_chromadb(video_id, title, channel, audio_url, segments)
        
        _jobs[job_id]["status"] = "complete"
        _jobs[job_id]["chunks_created"] = chunks
        _jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        
    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)[:300]


def _ingest_channel_videos(job_id: str, channel: str, podcast_name: Optional[str], limit: int):
    """Background: ingest multiple videos from a channel."""
    try:
        _jobs[job_id]["status"] = "listing_videos"
        
        # Get video list
        cmd = [
            "yt-dlp", "--flat-playlist", "--dump-json",
            f"https://www.youtube.com/{channel}/videos",
            "--playlist-end", str(limit)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        videos = []
        for line in result.stdout.strip().split('\n'):
            if line:
                try:
                    data = json.loads(line)
                    vid = {
                        "id": data.get("id"),
                        "title": data.get("title", "Unknown"),
                        "url": f"https://www.youtube.com/watch?v={data['id']}",
                        "duration": data.get("duration"),
                    }
                    # Skip short videos (< 5 min)
                    if vid.get("duration") and vid["duration"] < 300:
                        continue
                    videos.append(vid)
                except json.JSONDecodeError:
                    continue
        
        if not videos:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = "No videos found"
            return
        
        _jobs[job_id]["total_videos"] = len(videos)
        _jobs[job_id]["status"] = "ingesting"
        _jobs[job_id]["videos"] = []
        
        name = podcast_name or channel
        success = 0
        skipped = 0
        failed = 0
        
        for i, video in enumerate(videos):
            vid_status = {
                "id": video["id"],
                "title": video["title"],
                "status": "processing",
                "index": i + 1,
            }
            _jobs[job_id]["videos"].append(vid_status)
            _jobs[job_id]["current_video"] = i + 1
            
            try:
                url = video["url"]
                video_id = video["id"]
                
                if _is_already_ingested(video_id):
                    vid_status["status"] = "skipped"
                    skipped += 1
                    continue
                
                segments = _get_captions(url)
                if not segments:
                    vid_status["status"] = "no_captions"
                    failed += 1
                    continue
                
                audio_url = _download_audio(url, video_id)
                chunks = _ingest_to_chromadb(video_id, video["title"], name, audio_url, segments)
                
                vid_status["status"] = "complete"
                vid_status["chunks"] = chunks
                success += 1
                
                time.sleep(1)  # Rate limit
                
            except Exception as e:
                vid_status["status"] = "failed"
                vid_status["error"] = str(e)[:100]
                failed += 1
        
        _jobs[job_id]["status"] = "complete"
        _jobs[job_id]["success"] = success
        _jobs[job_id]["skipped"] = skipped
        _jobs[job_id]["failed"] = failed
        _jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        
    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)[:300]


@router.get("/podcasts")
async def list_curated_podcasts():
    """List curated podcasts available for ingestion."""
    return {"podcasts": CURATED_PODCASTS}


@router.post("/url")
async def ingest_url(request: IngestURLRequest, background_tasks: BackgroundTasks):
    """Ingest a single YouTube URL."""
    global _job_counter
    _job_counter += 1
    job_id = f"url_{_job_counter}_{int(time.time())}"
    
    video_id = _extract_video_id(request.url)
    
    _jobs[job_id] = {
        "job_id": job_id,
        "type": "single",
        "url": request.url,
        "video_id": video_id,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat(),
    }
    
    background_tasks.add_task(_ingest_single_video, job_id, request.url, request.podcast_name)
    
    return {"job_id": job_id, "status": "queued"}


@router.post("/channel")
async def ingest_channel(request: IngestChannelRequest, background_tasks: BackgroundTasks):
    """Batch ingest from a YouTube channel."""
    global _job_counter
    _job_counter += 1
    job_id = f"channel_{_job_counter}_{int(time.time())}"
    
    _jobs[job_id] = {
        "job_id": job_id,
        "type": "channel",
        "channel": request.channel,
        "limit": request.limit,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat(),
    }
    
    background_tasks.add_task(
        _ingest_channel_videos, job_id, request.channel, request.podcast_name, request.limit
    )
    
    return {"job_id": job_id, "status": "queued"}


@router.get("/jobs")
async def list_jobs():
    """List all ingestion jobs."""
    return {
        "jobs": sorted(
            _jobs.values(),
            key=lambda j: j.get("created_at", ""),
            reverse=True,
        )
    }


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get status of a specific ingestion job."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _jobs[job_id]


@router.delete("/jobs/{job_id}")
async def clear_job(job_id: str):
    """Clear a completed job from the list."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    del _jobs[job_id]
    return {"status": "cleared"}


@router.post("/reindex")
async def reindex_all(background_tasks: BackgroundTasks):
    """
    Re-index all existing ChromaDB entries through the new enrichment pipeline.
    This will update embeddings, extract key terms, and build enriched documents.
    """
    global _job_counter
    _job_counter += 1
    job_id = f"reindex_{_job_counter}_{int(time.time())}"
    
    _jobs[job_id] = {
        "job_id": job_id,
        "type": "reindex",
        "status": "queued",
        "created_at": datetime.utcnow().isoformat(),
    }
    
    background_tasks.add_task(_reindex_all_segments, job_id)
    
    return {"job_id": job_id, "status": "queued", "message": "Re-indexing all segments with new pipeline"}


def _reindex_all_segments(job_id: str):
    """Background task: re-process all ChromaDB segments."""
    import chromadb
    from app.config import get_embedding_function
    
    try:
        _jobs[job_id]["status"] = "loading_segments"
        
        client = chromadb.PersistentClient(path="./chroma_data")
        embedding_fn = get_embedding_function()
        
        # Get existing collection
        try:
            old_collection = client.get_collection("segments")
        except:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = "No segments collection found"
            return
        
        # Get all segments
        all_results = old_collection.get(include=["metadatas", "documents"])
        
        if not all_results["ids"]:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = "No segments found"
            return
        
        total_segments = len(all_results["ids"])
        _jobs[job_id]["total_segments"] = total_segments
        _jobs[job_id]["status"] = "reprocessing"
        
        # Group by episode for efficient re-processing
        episodes = {}
        for seg_id, meta, doc in zip(all_results["ids"], all_results["metadatas"], all_results["documents"]):
            episode_id = meta.get("episode_id", "unknown")
            if episode_id not in episodes:
                episodes[episode_id] = []
            episodes[episode_id].append({"id": seg_id, "meta": meta, "doc": doc})
        
        # Create new collection with embedding function
        try:
            client.delete_collection("segments_backup")
        except:
            pass
        
        # Rename old collection as backup
        try:
            # ChromaDB doesn't support rename, so we'll just create new and delete old after
            pass
        except:
            pass
        
        new_collection = client.get_or_create_collection(
            "segments_new",
            embedding_function=embedding_fn
        )
        
        processed = 0
        
        for episode_id, segs in episodes.items():
            try:
                # Re-process each segment with enrichment
                ids, docs, metas = [], [], []
                
                for seg_data in segs:
                    meta = seg_data["meta"]
                    old_doc = seg_data["doc"]
                    
                    # Extract transcript from old doc or metadata
                    transcript = old_doc[:2000] if len(old_doc) > 100 else meta.get("summary", "")
                    summary = meta.get("summary", transcript[:200])
                    
                    # Extract key terms
                    key_terms = _extract_key_terms(transcript, summary)
                    
                    # Get or default topic tags
                    topic_tags = meta.get("topic_tags", "general")
                    if not topic_tags:
                        topic_tags = "general"
                    
                    # Build enriched document
                    enriched_doc = f"""TOPIC: {topic_tags}
SUMMARY: {summary}
KEY TERMS: {', '.join(key_terms)}
TRANSCRIPT: {transcript}"""
                    
                    ids.append(seg_data["id"])
                    docs.append(enriched_doc)
                    
                    # Update metadata
                    updated_meta = meta.copy()
                    updated_meta["key_terms"] = ",".join(key_terms)
                    metas.append(updated_meta)
                
                # Batch upsert
                new_collection.upsert(
                    ids=ids,
                    documents=docs,
                    metadatas=metas
                )
                
                processed += len(segs)
                _jobs[job_id]["processed"] = processed
                
            except Exception as e:
                print(f"Failed to reindex episode {episode_id}: {e}")
                continue
        
        # Replace old collection with new
        _jobs[job_id]["status"] = "finalizing"
        
        try:
            client.delete_collection("segments_backup")
        except:
            pass
        
        # Backup old, swap in new
        try:
            old_collection_data = client.get_collection("segments")
            # Can't rename in ChromaDB, so delete and rename new
            client.delete_collection("segments")
            
            # Get new collection and "rename" by creating final one
            new_data = client.get_collection("segments_new")
            final_collection = client.get_or_create_collection(
                "segments",
                embedding_function=embedding_fn
            )
            
            # Copy all data from segments_new to segments
            all_new = new_collection.get(include=["documents", "metadatas"])
            batch_size = 50
            for i in range(0, len(all_new["ids"]), batch_size):
                final_collection.upsert(
                    ids=all_new["ids"][i:i+batch_size],
                    documents=all_new["documents"][i:i+batch_size],
                    metadatas=all_new["metadatas"][i:i+batch_size]
                )
            
            # Clean up
            client.delete_collection("segments_new")
            
        except Exception as e:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = f"Failed to swap collections: {e}"
            return
        
        _jobs[job_id]["status"] = "complete"
        _jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        _jobs[job_id]["processed"] = processed
        
    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)[:300]
