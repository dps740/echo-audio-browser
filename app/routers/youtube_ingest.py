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
    use_whisper: bool = False
    whisper_model: str = "tiny"


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


def _segments_to_words(segments: list) -> list:
    """
    Convert VTT subtitle segments into individual word-level tokens.
    
    VTT files have phrase-level timing, but the segmentation module needs
    word-level tokens for accurate quote anchor matching.
    """
    from app.services.transcription import TranscriptWord
    
    words = []
    for seg in segments:
        # Clean up text - remove HTML entities and speaker markers
        text = seg["text"]
        text = text.replace("&gt;", ">").replace("&lt;", "<")
        # Remove >> speaker markers for cleaner word extraction
        text = text.replace(">>", "").replace("> >", "")
        
        # Split into individual words
        seg_words = text.split()
        if not seg_words:
            continue
        
        # Distribute time evenly among words in this segment
        duration = seg["end_ms"] - seg["start_ms"]
        word_duration = duration / len(seg_words) if seg_words else duration
        
        for i, word in enumerate(seg_words):
            word_start = seg["start_ms"] + int(i * word_duration)
            word_end = seg["start_ms"] + int((i + 1) * word_duration)
            words.append(TranscriptWord(
                word=word,
                start_ms=word_start,
                end_ms=word_end,
                confidence=1.0
            ))
    
    return words


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
    
    # Convert caption segments to individual word-level TranscriptWords
    # This enables accurate quote anchor matching in segmentation
    transcript_words = _segments_to_words(segments)
    
    # Calculate total duration from segments
    total_duration_ms = max([seg["end_ms"] for seg in segments]) if segments else 0
    
    transcript = TranscriptResult(
        text=" ".join([seg["text"] for seg in segments]),
        words=transcript_words,
        duration_ms=total_duration_ms,
        speakers=[]
    )
    
    # Use LLM segmentation for smart chunking
    try:
        print(f"[DEBUG] Starting LLM segmentation for: {title}")
        print(f"[DEBUG] Transcript has {len(transcript.words)} words, {transcript.duration_ms}ms duration")
        llm_segments = asyncio.run(segment_transcript(transcript, title))
        print(f"[DEBUG] LLM segmentation returned {len(llm_segments)} segments")
    except Exception as e:
        import traceback
        print(f"LLM segmentation failed: {e}")
        print(f"[DEBUG] Full traceback:\n{traceback.format_exc()}")
        print("Falling back to simple chunking")
        # Fallback to dumb chunking if LLM fails
        llm_segments = _fallback_chunking(segments)
    
    # Extract key terms for each segment
    ids, docs, metas = [], [], []
    
    def truncate_at_sentence(text: str, max_chars: int = 2000) -> str:
        """Truncate text at sentence boundary, not mid-word."""
        if len(text) <= max_chars:
            return text
        # Find last sentence boundary before max_chars
        truncated = text[:max_chars]
        # Look for last sentence-ending punctuation
        for end_char in ['. ', '! ', '? ', '.\n', '!\n', '?\n']:
            last_end = truncated.rfind(end_char)
            if last_end > max_chars * 0.5:  # Only if we keep at least 50%
                return truncated[:last_end + 1].strip()
        # Fallback: find last space to avoid cutting mid-word
        last_space = truncated.rfind(' ')
        if last_space > max_chars * 0.7:
            return truncated[:last_space].strip() + '...'
        return truncated.strip() + '...'
    
    for i, seg in enumerate(llm_segments):
        # Extract key terms using GPT-4o-mini
        key_terms = _extract_key_terms(seg.transcript_text, seg.summary)
        
        # Build enriched document for embedding (truncate at sentence boundary)
        enriched_doc = f"""TOPIC: {', '.join(seg.topic_tags)}
SUMMARY: {seg.summary}
KEY TERMS: {', '.join(key_terms)}
TRANSCRIPT: {truncate_at_sentence(seg.transcript_text, 2000)}"""
        
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


@router.post("/deep-reindex")
async def deep_reindex_all(background_tasks: BackgroundTasks):
    """
    Deep re-analyze: re-runs full LLM segmentation pipeline on all content.
    This creates NEW segment boundaries (not just re-embedding existing ones).
    
    Pipeline per episode:
    1. Reconstruct transcript from existing segments
    2. Re-run LLM segmentation (new boundaries, summaries, topic tags, density)
    3. Extract key terms per segment
    4. Build enriched documents + re-embed
    """
    global _job_counter
    _job_counter += 1
    job_id = f"deep_reindex_{_job_counter}_{int(time.time())}"
    
    _jobs[job_id] = {
        "job_id": job_id,
        "type": "reindex",
        "status": "queued",
        "created_at": datetime.utcnow().isoformat(),
        "deep": True,
    }
    
    background_tasks.add_task(_deep_reindex_all_segments, job_id)
    
    return {"job_id": job_id, "status": "queued", "message": "Deep re-analysis: full LLM re-segmentation of all content"}


def _deep_reindex_all_segments(job_id: str):
    """Background task: full LLM re-segmentation of all content."""
    import chromadb
    from app.config import get_embedding_function
    from app.services.segmentation import segment_transcript, SegmentResult
    from app.services.transcription import TranscriptResult, TranscriptWord
    
    try:
        _jobs[job_id]["status"] = "loading_segments"
        
        client = chromadb.PersistentClient(path="./chroma_data")
        embedding_fn = get_embedding_function()
        
        collection = client.get_or_create_collection(
            "segments",
            embedding_function=embedding_fn
        )
        
        # Get all segments
        all_results = collection.get(include=["metadatas", "documents"])
        
        if not all_results["ids"]:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = "No segments found"
            return
        
        # Group by episode
        episodes = {}
        for seg_id, meta, doc in zip(all_results["ids"], all_results["metadatas"], all_results["documents"]):
            episode_id = meta.get("episode_id", "unknown")
            if episode_id not in episodes:
                episodes[episode_id] = {
                    "title": meta.get("episode_title", "Unknown"),
                    "podcast": meta.get("podcast_title", "Unknown"),
                    "audio_url": meta.get("audio_url", ""),
                    "source": meta.get("source", "youtube"),
                    "segments": []
                }
            episodes[episode_id]["segments"].append({
                "id": seg_id,
                "meta": meta,
                "doc": doc,
                "start_ms": meta.get("start_ms", 0),
                "end_ms": meta.get("end_ms", 0),
            })
        
        total_episodes = len(episodes)
        total_segments = len(all_results["ids"])
        _jobs[job_id]["total_segments"] = total_segments
        _jobs[job_id]["total_episodes"] = total_episodes
        _jobs[job_id]["status"] = "reprocessing"
        
        processed_episodes = 0
        processed_segments = 0
        
        for episode_id, ep_data in episodes.items():
            try:
                # Sort segments by start time
                ep_data["segments"].sort(key=lambda s: s["start_ms"])
                
                # Reconstruct transcript from existing segments
                transcript_words = []
                for seg in ep_data["segments"]:
                    doc = seg["doc"]
                    
                    # Extract transcript text from enriched format or raw
                    if "TRANSCRIPT:" in doc:
                        text = doc.split("TRANSCRIPT:")[-1].strip()
                    else:
                        text = doc
                    
                    # Create word entries (each segment becomes a "word" with timestamps)
                    # Split into ~30-second chunks for better LLM granularity
                    seg_duration = seg["end_ms"] - seg["start_ms"]
                    words_in_seg = text.split()
                    
                    if not words_in_seg:
                        continue
                    
                    # Split text into chunks of ~100 words (~30 seconds of speech)
                    chunk_size = 100
                    chunks = []
                    for ci in range(0, len(words_in_seg), chunk_size):
                        chunk_words = words_in_seg[ci:ci + chunk_size]
                        chunks.append(" ".join(chunk_words))
                    
                    # Distribute timestamps across chunks
                    chunk_duration = seg_duration / len(chunks) if chunks else seg_duration
                    for ci, chunk in enumerate(chunks):
                        chunk_start = seg["start_ms"] + int(ci * chunk_duration)
                        chunk_end = seg["start_ms"] + int((ci + 1) * chunk_duration)
                        transcript_words.append(TranscriptWord(
                            word=chunk,
                            start_ms=chunk_start,
                            end_ms=chunk_end,
                            confidence=1.0
                        ))
                
                if not transcript_words:
                    continue
                
                full_text = " ".join(w.word for w in transcript_words)
                transcript = TranscriptResult(
                    text=full_text,
                    words=transcript_words,
                    duration_ms=transcript_words[-1].end_ms if transcript_words else 0,
                    speakers=[]
                )
                
                # Run LLM segmentation
                import asyncio
                try:
                    new_segments = asyncio.run(segment_transcript(transcript, ep_data["title"]))
                except Exception as e:
                    print(f"LLM segmentation failed for {episode_id}: {e}, skipping")
                    continue
                
                # Delete old segments for this episode
                old_ids = [s["id"] for s in ep_data["segments"]]
                collection.delete(ids=old_ids)
                
                # Create new segments with enrichment
                ids, docs, metas = [], [], []
                
                for i, seg in enumerate(new_segments):
                    key_terms = _extract_key_terms(seg.transcript_text, seg.summary)
                    
                    enriched_doc = f"""TOPIC: {', '.join(seg.topic_tags)}
SUMMARY: {seg.summary}
KEY TERMS: {', '.join(key_terms)}
TRANSCRIPT: {seg.transcript_text[:2000]}"""
                    
                    ids.append(f"{episode_id}_{i}")
                    docs.append(enriched_doc)
                    metas.append({
                        "episode_id": episode_id,
                        "episode_title": ep_data["title"],
                        "podcast_title": ep_data["podcast"],
                        "audio_url": ep_data["audio_url"],
                        "start_ms": seg.start_ms,
                        "end_ms": seg.end_ms,
                        "summary": seg.summary,
                        "topic_tags": ",".join(seg.topic_tags),
                        "key_terms": ",".join(key_terms),
                        "density_score": seg.density_score,
                        "source": ep_data["source"],
                    })
                
                # Upsert new segments
                batch_size = 50
                for j in range(0, len(ids), batch_size):
                    collection.upsert(
                        ids=ids[j:j+batch_size],
                        documents=docs[j:j+batch_size],
                        metadatas=metas[j:j+batch_size]
                    )
                
                processed_episodes += 1
                processed_segments += len(new_segments)
                _jobs[job_id]["processed"] = processed_segments
                _jobs[job_id]["processed_episodes"] = processed_episodes
                
                time.sleep(1)  # Rate limit between episodes
                
            except Exception as e:
                print(f"Failed to deep reindex episode {episode_id}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        _jobs[job_id]["status"] = "complete"
        _jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        _jobs[job_id]["processed"] = processed_segments
        _jobs[job_id]["processed_episodes"] = processed_episodes
        _jobs[job_id]["old_segments"] = total_segments
        _jobs[job_id]["new_segments"] = processed_segments
        
    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)[:300]


@router.post("/scan-local")
async def scan_local_audio(background_tasks: BackgroundTasks):
    """
    Scan local audio folder for existing MP3s and re-ingest them.
    Fetches captions from YouTube without re-downloading audio.
    """
    global _job_counter
    _job_counter += 1
    job_id = f"scan-{_job_counter}"
    
    audio_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio")
    
    if not os.path.exists(audio_dir):
        raise HTTPException(status_code=404, detail=f"Audio directory not found: {audio_dir}")
    
    # Find all MP3 files
    mp3_files = [f for f in os.listdir(audio_dir) if f.endswith('.mp3')]
    
    if not mp3_files:
        raise HTTPException(status_code=404, detail="No MP3 files found in audio directory")
    
    _jobs[job_id] = {
        "id": job_id,
        "type": "scan-local",
        "status": "running",
        "started_at": datetime.utcnow().isoformat(),
        "total_files": len(mp3_files),
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "errors": []
    }
    
    background_tasks.add_task(_scan_local_audio, job_id, audio_dir, mp3_files)
    
    return {"job_id": job_id, "message": f"Scanning {len(mp3_files)} local audio files", "files": mp3_files}


def _scan_local_audio(job_id: str, audio_dir: str, mp3_files: list):
    """Background task to process local audio files."""
    import chromadb
    from app.config import get_embedding_function
    
    for i, filename in enumerate(mp3_files):
        try:
            # Extract video ID from filename (e.g., "dQw4w9WgXcQ.mp3" -> "dQw4w9WgXcQ")
            video_id = filename.replace('.mp3', '')
            
            _jobs[job_id]["current_file"] = filename
            _jobs[job_id]["progress"] = f"{i+1}/{len(mp3_files)}"
            
            # Check if already ingested
            if _is_already_ingested(video_id):
                _jobs[job_id]["skipped"] += 1
                continue
            
            # Fetch video metadata from YouTube
            url = f"https://www.youtube.com/watch?v={video_id}"
            cmd = ["yt-dlp", "--dump-json", "--no-download", url]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                _jobs[job_id]["failed"] += 1
                _jobs[job_id]["errors"].append(f"{filename}: Could not fetch metadata")
                continue
            
            metadata = json.loads(result.stdout)
            title = metadata.get("title", video_id)
            channel = metadata.get("channel", "Unknown")
            
            # Fetch captions
            captions = _get_captions(url)
            
            if not captions:
                _jobs[job_id]["failed"] += 1
                _jobs[job_id]["errors"].append(f"{filename}: No captions available")
                continue
            
            # Audio URL is local
            audio_url = f"/audio/{filename}"
            
            # Run full ingestion pipeline (segmentation + embedding)
            _ingest_to_chromadb(
                episode_id=video_id,
                title=title,
                podcast=channel,
                audio_url=audio_url,
                segments=captions,
                source="youtube"
            )
            
            _jobs[job_id]["processed"] += 1
            
        except Exception as e:
            _jobs[job_id]["failed"] += 1
            _jobs[job_id]["errors"].append(f"{filename}: {str(e)[:100]}")
    
    _jobs[job_id]["status"] = "complete"
    _jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()


@router.post("/repair")
async def repair_collection():
    """
    Repair ChromaDB after a failed reindex swap.
    If segments_new exists but segments is empty/missing, restore from segments_new.
    """
    import chromadb
    from app.config import get_embedding_function
    
    embedding_fn = get_embedding_function()
    client = chromadb.PersistentClient(path="./chroma_data")
    
    report = {"actions": []}
    
    # Check current state
    try:
        segments = client.get_or_create_collection("segments", embedding_function=embedding_fn)
        seg_count = segments.count()
        report["segments_count"] = seg_count
    except Exception as e:
        report["segments_error"] = str(e)
        seg_count = 0
    
    # Check for leftover segments_new
    try:
        segments_new = client.get_collection("segments_new")
        new_count = segments_new.count()
        report["segments_new_count"] = new_count
        
        # If segments is empty but segments_new has data, restore
        if seg_count == 0 and new_count > 0:
            report["actions"].append(f"Restoring {new_count} segments from segments_new")
            all_data = segments_new.get(include=["documents", "metadatas"])
            batch_size = 50
            for i in range(0, len(all_data["ids"]), batch_size):
                segments.upsert(
                    ids=all_data["ids"][i:i+batch_size],
                    documents=all_data["documents"][i:i+batch_size],
                    metadatas=all_data["metadatas"][i:i+batch_size]
                )
            report["actions"].append(f"Restored {new_count} segments")
        
        # Clean up segments_new
        client.delete_collection("segments_new")
        report["actions"].append("Deleted segments_new")
        
    except Exception:
        report["segments_new"] = "not found (ok)"
    
    # Clean up segments_backup
    try:
        client.delete_collection("segments_backup")
        report["actions"].append("Deleted segments_backup")
    except Exception:
        pass
    
    # Final count
    try:
        segments = client.get_or_create_collection("segments", embedding_function=embedding_fn)
        report["final_segments_count"] = segments.count()
    except Exception as e:
        report["final_segments_count"] = f"error: {e}"
    
    return report


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
    """Background task: re-process all ChromaDB segments in place.
    
    Updates documents and metadata directly in the existing collection.
    No fragile delete/swap needed — ChromaDB upsert recomputes embeddings
    when documents change.
    """
    import chromadb
    from app.config import get_embedding_function
    
    try:
        _jobs[job_id]["status"] = "loading_segments"
        
        client = chromadb.PersistentClient(path="./chroma_data")
        embedding_fn = get_embedding_function()
        
        # Get existing collection with proper embedding function
        collection = client.get_or_create_collection(
            "segments",
            embedding_function=embedding_fn
        )
        
        count = collection.count()
        if count == 0:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = "No segments found"
            return
        
        # Get all segments
        all_results = collection.get(include=["metadatas", "documents"])
        
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
        
        processed = 0
        
        for episode_id, segs in episodes.items():
            try:
                ids, docs, metas = [], [], []
                
                for seg_data in segs:
                    meta = seg_data["meta"]
                    old_doc = seg_data["doc"]
                    
                    # Extract transcript from old doc or metadata
                    # If old_doc has the enriched format, extract TRANSCRIPT section
                    transcript = old_doc
                    if "TRANSCRIPT:" in old_doc:
                        transcript = old_doc.split("TRANSCRIPT:")[-1].strip()[:2000]
                    elif len(old_doc) > 100:
                        transcript = old_doc[:2000]
                    else:
                        transcript = meta.get("summary", "")
                    
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
                
                # Upsert directly to existing collection (recomputes embeddings)
                collection.upsert(
                    ids=ids,
                    documents=docs,
                    metadatas=metas
                )
                
                processed += len(segs)
                _jobs[job_id]["processed"] = processed
                
            except Exception as e:
                print(f"Failed to reindex episode {episode_id}: {e}")
                continue
        
        # Clean up any leftover temp collections from previous failed reindexes
        try:
            client.delete_collection("segments_new")
        except:
            pass
        try:
            client.delete_collection("segments_backup")
        except:
            pass
        
        _jobs[job_id]["status"] = "complete"
        _jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        _jobs[job_id]["processed"] = processed
        
    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)[:300]


# =============================================================================
# Download & Transcribe Only (no ChromaDB ingestion)
# =============================================================================

class DownloadTranscribeRequest(BaseModel):
    channel: str
    podcast_name: Optional[str] = None
    limit: int = 10
    whisper_model: str = "tiny"  # tiny, base, small, medium, large


def _save_transcript(video_id: str, title: str, podcast: str, segments: list, output_dir: str):
    """Save transcript to files (txt and json)."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Sanitize filename
    safe_title = re.sub(r'[<>:"/\\|?*]', '', title)
    safe_title = re.sub(r'\s+', '_', safe_title)[:80]
    
    timestamp = datetime.now().strftime("%Y%m%d")
    base_name = f"{timestamp}_{safe_title}"
    
    # Full transcript text
    full_text = " ".join([seg["text"] for seg in segments])
    
    # Save plain text
    txt_path = os.path.join(output_dir, f"{base_name}.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Title: {title}\n")
        f.write(f"Podcast: {podcast}\n")
        f.write(f"Video ID: {video_id}\n")
        f.write(f"Duration: {segments[-1]['end_ms'] // 1000 if segments else 0} seconds\n")
        f.write("=" * 60 + "\n\n")
        f.write(full_text)
    
    # Save JSON with timestamps
    json_path = os.path.join(output_dir, f"{base_name}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "video_id": video_id,
            "title": title,
            "podcast": podcast,
            "segments": segments,
            "full_text": full_text,
            "duration_ms": segments[-1]['end_ms'] if segments else 0,
        }, f, indent=2)
    
    return txt_path, json_path


def _transcribe_with_whisper(audio_path: str, output_dir: str, model: str = "tiny") -> list:
    """
    Transcribe audio file using faster-whisper with CUDA + int8.
    Falls back to CPU if CUDA unavailable.
    Returns list of segments with timestamps.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise Exception("faster-whisper not installed. Run: pip install faster-whisper")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Try CUDA with int8 quantization (good for older GPUs like GTX 1050 Ti)
    # Falls back to CPU if CUDA not available
    try:
        print(f"[Whisper] Loading {model} model with CUDA int8...")
        whisper_model = WhisperModel(model, device="cuda", compute_type="int8")
        device_used = "cuda (int8)"
    except Exception as e:
        print(f"[Whisper] CUDA failed ({e}), falling back to CPU...")
        whisper_model = WhisperModel(model, device="cpu", compute_type="int8")
        device_used = "cpu (int8)"
    
    print(f"[Whisper] Transcribing with {device_used}: {audio_path}")
    
    # Transcribe
    segments_iter, info = whisper_model.transcribe(
        audio_path,
        language="en",
        beam_size=5,
        vad_filter=True,  # Filter out silence
    )
    
    print(f"[Whisper] Detected language: {info.language} (prob: {info.language_probability:.2f})")
    
    # Convert to our segment format
    segments = []
    for seg in segments_iter:
        segments.append({
            "start_ms": int(seg.start * 1000),
            "end_ms": int(seg.end * 1000),
            "text": seg.text.strip()
        })
    
    print(f"[Whisper] Transcribed {len(segments)} segments using {device_used}")
    
    # Also save JSON for reference
    audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
    json_path = os.path.join(output_dir, f"{audio_basename}_whisper.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            "audio": audio_path,
            "model": model,
            "device": device_used,
            "language": info.language,
            "segments": segments,
        }, f, indent=2)
    
    return segments


def _download_transcribe_channel(job_id: str, channel: str, podcast_name: Optional[str], limit: int, whisper_model: str = "tiny"):
    """Background: download and transcribe videos from a channel using Whisper (no indexing)."""
    try:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "transcripts")
        audio_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio")
        
        _jobs[job_id]["status"] = "listing_videos"
        _jobs[job_id]["output_dir"] = output_dir
        _jobs[job_id]["whisper_model"] = whisper_model
        
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
        _jobs[job_id]["status"] = "downloading"
        _jobs[job_id]["videos"] = []
        
        name = podcast_name or channel
        success = 0
        failed = 0
        transcripts = []
        
        for i, video in enumerate(videos):
            vid_status = {
                "id": video["id"],
                "title": video["title"],
                "status": "downloading",
                "index": i + 1,
            }
            _jobs[job_id]["videos"].append(vid_status)
            _jobs[job_id]["current_video"] = i + 1
            
            try:
                url = video["url"]
                video_id = video["id"]
                
                # Download audio first
                vid_status["status"] = "downloading"
                audio_path = _download_audio(url, video_id)
                
                # Get full path to audio file
                full_audio_path = os.path.join(audio_dir, f"{video_id}.mp3")
                if not os.path.exists(full_audio_path):
                    vid_status["status"] = "failed"
                    vid_status["error"] = "Audio download failed"
                    failed += 1
                    continue
                
                # Transcribe with Whisper
                vid_status["status"] = "transcribing"
                _jobs[job_id]["status"] = "transcribing"
                
                segments = _transcribe_with_whisper(full_audio_path, output_dir, whisper_model)
                
                if not segments:
                    vid_status["status"] = "failed"
                    vid_status["error"] = "Whisper returned no segments"
                    failed += 1
                    continue
                
                # Save transcript files
                txt_path, json_path = _save_transcript(video_id, video["title"], name, segments, output_dir)
                
                vid_status["status"] = "complete"
                vid_status["transcript"] = txt_path
                vid_status["segments"] = len(segments)
                transcripts.append({
                    "video_id": video_id,
                    "title": video["title"],
                    "txt": txt_path,
                    "json": json_path,
                    "audio": audio_path,
                    "segments": len(segments),
                })
                success += 1
                
            except Exception as e:
                vid_status["status"] = "failed"
                vid_status["error"] = str(e)[:100]
                failed += 1
        
        _jobs[job_id]["status"] = "complete"
        _jobs[job_id]["success"] = success
        _jobs[job_id]["failed"] = failed
        _jobs[job_id]["transcripts"] = transcripts
        _jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        
    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)[:300]


@router.post("/download-transcribe")
async def download_transcribe_channel(request: DownloadTranscribeRequest, background_tasks: BackgroundTasks):
    """
    Download and transcribe videos from a channel using Whisper.
    Outputs raw transcript files (txt + json) to app/transcripts/ folder.
    Does NOT index to ChromaDB — raw files only.
    """
    global _job_counter
    _job_counter += 1
    job_id = f"transcribe_{_job_counter}_{int(time.time())}"
    
    _jobs[job_id] = {
        "job_id": job_id,
        "type": "download_transcribe",
        "channel": request.channel,
        "limit": request.limit,
        "whisper_model": request.whisper_model,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat(),
    }
    
    background_tasks.add_task(
        _download_transcribe_channel, job_id, request.channel, request.podcast_name, request.limit, request.whisper_model
    )
    
    return {"job_id": job_id, "status": "queued", "whisper_model": request.whisper_model}


@router.get("/transcripts")
async def list_transcripts():
    """List all downloaded transcripts."""
    transcript_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "transcripts")
    if not os.path.exists(transcript_dir):
        return {"transcripts": [], "path": transcript_dir}
    
    transcripts = []
    for f in Path(transcript_dir).glob("*.txt"):
        json_file = f.with_suffix(".json")
        transcripts.append({
            "txt": str(f),
            "json": str(json_file) if json_file.exists() else None,
            "name": f.stem,
            "size_kb": f.stat().st_size // 1024,
        })
    
    return {
        "transcripts": sorted(transcripts, key=lambda x: x["name"], reverse=True),
        "path": transcript_dir,
        "count": len(transcripts),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline V2 - WAV + MFA for accurate timestamps
# ═══════════════════════════════════════════════════════════════════════════════

# V2 job storage (separate from v1 for clarity)
_v2_jobs: Dict[str, dict] = {}


class IngestV2Request(BaseModel):
    url: str
    podcast_name: Optional[str] = None
    use_whisper: bool = False  # Force Whisper instead of YouTube captions
    whisper_model: str = "tiny"


class IngestChannelV2Request(BaseModel):
    channel: str
    podcast_name: Optional[str] = None
    limit: int = 10
    use_whisper: bool = False
    whisper_model: str = "tiny"


def _run_v2_pipeline(job_id: str, url: str, podcast_name: Optional[str], use_whisper: bool, whisper_model: str):
    """Background task: run v2 pipeline for single video."""
    from app.services.pipeline import PipelineJob, run_pipeline, extract_video_id
    
    video_id = extract_video_id(url) or f"unknown_{int(time.time())}"
    audio_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio")
    
    job = PipelineJob(
        job_id=job_id,
        video_id=video_id,
        url=url,
        podcast_name=podcast_name,
        use_whisper=use_whisper,
        whisper_model=whisper_model
    )
    
    def on_update(j: PipelineJob):
        _v2_jobs[job_id] = j.to_dict()
    
    _v2_jobs[job_id] = job.to_dict()
    
    success = run_pipeline(job, audio_dir, on_update)
    _v2_jobs[job_id] = job.to_dict()


def _run_v2_channel_pipeline(
    job_id: str, 
    channel: str, 
    podcast_name: Optional[str], 
    limit: int,
    use_whisper: bool, 
    whisper_model: str
):
    """Background task: run v2 pipeline for channel."""
    from app.services.pipeline import PipelineJob, run_pipeline, extract_video_id
    
    audio_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "audio")
    
    # Initialize channel job
    _v2_jobs[job_id] = {
        "job_id": job_id,
        "type": "channel_v2",
        "channel": channel,
        "status": "listing_videos",
        "videos": [],
        "total_videos": 0,
        "current_video": 0,
        "success": 0,
        "failed": 0,
        "skipped": 0,
        "created_at": datetime.utcnow().isoformat(),
    }
    
    try:
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
            _v2_jobs[job_id]["status"] = "failed"
            _v2_jobs[job_id]["error"] = "No videos found"
            return
        
        _v2_jobs[job_id]["total_videos"] = len(videos)
        _v2_jobs[job_id]["status"] = "processing"
        
        name = podcast_name or channel
        
        for i, video in enumerate(videos):
            video_id = video["id"]
            
            # Check if already ingested
            if _is_already_ingested(video_id):
                _v2_jobs[job_id]["videos"].append({
                    "id": video_id,
                    "title": video["title"],
                    "status": "skipped",
                    "reason": "already ingested"
                })
                _v2_jobs[job_id]["skipped"] += 1
                _v2_jobs[job_id]["current_video"] = i + 1
                continue
            
            # Create pipeline job for this video
            video_job = PipelineJob(
                job_id=f"{job_id}_{video_id}",
                video_id=video_id,
                url=video["url"],
                podcast_name=name,
                use_whisper=use_whisper,
                whisper_model=whisper_model
            )
            
            _v2_jobs[job_id]["videos"].append({
                "id": video_id,
                "title": video["title"],
                "status": "processing",
                "steps": {}
            })
            _v2_jobs[job_id]["current_video"] = i + 1
            
            def on_video_update(j: PipelineJob):
                # Update the video entry in channel job
                for v in _v2_jobs[job_id]["videos"]:
                    if v["id"] == video_id:
                        v["status"] = j.status
                        v["steps"] = {k: s.to_dict() for k, s in j.steps.items()}
                        v["current_step"] = j.current_step
                        v["progress_pct"] = j.progress_pct
                        break
            
            success = run_pipeline(video_job, audio_dir, on_video_update)
            
            # Update final status
            for v in _v2_jobs[job_id]["videos"]:
                if v["id"] == video_id:
                    v["status"] = "complete" if success else "failed"
                    if not success and video_job.error:
                        v["error"] = video_job.error
                    break
            
            if success:
                _v2_jobs[job_id]["success"] += 1
            else:
                _v2_jobs[job_id]["failed"] += 1
            
            time.sleep(1)  # Rate limit
        
        _v2_jobs[job_id]["status"] = "complete"
        _v2_jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()
        
    except Exception as e:
        _v2_jobs[job_id]["status"] = "failed"
        _v2_jobs[job_id]["error"] = str(e)


@router.post("/v2/url")
async def ingest_url_v2(request: IngestV2Request, background_tasks: BackgroundTasks):
    """
    Ingest a single YouTube URL using the v2 pipeline (WAV + MFA).
    
    Pipeline steps:
    1. Download MP3
    2. Get transcript (YouTube captions, Whisper fallback)
    3. Convert to WAV (16kHz mono)
    4. MFA forced alignment
    5. Parse TextGrid → word timestamps
    6. LLM segmentation
    7. Store in ChromaDB
    """
    global _job_counter
    _job_counter += 1
    job_id = f"v2_{_job_counter}_{int(time.time())}"
    
    background_tasks.add_task(
        _run_v2_pipeline, 
        job_id, 
        request.url, 
        request.podcast_name,
        request.use_whisper,
        request.whisper_model
    )
    
    return {"job_id": job_id, "status": "queued", "pipeline": "v2"}


@router.post("/v2/channel")
async def ingest_channel_v2(request: IngestChannelV2Request, background_tasks: BackgroundTasks):
    """
    Ingest videos from a channel using the v2 pipeline.
    """
    global _job_counter
    _job_counter += 1
    job_id = f"v2_channel_{_job_counter}_{int(time.time())}"
    
    background_tasks.add_task(
        _run_v2_channel_pipeline,
        job_id,
        request.channel,
        request.podcast_name,
        request.limit,
        request.use_whisper,
        request.whisper_model
    )
    
    return {"job_id": job_id, "status": "queued", "pipeline": "v2", "channel": request.channel}


@router.get("/v2/jobs")
async def list_v2_jobs():
    """List all v2 pipeline jobs (recent first)."""
    jobs = list(_v2_jobs.values())
    # Sort by created_at descending
    jobs.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    return {"jobs": jobs[:50]}  # Return last 50


@router.get("/v2/jobs/{job_id}")
async def get_v2_job(job_id: str):
    """Get status of a specific v2 job."""
    if job_id not in _v2_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return _v2_jobs[job_id]


@router.get("/v2/status")
async def get_v2_status():
    """Check v2 pipeline dependencies."""
    from app.services.mfa_align import check_mfa_installed, check_mfa_models
    
    mfa_installed = check_mfa_installed()
    mfa_models = check_mfa_models() if mfa_installed else {"dictionary": False, "acoustic": False}
    
    # Check ffmpeg
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, timeout=5)
        ffmpeg_installed = result.returncode == 0
    except:
        ffmpeg_installed = False
    
    # Check faster-whisper
    try:
        result = subprocess.run(["faster-whisper", "--help"], capture_output=True, timeout=5)
        whisper_installed = result.returncode == 0
    except:
        whisper_installed = False
    
    return {
        "v2_ready": mfa_installed and mfa_models["dictionary"] and mfa_models["acoustic"] and ffmpeg_installed,
        "dependencies": {
            "ffmpeg": ffmpeg_installed,
            "mfa": mfa_installed,
            "mfa_dictionary": mfa_models["dictionary"],
            "mfa_acoustic_model": mfa_models["acoustic"],
            "faster_whisper": whisper_installed,
        },
        "notes": {
            "mfa_install": "conda install -c conda-forge montreal-forced-aligner",
            "mfa_models": "mfa model download dictionary english_mfa && mfa model download acoustic english_mfa",
        }
    }
