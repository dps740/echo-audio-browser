"""Download-only endpoints — download MP3 + VTT from YouTube channels."""

import os
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/ingest/youtube", tags=["download"])


# ── Curated podcast list ──────────────────────────────────────────────────

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


# ── Job tracking ──────────────────────────────────────────────────────────

_jobs: dict = {}
_job_counter = 0


# ── Download logic ────────────────────────────────────────────────────────

class DownloadOnlyRequest(BaseModel):
    channel: str
    podcast_name: Optional[str] = None
    limit: int = 10


def _download_only_channel(
    job_id: str, channel: str, podcast_name: Optional[str], limit: int
):
    """Background task: download MP3 + VTT from a channel."""
    try:
        downloads_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "downloads"
        )
        os.makedirs(downloads_dir, exist_ok=True)

        _jobs[job_id]["status"] = "listing_videos"
        _jobs[job_id]["output_dir"] = downloads_dir

        # List videos in channel
        cmd = [
            "yt-dlp", "--flat-playlist", "--dump-json",
            f"https://www.youtube.com/{channel}/videos",
            "--playlist-end", str(limit)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        videos = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            try:
                data = json.loads(line)
                vid = {
                    "id": data.get("id"),
                    "title": data.get("title", "Unknown"),
                    "url": f"https://www.youtube.com/watch?v={data['id']}",
                    "duration": data.get("duration"),
                }
                if vid.get("duration") and vid["duration"] < 300:
                    continue  # Skip < 5 min
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

        name = podcast_name or channel.replace("@", "")
        podcast_dir = os.path.join(downloads_dir, name)
        os.makedirs(podcast_dir, exist_ok=True)

        success = 0
        failed = 0

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
                video_id = video["id"]
                mp3_path = os.path.join(podcast_dir, f"{video_id}.mp3")
                vtt_path = os.path.join(podcast_dir, f"{video_id}.en.vtt")

                if os.path.exists(mp3_path) and os.path.exists(vtt_path):
                    vid_status["status"] = "skipped"
                    vid_status["reason"] = "already downloaded"
                    continue

                cmd = [
                    "yt-dlp",
                    "-x", "--audio-format", "mp3",
                    "--write-auto-sub", "--write-sub",
                    "--sub-lang", "en",
                    "--sub-format", "vtt",
                    "-o", os.path.join(podcast_dir, "%(id)s.%(ext)s"),
                    video["url"]
                ]

                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=600
                )

                if result.returncode != 0:
                    vid_status["status"] = "failed"
                    vid_status["error"] = (result.stderr[:100] if result.stderr
                                           else "Download failed")
                    failed += 1
                    continue

                has_mp3 = os.path.exists(mp3_path)
                has_vtt = any(Path(podcast_dir).glob(f"{video_id}*.vtt"))

                if has_mp3:
                    vid_status["status"] = "complete"
                    vid_status["mp3"] = True
                    vid_status["vtt"] = has_vtt
                    success += 1
                else:
                    vid_status["status"] = "failed"
                    vid_status["error"] = "MP3 not created"
                    failed += 1

                time.sleep(1)  # Rate limit

            except Exception as e:
                vid_status["status"] = "failed"
                vid_status["error"] = str(e)[:100]
                failed += 1

        _jobs[job_id]["status"] = "complete"
        _jobs[job_id]["success"] = success
        _jobs[job_id]["failed"] = failed
        _jobs[job_id]["output_dir"] = podcast_dir
        _jobs[job_id]["completed_at"] = datetime.utcnow().isoformat()

    except Exception as e:
        _jobs[job_id]["status"] = "failed"
        _jobs[job_id]["error"] = str(e)[:300]


# ── Endpoints ─────────────────────────────────────────────────────────────

@router.get("/podcasts")
async def list_curated_podcasts():
    """List curated podcasts available for download."""
    return {"podcasts": CURATED_PODCASTS}


@router.post("/download-only")
async def download_only_channel(
    request: DownloadOnlyRequest, background_tasks: BackgroundTasks
):
    """
    Download MP3 + VTT subtitles from a YouTube channel.
    No processing — just downloads for later indexing.
    """
    global _job_counter
    _job_counter += 1
    job_id = f"download_{_job_counter}_{int(time.time())}"

    _jobs[job_id] = {
        "job_id": job_id,
        "type": "download_only",
        "channel": request.channel,
        "limit": request.limit,
        "status": "queued",
        "created_at": datetime.utcnow().isoformat(),
    }

    background_tasks.add_task(
        _download_only_channel,
        job_id, request.channel, request.podcast_name, request.limit
    )

    return {"job_id": job_id, "status": "queued", "type": "download_only"}


@router.get("/downloads")
async def list_downloads():
    """List all downloaded files (MP3 + VTT)."""
    downloads_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "downloads"
    )
    if not os.path.exists(downloads_dir):
        return {"podcasts": [], "path": downloads_dir}

    podcasts = []
    for name in sorted(os.listdir(downloads_dir)):
        podcast_dir = os.path.join(downloads_dir, name)
        if not os.path.isdir(podcast_dir):
            continue

        files = os.listdir(podcast_dir)
        mp3s = [f for f in files if f.endswith('.mp3')]
        vtts = [f for f in files if f.endswith('.vtt')]

        podcasts.append({
            "name": name,
            "mp3_count": len(mp3s),
            "vtt_count": len(vtts),
            "path": podcast_dir,
        })

    return {"podcasts": podcasts, "path": downloads_dir}


@router.get("/jobs")
async def list_jobs():
    """List all download jobs."""
    return {
        "jobs": sorted(
            _jobs.values(),
            key=lambda j: j.get("created_at", ""),
            reverse=True,
        )
    }


@router.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get status of a specific job."""
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
