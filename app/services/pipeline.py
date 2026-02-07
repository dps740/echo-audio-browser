"""
Echo Ingestion Pipeline v2 - WAV + MFA for accurate timestamps.

Pipeline steps:
1. Download MP3 + transcript (YouTube captions or Whisper)
2. Convert MP3 → WAV (16kHz mono)
3. MFA Alignment → TextGrid with word timestamps
4. Parse TextGrid → JSON word timestamps  
5. LLM Segmentation → Topic segments
6. Store in ChromaDB
7. Generate MP3 clips for serving
"""

import os
import json
import time
import subprocess
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable
from datetime import datetime
import re


@dataclass
class PipelineStep:
    """A single step in the pipeline."""
    name: str
    status: str = "pending"  # pending, running, complete, failed, skipped
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    details: Dict = field(default_factory=dict)
    
    @property
    def duration_sec(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return self.completed_at - self.started_at
        elif self.started_at:
            return time.time() - self.started_at
        return None
    
    def start(self):
        self.status = "running"
        self.started_at = time.time()
    
    def complete(self, details: Dict = None):
        self.status = "complete"
        self.completed_at = time.time()
        if details:
            self.details.update(details)
    
    def fail(self, error: str):
        self.status = "failed"
        self.completed_at = time.time()
        self.error = error
    
    def skip(self, reason: str = None):
        self.status = "skipped"
        self.completed_at = time.time()
        if reason:
            self.details["skip_reason"] = reason
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "status": self.status,
            "duration_sec": round(self.duration_sec, 1) if self.duration_sec else None,
            "error": self.error,
            "details": self.details
        }


@dataclass
class PipelineJob:
    """Full pipeline job with all steps."""
    job_id: str
    video_id: str
    url: str
    podcast_name: Optional[str] = None
    
    # Video metadata
    title: str = ""
    channel: str = ""
    duration_sec: int = 0
    
    # Pipeline configuration
    use_whisper: bool = False  # If True, always use Whisper (not YouTube captions)
    whisper_model: str = "tiny"
    
    # Paths
    work_dir: str = ""
    mp3_path: str = ""
    wav_path: str = ""
    transcript_path: str = ""
    textgrid_path: str = ""
    words_json_path: str = ""
    
    # Steps
    steps: Dict[str, PipelineStep] = field(default_factory=dict)
    
    # Overall status
    status: str = "queued"  # queued, running, complete, failed
    created_at: str = ""
    completed_at: str = ""
    error: str = ""
    
    # Results
    segments_created: int = 0
    clips_created: int = 0
    transcript_source: str = ""  # "youtube" or "whisper"
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat()
        
        if not self.steps:
            self.steps = {
                "download": PipelineStep("Download MP3"),
                "transcript": PipelineStep("Get Transcript"),
                "convert_wav": PipelineStep("Convert to WAV"),
                "mfa_align": PipelineStep("MFA Alignment"),
                "parse_textgrid": PipelineStep("Parse Timestamps"),
                "segment": PipelineStep("LLM Segmentation"),
                "store": PipelineStep("Store & Index"),
            }
    
    @property
    def current_step(self) -> Optional[str]:
        for name, step in self.steps.items():
            if step.status == "running":
                return name
        return None
    
    @property
    def progress_pct(self) -> int:
        total = len(self.steps)
        done = sum(1 for s in self.steps.values() if s.status in ("complete", "skipped"))
        return int((done / total) * 100)
    
    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "type": "pipeline_v2",
            "video_id": self.video_id,
            "url": self.url,
            "title": self.title,
            "channel": self.channel,
            "duration_sec": self.duration_sec,
            "status": self.status,
            "current_step": self.current_step,
            "progress_pct": self.progress_pct,
            "steps": {k: v.to_dict() for k, v in self.steps.items()},
            "transcript_source": self.transcript_source,
            "segments_created": self.segments_created,
            "clips_created": self.clips_created,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error,
        }


def extract_video_id(url: str) -> Optional[str]:
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


def run_pipeline(
    job: PipelineJob,
    audio_dir: str,
    on_update: Callable[[PipelineJob], None] = None
) -> bool:
    """
    Run the full ingestion pipeline.
    
    Args:
        job: Pipeline job configuration
        audio_dir: Directory to store audio files
        on_update: Callback for job updates
    
    Returns:
        True if successful, False otherwise
    """
    def update():
        if on_update:
            on_update(job)
    
    job.status = "running"
    update()
    
    # Create work directory
    job.work_dir = tempfile.mkdtemp(prefix=f"echo_{job.video_id}_")
    
    try:
        # Step 1: Download MP3
        if not _step_download(job, audio_dir):
            return False
        update()
        
        # Step 2: Get transcript
        if not _step_transcript(job):
            return False
        update()
        
        # Step 3: Convert to WAV
        if not _step_convert_wav(job):
            return False
        update()
        
        # Step 4: MFA Alignment
        if not _step_mfa_align(job):
            return False
        update()
        
        # Step 5: Parse TextGrid
        if not _step_parse_textgrid(job):
            return False
        update()
        
        # Step 6: LLM Segmentation
        if not _step_segment(job):
            return False
        update()
        
        # Step 7: Store in ChromaDB
        if not _step_store(job, audio_dir):
            return False
        update()
        
        job.status = "complete"
        job.completed_at = datetime.utcnow().isoformat()
        update()
        return True
        
    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.completed_at = datetime.utcnow().isoformat()
        update()
        return False
        
    finally:
        # Cleanup work directory (keep audio files)
        try:
            if job.work_dir and os.path.exists(job.work_dir):
                shutil.rmtree(job.work_dir)
        except:
            pass


def _step_download(job: PipelineJob, audio_dir: str) -> bool:
    """Download MP3 from YouTube."""
    step = job.steps["download"]
    step.start()
    
    try:
        # Get video info first
        cmd = ["yt-dlp", "--dump-json", "--no-download", job.url]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            step.fail(f"yt-dlp info failed: {result.stderr[:200]}")
            return False
        
        info = json.loads(result.stdout)
        job.video_id = info["id"]
        job.title = info.get("title", "Unknown")
        job.channel = job.podcast_name or info.get("channel", info.get("uploader", "Unknown"))
        job.duration_sec = info.get("duration", 0)
        
        # Check if MP3 already exists
        mp3_path = os.path.join(audio_dir, f"{job.video_id}.mp3")
        if os.path.exists(mp3_path):
            job.mp3_path = mp3_path
            step.complete({"cached": True})
            return True
        
        # Download audio
        os.makedirs(audio_dir, exist_ok=True)
        cmd = [
            "yt-dlp",
            "-f", "bestaudio",
            "-x", "--audio-format", "mp3",
            "--audio-quality", "5",
            "-o", mp3_path.replace('.mp3', '.%(ext)s'),
            job.url
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        # Handle different output extensions
        base = mp3_path.replace('.mp3', '')
        for ext in ['.mp3', '.m4a', '.opus', '.webm', '.wav']:
            if os.path.exists(base + ext):
                if ext != '.mp3':
                    # Convert to MP3
                    subprocess.run([
                        "ffmpeg", "-y", "-i", base + ext,
                        "-codec:a", "libmp3lame", "-b:a", "128k",
                        mp3_path
                    ], capture_output=True, timeout=300)
                    os.remove(base + ext)
                break
        
        if os.path.exists(mp3_path):
            job.mp3_path = mp3_path
            step.complete({"size_mb": round(os.path.getsize(mp3_path) / 1024 / 1024, 1)})
            return True
        else:
            step.fail("MP3 download failed - file not created")
            return False
            
    except Exception as e:
        step.fail(str(e))
        return False


def _step_transcript(job: PipelineJob) -> bool:
    """Get transcript (YouTube captions or Whisper)."""
    step = job.steps["transcript"]
    step.start()
    
    transcript_text = ""
    
    try:
        # Try YouTube captions first (unless use_whisper is forced)
        if not job.use_whisper:
            transcript_text = _get_youtube_transcript(job.url)
            if transcript_text:
                job.transcript_source = "youtube"
        
        # Fallback to Whisper
        if not transcript_text:
            transcript_text = _run_whisper(job.mp3_path, job.whisper_model)
            if transcript_text:
                job.transcript_source = "whisper"
        
        if not transcript_text:
            step.fail("No transcript available (YouTube captions and Whisper both failed)")
            return False
        
        # Save transcript
        job.transcript_path = os.path.join(job.work_dir, "transcript.txt")
        with open(job.transcript_path, 'w', encoding='utf-8') as f:
            f.write(transcript_text)
        
        word_count = len(transcript_text.split())
        step.complete({
            "source": job.transcript_source,
            "word_count": word_count
        })
        return True
        
    except Exception as e:
        step.fail(str(e))
        return False


def _get_youtube_transcript(url: str) -> Optional[str]:
    """Get transcript from YouTube captions."""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Try to get captions
            cmd = [
                "yt-dlp",
                "--write-auto-sub", "--write-sub",
                "--sub-lang", "en",
                "--sub-format", "vtt",
                "--skip-download",
                "-o", f"{tmpdir}/subs",
                url
            ]
            subprocess.run(cmd, capture_output=True, timeout=120)
            
            # Find VTT file
            vtt_files = list(Path(tmpdir).glob("*.vtt"))
            if not vtt_files:
                return None
            
            # Parse VTT to plain text
            with open(vtt_files[0], 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract text from VTT
            lines = []
            for line in content.split('\n'):
                # Skip timing lines and headers
                if '-->' in line or line.startswith('WEBVTT') or not line.strip():
                    continue
                # Remove HTML tags
                clean = re.sub(r'<[^>]+>', '', line).strip()
                if clean and not clean[0].isdigit():  # Skip cue numbers
                    lines.append(clean)
            
            # Deduplicate consecutive identical lines (common in auto-captions)
            deduped = []
            prev = ""
            for line in lines:
                if line != prev:
                    deduped.append(line)
                    prev = line
            
            return ' '.join(deduped)
            
    except Exception as e:
        print(f"YouTube transcript failed: {e}")
        return None


def _run_whisper(audio_path: str, model: str = "tiny") -> Optional[str]:
    """Run Whisper transcription."""
    try:
        # Check for faster-whisper first
        result = subprocess.run(
            ["faster-whisper", audio_path, "--model", model, "--output_format", "txt"],
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout for long files
        )
        
        if result.returncode == 0:
            # faster-whisper outputs to {filename}.txt
            txt_path = audio_path.rsplit('.', 1)[0] + '.txt'
            if os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    return f.read()
        
        return None
        
    except FileNotFoundError:
        # faster-whisper not installed
        print("faster-whisper not found, Whisper fallback not available")
        return None
    except Exception as e:
        print(f"Whisper failed: {e}")
        return None


def _step_convert_wav(job: PipelineJob) -> bool:
    """Convert MP3 to WAV (16kHz mono for MFA)."""
    step = job.steps["convert_wav"]
    step.start()
    
    try:
        from app.services.mfa_align import convert_to_wav, MFAConfig
        
        job.wav_path = os.path.join(job.work_dir, "audio.wav")
        config = MFAConfig()
        
        if convert_to_wav(job.mp3_path, job.wav_path, config):
            size_mb = round(os.path.getsize(job.wav_path) / 1024 / 1024, 1)
            step.complete({"size_mb": size_mb})
            return True
        else:
            step.fail("WAV conversion failed")
            return False
            
    except Exception as e:
        step.fail(str(e))
        return False


def _step_mfa_align(job: PipelineJob) -> bool:
    """Run MFA forced alignment."""
    step = job.steps["mfa_align"]
    step.start()
    
    try:
        from app.services.mfa_align import (
            prepare_mfa_input, 
            run_mfa_align, 
            MFAConfig,
            check_mfa_installed
        )
        
        # Check MFA is available
        if not check_mfa_installed():
            step.fail("MFA not installed or not in PATH")
            return False
        
        # Load transcript
        with open(job.transcript_path, 'r', encoding='utf-8') as f:
            transcript_text = f.read()
        
        # Prepare MFA input
        mfa_input, mfa_output = prepare_mfa_input(
            job.wav_path,
            transcript_text,
            job.work_dir
        )
        
        # Run alignment
        config = MFAConfig()
        textgrid_path = run_mfa_align(mfa_input, mfa_output, config)
        
        if textgrid_path and os.path.exists(textgrid_path):
            job.textgrid_path = textgrid_path
            step.complete()
            return True
        else:
            step.fail("MFA alignment produced no output")
            return False
            
    except Exception as e:
        step.fail(str(e))
        return False


def _step_parse_textgrid(job: PipelineJob) -> bool:
    """Parse TextGrid to extract word timestamps."""
    step = job.steps["parse_textgrid"]
    step.start()
    
    try:
        from app.services.textgrid_parser import parse_textgrid, export_to_json
        
        words = parse_textgrid(job.textgrid_path)
        
        if not words:
            step.fail("No words found in TextGrid")
            return False
        
        # Save as JSON
        job.words_json_path = os.path.join(job.work_dir, "words.json")
        with open(job.words_json_path, 'w', encoding='utf-8') as f:
            json.dump(export_to_json(words), f)
        
        total_duration = words[-1].end if words else 0
        step.complete({
            "word_count": len(words),
            "duration_sec": round(total_duration, 1)
        })
        return True
        
    except Exception as e:
        step.fail(str(e))
        return False


def _step_segment(job: PipelineJob) -> bool:
    """Run LLM segmentation using word timestamps."""
    step = job.steps["segment"]
    step.start()
    
    try:
        from app.services.textgrid_parser import load_from_json
        from app.services.segmentation import segment_transcript
        from app.services.transcription import TranscriptResult, TranscriptWord
        import asyncio
        
        # Load word timestamps
        with open(job.words_json_path, 'r', encoding='utf-8') as f:
            words_data = json.load(f)
        
        words = load_from_json(words_data)
        
        # Convert to TranscriptResult format
        transcript_words = [
            TranscriptWord(
                word=w.word,
                start_ms=w.start_ms,
                end_ms=w.end_ms,
                confidence=1.0
            )
            for w in words
        ]
        
        transcript = TranscriptResult(
            text=' '.join(w.word for w in words),
            words=transcript_words,
            duration_ms=words[-1].end_ms if words else 0,
            speakers=[]
        )
        
        # Run LLM segmentation
        segments = asyncio.run(segment_transcript(transcript, job.title))
        job.segments_created = len(segments)
        
        # Save segments for next step
        segments_path = os.path.join(job.work_dir, "segments.json")
        with open(segments_path, 'w', encoding='utf-8') as f:
            json.dump([{
                "start_ms": s.start_ms,
                "end_ms": s.end_ms,
                "summary": s.summary,
                "topic_tags": s.topic_tags,
                "density_score": s.density_score,
                "transcript_text": s.transcript_text
            } for s in segments], f)
        
        step.complete({"segment_count": len(segments)})
        return True
        
    except Exception as e:
        step.fail(str(e))
        return False


def _step_store(job: PipelineJob, audio_dir: str) -> bool:
    """Store segments in ChromaDB and optionally generate clips."""
    step = job.steps["store"]
    step.start()
    
    try:
        import chromadb
        from app.config import get_embedding_function
        
        # Load segments
        segments_path = os.path.join(job.work_dir, "segments.json")
        with open(segments_path, 'r', encoding='utf-8') as f:
            segments = json.load(f)
        
        # Get ChromaDB collection
        client = chromadb.PersistentClient(path="./chroma_data")
        embedding_fn = get_embedding_function()
        collection = client.get_or_create_collection(
            "segments",
            embedding_function=embedding_fn
        )
        
        # Prepare for ChromaDB
        ids = []
        docs = []
        metas = []
        
        # Audio URL for playback (WAV for accuracy)
        # Copy WAV to audio dir for serving
        wav_dest = os.path.join(audio_dir, f"{job.video_id}.wav")
        if not os.path.exists(wav_dest):
            shutil.copy(job.wav_path, wav_dest)
        audio_url = f"/audio/{job.video_id}.wav"
        
        for i, seg in enumerate(segments):
            segment_id = f"{job.video_id}_{i}"
            
            # Build enriched document for embedding
            enriched_doc = f"""TOPIC: {', '.join(seg['topic_tags'])}
SUMMARY: {seg['summary']}
TRANSCRIPT: {seg['transcript_text'][:2000]}"""
            
            ids.append(segment_id)
            docs.append(enriched_doc)
            metas.append({
                "episode_id": job.video_id,
                "episode_title": job.title,
                "podcast_title": job.channel,
                "audio_url": audio_url,
                "start_ms": seg["start_ms"],
                "end_ms": seg["end_ms"],
                "summary": seg["summary"],
                "topic_tags": ",".join(seg["topic_tags"]),
                "density_score": seg["density_score"],
                "source": job.transcript_source,
            })
        
        # Batch upsert
        batch_size = 50
        for j in range(0, len(ids), batch_size):
            collection.upsert(
                ids=ids[j:j+batch_size],
                documents=docs[j:j+batch_size],
                metadatas=metas[j:j+batch_size]
            )
        
        step.complete({
            "segments_stored": len(segments),
            "audio_format": "wav"
        })
        return True
        
    except Exception as e:
        step.fail(str(e))
        return False
