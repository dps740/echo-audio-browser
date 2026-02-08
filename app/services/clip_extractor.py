"""
Clip Extractor: Generate MP3 clips from WAV source files.

WAV files have accurate timestamps (MP3 seeking is unreliable).
We extract clips from WAV and convert to MP3 for delivery.
"""

import os
import subprocess
import hashlib
from pathlib import Path
from typing import Optional

# Clip cache directory
CLIP_CACHE_DIR = Path("audio/clips")
CLIP_CACHE_DIR.mkdir(exist_ok=True)


def get_clip_path(video_id: str, start_ms: int, end_ms: int) -> Path:
    """Generate a unique path for a clip based on its parameters."""
    # Create hash for unique filename
    clip_id = f"{video_id}_{start_ms}_{end_ms}"
    clip_hash = hashlib.md5(clip_id.encode()).hexdigest()[:12]
    return CLIP_CACHE_DIR / f"{video_id}_{clip_hash}.mp3"


def extract_clip(
    video_id: str,
    start_ms: int,
    end_ms: int,
    audio_dir: Path = Path("audio"),
    padding_ms: int = 500  # Small padding for clean cuts
) -> Optional[Path]:
    """
    Extract a clip from the source WAV file.
    
    Args:
        video_id: The video/episode ID
        start_ms: Start time in milliseconds
        end_ms: End time in milliseconds
        audio_dir: Directory containing audio files
        padding_ms: Padding to add at start/end for cleaner cuts
    
    Returns:
        Path to the generated MP3 clip, or None if failed
    """
    # Check cache first
    clip_path = get_clip_path(video_id, start_ms, end_ms)
    if clip_path.exists():
        return clip_path
    
    # Find source WAV file
    wav_path = audio_dir / f"{video_id}.wav"
    if not wav_path.exists():
        # Try without extension variations
        for ext in ['.wav', '.mp3', '.m4a']:
            test_path = audio_dir / f"{video_id}{ext}"
            if test_path.exists():
                wav_path = test_path
                break
        else:
            print(f"No audio file found for {video_id}")
            return None
    
    # Calculate times with padding
    start_sec = max(0, (start_ms - padding_ms)) / 1000
    duration_sec = (end_ms - start_ms + 2 * padding_ms) / 1000
    
    # Extract with ffmpeg
    cmd = [
        "ffmpeg",
        "-y",  # Overwrite output
        "-ss", str(start_sec),  # Start time
        "-i", str(wav_path),  # Input file
        "-t", str(duration_sec),  # Duration
        "-vn",  # No video
        "-acodec", "libmp3lame",  # MP3 codec
        "-ab", "128k",  # Bitrate
        "-ar", "44100",  # Sample rate
        str(clip_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            print(f"ffmpeg error: {result.stderr}")
            return None
        
        if clip_path.exists():
            return clip_path
        else:
            print(f"Clip not created: {clip_path}")
            return None
            
    except subprocess.TimeoutExpired:
        print(f"ffmpeg timeout for {video_id}")
        return None
    except Exception as e:
        print(f"Error extracting clip: {e}")
        return None


def get_clip_url(video_id: str, start_ms: int, end_ms: int) -> Optional[str]:
    """
    Get URL for a clip, extracting it if needed.
    
    Returns:
        URL path like "/audio/clips/xxx.mp3" or None if failed
    """
    clip_path = extract_clip(video_id, start_ms, end_ms)
    if clip_path:
        return f"/audio/clips/{clip_path.name}"
    return None


# Test
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python clip_extractor.py <video_id> <start_ms> <end_ms>")
        sys.exit(1)
    
    video_id = sys.argv[1]
    start_ms = int(sys.argv[2])
    end_ms = int(sys.argv[3])
    
    clip_url = get_clip_url(video_id, start_ms, end_ms)
    if clip_url:
        print(f"Clip created: {clip_url}")
    else:
        print("Failed to create clip")
