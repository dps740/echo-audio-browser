"""Transcription service using Deepgram or Whisper."""

import httpx
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from app.config import settings


@dataclass
class TranscriptWord:
    """A word with timestamp."""
    word: str
    start_ms: int
    end_ms: int
    confidence: float
    speaker: Optional[int] = None


@dataclass
class TranscriptResult:
    """Full transcription result."""
    text: str
    words: List[TranscriptWord]
    duration_ms: int
    speakers: List[int]


async def transcribe_audio(audio_url: str) -> TranscriptResult:
    """
    Transcribe audio from URL using Deepgram.
    
    Args:
        audio_url: URL of the audio file
        
    Returns:
        TranscriptResult with word-level timestamps and speaker diarization
    """
    if not settings.deepgram_api_key:
        raise ValueError("DEEPGRAM_API_KEY not set")
    
    # Use Deepgram API
    api_url = "https://api.deepgram.com/v1/listen"
    
    params = {
        "model": "nova-2",
        "smart_format": "true",
        "diarize": "true",
        "punctuate": "true",
        "utterances": "true",
    }
    
    headers = {
        "Authorization": f"Token {settings.deepgram_api_key}",
        "Content-Type": "application/json",
    }
    
    payload = {
        "url": audio_url
    }
    
    async with httpx.AsyncClient(timeout=600.0) as client:  # 10 min timeout for long episodes
        response = await client.post(
            api_url,
            params=params,
            headers=headers,
            json=payload
        )
        response.raise_for_status()
        data = response.json()
    
    # Parse response
    return _parse_deepgram_response(data)


def _parse_deepgram_response(data: Dict[str, Any]) -> TranscriptResult:
    """Parse Deepgram API response into TranscriptResult."""
    results = data.get("results", {})
    channels = results.get("channels", [])
    
    if not channels:
        raise ValueError("No transcription results")
    
    channel = channels[0]
    alternatives = channel.get("alternatives", [])
    
    if not alternatives:
        raise ValueError("No transcription alternatives")
    
    alt = alternatives[0]
    
    # Extract words with timestamps
    words = []
    speakers_seen = set()
    
    for word_data in alt.get("words", []):
        speaker = word_data.get("speaker")
        if speaker is not None:
            speakers_seen.add(speaker)
        
        words.append(TranscriptWord(
            word=word_data.get("word", ""),
            start_ms=int(word_data.get("start", 0) * 1000),
            end_ms=int(word_data.get("end", 0) * 1000),
            confidence=word_data.get("confidence", 0.0),
            speaker=speaker,
        ))
    
    # Get full text
    text = alt.get("transcript", "")
    
    # Calculate duration
    duration_ms = 0
    if words:
        duration_ms = words[-1].end_ms
    
    return TranscriptResult(
        text=text,
        words=words,
        duration_ms=duration_ms,
        speakers=list(speakers_seen),
    )


async def transcribe_audio_whisper(audio_url: str) -> TranscriptResult:
    """
    Fallback: Transcribe using local Whisper model.
    
    Note: This requires whisper to be installed locally.
    """
    # TODO: Implement local Whisper fallback
    # For MVP, we'll rely on Deepgram
    raise NotImplementedError("Local Whisper transcription not yet implemented")
