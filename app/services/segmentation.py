"""Segmentation service using LLM to identify atomic ideas."""

import json
from typing import List, Dict, Any
from dataclasses import dataclass
import openai
import anthropic

from app.config import settings
from app.services.transcription import TranscriptResult, TranscriptWord


@dataclass
class SegmentResult:
    """A semantic segment identified by the LLM."""
    start_ms: int
    end_ms: int
    summary: str
    topic_tags: List[str]
    density_score: float
    transcript_text: str


SEGMENTATION_PROMPT = """You are analyzing a podcast transcript to identify "Atomic Ideas" - standalone segments where a specific topic is discussed comprehensively.

Your task:
1. Identify 5-15 segments of 2-10 minutes each
2. Each segment should be a complete discussion of one topic
3. Rate each segment's "density score" (0.0-1.0) - how much valuable information vs filler/tangents
4. Assign 1-3 topic tags per segment

Rules:
- Segments should not overlap
- Prefer natural break points (topic changes, pauses)
- Higher density = more actionable information, insights, or arguments
- Lower density = small talk, lengthy anecdotes, repeated content

Return JSON array:
[
  {
    "start_ms": 0,
    "end_ms": 180000,
    "summary": "Brief 1-2 sentence summary of the segment",
    "topic_tags": ["AI", "Machine Learning"],
    "density_score": 0.8
  }
]

TRANSCRIPT:
{transcript}

Return ONLY valid JSON, no other text."""


async def segment_transcript(
    transcript: TranscriptResult,
    episode_title: str = "",
) -> List[SegmentResult]:
    """
    Use LLM to segment a transcript into atomic ideas.
    
    Args:
        transcript: The full transcript with word timestamps
        episode_title: Title for context
        
    Returns:
        List of identified segments
    """
    # Prepare transcript text with approximate timestamps every ~1 minute
    transcript_with_times = _prepare_transcript_with_times(transcript)
    
    # Choose LLM based on config
    if settings.anthropic_api_key and "claude" in settings.segmentation_model.lower():
        segments_json = await _segment_with_claude(transcript_with_times)
    elif settings.openai_api_key:
        segments_json = await _segment_with_openai(transcript_with_times)
    else:
        raise ValueError("No LLM API key configured")
    
    # Parse response and create SegmentResults
    return _parse_segments(segments_json, transcript)


def _prepare_transcript_with_times(transcript: TranscriptResult) -> str:
    """Add timestamp markers to transcript for LLM context."""
    lines = []
    current_minute = -1
    current_line = []
    
    for word in transcript.words:
        minute = word.start_ms // 60000
        if minute > current_minute:
            if current_line:
                lines.append(" ".join(current_line))
            current_minute = minute
            current_line = [f"[{minute}m]"]
        current_line.append(word.word)
    
    if current_line:
        lines.append(" ".join(current_line))
    
    return "\n".join(lines)


async def _segment_with_openai(transcript_text: str) -> List[Dict]:
    """Use OpenAI GPT to segment."""
    client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
    
    response = await client.chat.completions.create(
        model=settings.segmentation_model,
        messages=[
            {
                "role": "user",
                "content": SEGMENTATION_PROMPT.format(transcript=transcript_text[:50000])  # Limit length
            }
        ],
        temperature=0.3,
        response_format={"type": "json_object"},
    )
    
    content = response.choices[0].message.content
    
    # Parse JSON
    try:
        data = json.loads(content)
        if isinstance(data, dict) and "segments" in data:
            return data["segments"]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError("Unexpected response format")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response: {e}")


async def _segment_with_claude(transcript_text: str) -> List[Dict]:
    """Use Anthropic Claude to segment."""
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    
    response = await client.messages.create(
        model=settings.segmentation_model,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": SEGMENTATION_PROMPT.format(transcript=transcript_text[:50000])
            }
        ],
    )
    
    content = response.content[0].text
    
    # Parse JSON (Claude might include markdown code blocks)
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    
    try:
        data = json.loads(content.strip())
        if isinstance(data, dict) and "segments" in data:
            return data["segments"]
        elif isinstance(data, list):
            return data
        else:
            raise ValueError("Unexpected response format")
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse LLM response: {e}")


def _parse_segments(segments_json: List[Dict], transcript: TranscriptResult) -> List[SegmentResult]:
    """Convert JSON segments to SegmentResult objects with transcript text."""
    results = []
    
    for seg in segments_json:
        start_ms = seg.get("start_ms", 0)
        end_ms = seg.get("end_ms", 0)
        
        # Extract transcript text for this segment
        segment_words = [
            w.word for w in transcript.words
            if w.start_ms >= start_ms and w.end_ms <= end_ms
        ]
        transcript_text = " ".join(segment_words)
        
        results.append(SegmentResult(
            start_ms=start_ms,
            end_ms=end_ms,
            summary=seg.get("summary", ""),
            topic_tags=seg.get("topic_tags", []),
            density_score=float(seg.get("density_score", 0.5)),
            transcript_text=transcript_text,
        ))
    
    return results
