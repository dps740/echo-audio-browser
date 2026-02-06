"""Segmentation v2 - Topic-based, variable-length segments."""

import json
import re
from typing import List, Dict, Optional
from dataclasses import dataclass
from openai import OpenAI

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
    segment_type: str = "content"  # content, intro, outro, ad


TOPIC_BOUNDARY_PROMPT = """Analyze this podcast transcript and identify where TOPIC CHANGES occur.

Your job is to find the natural breakpoints where the conversation shifts to a new subject. 
DO NOT create fixed-length segments. Segments should vary based on how long each topic is discussed.

A topic might be discussed for:
- 90 seconds (brief tangent)
- 5 minutes (moderate discussion)  
- 12 minutes (deep dive)

CRITICAL RULES:
1. SKIP intro sections ("today we'll discuss...", "welcome back...", "before we start...")
2. SKIP outro sections ("thanks for listening", "subscribe", "next week we'll...")
3. SKIP ad reads and sponsor mentions
4. Each segment = ONE coherent topic, however long it takes
5. Boundaries should be at NATURAL TOPIC SHIFTS, not arbitrary time intervals

BAD OUTPUT (what we DON'T want):
- Segments at 0:00, 3:00, 6:00, 9:00, 12:00 (fixed intervals)
- First segment starting at 0:00 with intro content

GOOD OUTPUT (what we want):
- First segment starts AFTER intro (e.g., 2:30 when actual content begins)
- Segments of varying length: 1:45, 7:20, 4:15, 11:30
- Boundaries at actual topic transitions

For each segment provide:
1. start_time: When this topic BEGINS (in "MM:SS" format)
2. end_time: When this topic ENDS (in "MM:SS" format)  
3. topic: 2-5 word topic label
4. summary: 3-4 sentences capturing the SPECIFIC argument/insight (include names, numbers, claims)
5. density_score: 0.0-1.0 (how information-dense vs filler)
6. segment_type: "content", "intro", "outro", or "ad"

TRANSCRIPT:
{transcript}

Return JSON:
{{
  "segments": [
    {{
      "start_time": "2:30",
      "end_time": "9:45",
      "topic": "specific topic name",
      "summary": "Detailed summary with names, numbers, specific claims...",
      "density_score": 0.85,
      "segment_type": "content"
    }}
  ],
  "skipped": [
    {{"start_time": "0:00", "end_time": "2:30", "reason": "intro/preview"}}
  ]
}}

Return ONLY valid JSON."""


def parse_time_to_ms(time_str: str) -> int:
    """Convert MM:SS or HH:MM:SS to milliseconds."""
    parts = time_str.strip().split(":")
    if len(parts) == 2:
        minutes, seconds = int(parts[0]), float(parts[1])
        return int((minutes * 60 + seconds) * 1000)
    elif len(parts) == 3:
        hours, minutes, seconds = int(parts[0]), int(parts[1]), float(parts[2])
        return int((hours * 3600 + minutes * 60 + seconds) * 1000)
    else:
        return 0


def prepare_transcript_with_times(transcript: TranscriptResult) -> str:
    """Format transcript with timestamps for LLM."""
    lines = []
    current_minute = -1
    current_line = []
    
    for word in transcript.words:
        minute = word.start_ms // 60000
        second = (word.start_ms % 60000) // 1000
        
        # Add timestamp every 30 seconds
        if minute > current_minute or (minute == current_minute and second >= 30 and "[" not in " ".join(current_line[-3:])):
            if current_line:
                lines.append(" ".join(current_line))
            current_minute = minute
            time_marker = f"[{minute}:{second:02d}]"
            current_line = [time_marker]
        
        current_line.append(word.word)
    
    if current_line:
        lines.append(" ".join(current_line))
    
    return "\n".join(lines)


def segment_transcript_v2(
    transcript: TranscriptResult,
    episode_title: str = "",
) -> List[SegmentResult]:
    """
    Use LLM to segment transcript into topic-based, variable-length segments.
    """
    client = OpenAI(api_key=settings.openai_api_key)
    
    # Prepare transcript with timestamps
    transcript_text = prepare_transcript_with_times(transcript)
    
    print(f"[v2] Segmenting transcript ({len(transcript_text)} chars)")
    
    response = client.chat.completions.create(
        model="gpt-4o",  # Use best model for segmentation
        messages=[{
            "role": "user",
            "content": TOPIC_BOUNDARY_PROMPT.format(transcript=transcript_text[:60000])
        }],
        temperature=0.2,
        max_tokens=4000,
    )
    
    content = response.choices[0].message.content.strip()
    
    # Strip markdown code blocks
    if content.startswith("```"):
        content = re.sub(r'^```(?:json)?\n?', '', content)
        content = re.sub(r'\n?```$', '', content)
    
    # Parse JSON
    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"[v2] JSON parse error: {e}")
        print(f"[v2] Raw content: {content[:500]}")
        raise
    
    segments_json = data.get("segments", [])
    skipped = data.get("skipped", [])
    
    print(f"[v2] Found {len(segments_json)} content segments, skipped {len(skipped)} sections")
    
    # Convert to SegmentResult objects
    results = []
    for seg in segments_json:
        # Skip non-content segments
        if seg.get("segment_type") in ("intro", "outro", "ad"):
            print(f"[v2] Skipping {seg.get('segment_type')}: {seg.get('start_time')} - {seg.get('end_time')}")
            continue
        
        start_ms = parse_time_to_ms(seg.get("start_time", "0:00"))
        end_ms = parse_time_to_ms(seg.get("end_time", "0:00"))
        
        # Extract transcript text for this segment
        segment_words = [
            w.word for w in transcript.words
            if w.start_ms >= start_ms and w.end_ms <= end_ms
        ]
        transcript_text = " ".join(segment_words)
        
        # Parse topic tags from topic field
        topic = seg.get("topic", "")
        topic_tags = [t.strip() for t in topic.split(",") if t.strip()]
        if not topic_tags and topic:
            topic_tags = [topic]
        
        duration_min = (end_ms - start_ms) / 60000
        print(f"[v2] Segment: {seg.get('start_time')} - {seg.get('end_time')} ({duration_min:.1f}m) - {topic}")
        
        results.append(SegmentResult(
            start_ms=start_ms,
            end_ms=end_ms,
            summary=seg.get("summary", ""),
            topic_tags=topic_tags,
            density_score=float(seg.get("density_score", 0.7)),
            transcript_text=transcript_text,
            segment_type=seg.get("segment_type", "content"),
        ))
    
    # Validate - check for fixed-interval problem
    if len(results) >= 3:
        durations = [(r.end_ms - r.start_ms) for r in results]
        if len(set(durations)) == 1:
            print(f"[v2] WARNING: All segments same duration ({durations[0]/60000:.1f}m) - may be fixed intervals!")
    
    return results


# Test function
if __name__ == "__main__":
    import sys
    
    # Test with a sample transcript
    print("Segmentation v2 module loaded")
    print("Use segment_transcript_v2(transcript) to segment")
