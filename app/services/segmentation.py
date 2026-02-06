"""Segmentation service using LLM to identify atomic ideas."""

import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

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
4. Assign 2-4 SPECIFIC topic tags per segment

Rules:
- Segments should not overlap
- Prefer natural break points (topic changes, pauses)
- Higher density = more actionable information, insights, or arguments
- Lower density = small talk, lengthy anecdotes, repeated content

CRITICAL REQUIREMENTS FOR SUMMARIES:
- Write 3-4 sentences capturing the SPECIFIC argument, claim, or insight discussed
- Include names of people, companies, or concepts mentioned
- Include specific numbers, dates, predictions, or claims made
- Generic summaries like "discussion of AI" or "talks about technology" are NOT acceptable
- Each summary must be detailed enough that someone could understand the key insight without listening

CRITICAL REQUIREMENTS FOR TAGS:
- Tags MUST be specific topics, not broad categories
- BAD tags: "AI", "technology", "business", "philosophy", "science"
- GOOD tags: "AGI timeline predictions", "transformer scaling laws", "YC application advice", "Stoic death meditation", "Bitcoin ETF impact"
- Each tag should be 2-5 words describing a specific subtopic

Return JSON array:
[
  {{
    "start_ms": 0,
    "end_ms": 180000,
    "summary": "Detailed 3-4 sentence summary capturing the SPECIFIC argument, claim, or insight discussed. Include names, numbers, and concrete details.",
    "topic_tags": ["specific subtopic 1", "specific subtopic 2"],
    "density_score": 0.8
  }}
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
    
    print(f"[DEBUG] Preparing transcript with {len(transcript.words)} words")
    if transcript.words:
        print(f"[DEBUG] First word type: {type(transcript.words[0])}")
        print(f"[DEBUG] First word: {transcript.words[0]}")
    
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
    print(f"[DEBUG] Calling OpenAI {settings.segmentation_model} with {len(transcript_text)} chars")
    
    try:
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
        )
        
        content = response.choices[0].message.content
        print(f"[DEBUG] OpenAI response length: {len(content)} chars")
        print(f"[DEBUG] OpenAI response preview: {content[:500]}...")
    except Exception as api_error:
        import traceback
        print(f"[DEBUG] OpenAI API call failed: {api_error}")
        print(f"[DEBUG] Exception type: {type(api_error)}")
        print(f"[DEBUG] Full traceback:\n{traceback.format_exc()}")
        raise
    
    # Strip markdown code blocks if present
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    content = content.strip()
    
    # Parse JSON
    try:
        data = json.loads(content)
        print(f"[DEBUG] Parsed JSON type: {type(data)}, keys: {data.keys() if isinstance(data, dict) else 'N/A'}")
        if isinstance(data, dict) and "segments" in data:
            print(f"[DEBUG] Found {len(data['segments'])} segments in response")
            return data["segments"]
        elif isinstance(data, list):
            print(f"[DEBUG] Response is list with {len(data)} items")
            return data
        else:
            raise ValueError(f"Unexpected response format: {type(data)}")
    except json.JSONDecodeError as e:
        print(f"[DEBUG] JSON decode failed: {e}")
        print(f"[DEBUG] Raw content: {content[:1000]}")
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
    print(f"[DEBUG] Parsing {len(segments_json)} segments")
    print(f"[DEBUG] segments_json type: {type(segments_json)}")
    if segments_json:
        print(f"[DEBUG] First segment type: {type(segments_json[0])}")
        print(f"[DEBUG] First segment: {str(segments_json[0])[:200]}")
    
    for i, seg in enumerate(segments_json):
        try:
            print(f"[DEBUG] Processing segment {i}, type: {type(seg)}")
            if isinstance(seg, str):
                print(f"[DEBUG] ERROR: segment is a string, not dict: {seg[:100]}")
                raise ValueError(f"Segment {i} is a string, not dict")
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
        except Exception as e:
            print(f"[DEBUG] Error parsing segment {i}: {e}")
            print(f"[DEBUG] Segment data: {seg}")
            raise
    
    print(f"[DEBUG] Successfully parsed {len(results)} segments")
    return results


@dataclass
class SegmentQualityMetrics:
    """Quality metrics for a set of segments."""
    segment_count: int
    avg_duration_min: float
    duration_std_min: float  # Standard deviation - consistency
    coverage_pct: float  # % of episode covered by segments
    avg_summary_length: float  # Words per summary
    avg_tags_per_segment: float
    unique_tags: int
    density_mean: float
    density_std: float  # Variance in density scores
    specificity_score: float  # 0-1, higher = more specific summaries


@dataclass
class ModelComparisonResult:
    """Results from comparing two models side-by-side."""
    model_a: str
    model_b: str
    segments_a: List[SegmentResult]
    segments_b: List[SegmentResult]
    cost_a: float  # Estimated cost in USD
    cost_b: float
    metrics_a: Optional[SegmentQualityMetrics] = None
    metrics_b: Optional[SegmentQualityMetrics] = None
    boundary_agreement: float = 0.0  # How much segment boundaries overlap
    

async def compare_models(
    transcript: TranscriptResult,
    model_a: str = "gpt-4o",
    model_b: str = "gpt-4o-mini",
) -> ModelComparisonResult:
    """
    Run segmentation with two different models for A/B comparison.
    
    Args:
        transcript: The transcript to segment
        model_a: First model (default: gpt-4o)
        model_b: Second model (default: gpt-4o-mini)
        
    Returns:
        ModelComparisonResult with both outputs and cost estimates
    """
    if not openai:
        raise ValueError("OpenAI not installed")
    if not settings.openai_api_key:
        raise ValueError("OpenAI API key not configured")
    
    transcript_text = _prepare_transcript_with_times(transcript)
    
    # Run both models in parallel
    async def run_model(model: str) -> Tuple[List[Dict], int, int]:
        client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
        response = await client.chat.completions.create(
            model=model,
            messages=[{
                "role": "user",
                "content": SEGMENTATION_PROMPT.format(transcript=transcript_text[:50000])
            }],
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        
        content = response.choices[0].message.content
        usage = response.usage
        
        # Parse response
        try:
            data = json.loads(content)
            if isinstance(data, dict) and "segments" in data:
                segments = data["segments"]
            elif isinstance(data, list):
                segments = data
            else:
                segments = []
        except json.JSONDecodeError:
            segments = []
        
        return segments, usage.prompt_tokens, usage.completion_tokens
    
    # Run both in parallel
    results = await asyncio.gather(
        run_model(model_a),
        run_model(model_b),
        return_exceptions=True
    )
    
    # Process results
    segments_a, tokens_in_a, tokens_out_a = results[0] if not isinstance(results[0], Exception) else ([], 0, 0)
    segments_b, tokens_in_b, tokens_out_b = results[1] if not isinstance(results[1], Exception) else ([], 0, 0)
    
    # Cost estimates (per 1M tokens)
    COSTS = {
        "gpt-4o": {"input": 2.50, "output": 10.00},
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},
        "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    }
    
    def calc_cost(model: str, tokens_in: int, tokens_out: int) -> float:
        rates = COSTS.get(model, COSTS["gpt-4o-mini"])
        return (tokens_in * rates["input"] + tokens_out * rates["output"]) / 1_000_000
    
    cost_a = calc_cost(model_a, tokens_in_a, tokens_out_a)
    cost_b = calc_cost(model_b, tokens_in_b, tokens_out_b)
    
    parsed_a = _parse_segments(segments_a, transcript) if segments_a else []
    parsed_b = _parse_segments(segments_b, transcript) if segments_b else []
    
    # Calculate total episode duration
    total_duration_ms = transcript.words[-1].end_ms if transcript.words else 0
    
    # Calculate quality metrics
    metrics_a = _calculate_metrics(parsed_a, total_duration_ms) if parsed_a else None
    metrics_b = _calculate_metrics(parsed_b, total_duration_ms) if parsed_b else None
    
    # Calculate boundary agreement between models
    boundary_agreement = _calculate_boundary_agreement(parsed_a, parsed_b, total_duration_ms)
    
    return ModelComparisonResult(
        model_a=model_a,
        model_b=model_b,
        segments_a=parsed_a,
        segments_b=parsed_b,
        cost_a=cost_a,
        cost_b=cost_b,
        metrics_a=metrics_a,
        metrics_b=metrics_b,
        boundary_agreement=boundary_agreement,
    )


def _calculate_metrics(segments: List[SegmentResult], total_duration_ms: int) -> SegmentQualityMetrics:
    """Calculate quality metrics for a set of segments."""
    import statistics
    
    if not segments:
        return SegmentQualityMetrics(
            segment_count=0, avg_duration_min=0, duration_std_min=0,
            coverage_pct=0, avg_summary_length=0, avg_tags_per_segment=0,
            unique_tags=0, density_mean=0, density_std=0, specificity_score=0
        )
    
    # Duration metrics
    durations_min = [(s.end_ms - s.start_ms) / 60000 for s in segments]
    avg_duration = statistics.mean(durations_min)
    duration_std = statistics.stdev(durations_min) if len(durations_min) > 1 else 0
    
    # Coverage
    covered_ms = sum(s.end_ms - s.start_ms for s in segments)
    coverage_pct = (covered_ms / total_duration_ms * 100) if total_duration_ms > 0 else 0
    
    # Summary metrics
    summary_lengths = [len(s.summary.split()) for s in segments]
    avg_summary_length = statistics.mean(summary_lengths)
    
    # Tag metrics
    all_tags = [tag for s in segments for tag in s.topic_tags]
    avg_tags = len(all_tags) / len(segments) if segments else 0
    unique_tags = len(set(all_tags))
    
    # Density metrics
    densities = [s.density_score for s in segments]
    density_mean = statistics.mean(densities)
    density_std = statistics.stdev(densities) if len(densities) > 1 else 0
    
    # Specificity score - based on summary detail and tag uniqueness
    # Higher if: longer summaries, more unique tags, varied density scores
    specificity_score = min(1.0, (
        (avg_summary_length / 30) * 0.4 +  # ~30 words is good
        (unique_tags / max(len(segments), 1)) * 0.3 +  # Unique tags per segment
        (density_std * 2) * 0.3  # Variance suggests nuanced scoring
    ))
    
    return SegmentQualityMetrics(
        segment_count=len(segments),
        avg_duration_min=round(avg_duration, 1),
        duration_std_min=round(duration_std, 1),
        coverage_pct=round(coverage_pct, 1),
        avg_summary_length=round(avg_summary_length, 1),
        avg_tags_per_segment=round(avg_tags, 1),
        unique_tags=unique_tags,
        density_mean=round(density_mean, 2),
        density_std=round(density_std, 2),
        specificity_score=round(specificity_score, 2),
    )


def _calculate_boundary_agreement(
    segments_a: List[SegmentResult],
    segments_b: List[SegmentResult],
    total_duration_ms: int
) -> float:
    """
    Calculate how much the segment boundaries agree between two models.
    Returns 0-1 where 1 = perfect agreement.
    """
    if not segments_a or not segments_b or total_duration_ms == 0:
        return 0.0
    
    # Create binary timeline (1 = in segment, 0 = not)
    # Sample at 1-second resolution
    resolution_ms = 1000
    timeline_length = total_duration_ms // resolution_ms + 1
    
    def make_timeline(segments: List[SegmentResult]) -> List[int]:
        timeline = [0] * timeline_length
        for s in segments:
            start_idx = s.start_ms // resolution_ms
            end_idx = min(s.end_ms // resolution_ms, timeline_length - 1)
            for i in range(start_idx, end_idx + 1):
                timeline[i] = 1
        return timeline
    
    timeline_a = make_timeline(segments_a)
    timeline_b = make_timeline(segments_b)
    
    # Calculate agreement (Jaccard similarity)
    both = sum(1 for a, b in zip(timeline_a, timeline_b) if a == 1 and b == 1)
    either = sum(1 for a, b in zip(timeline_a, timeline_b) if a == 1 or b == 1)
    
    return round(both / either, 2) if either > 0 else 0.0
