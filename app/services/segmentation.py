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
    topic_tags: List[str]  # kept for backward compat, now = [primary] + secondary
    density_score: float
    transcript_text: str
    primary_topic: str = ""
    secondary_topics: List[str] = field(default_factory=list)
    content_type: str = "content"  # content | ad | intro | outro


def snap_to_sentence_boundaries(
    start_ms: int, 
    end_ms: int, 
    words: List[TranscriptWord],
    max_adjust_ms: int = 5000  # Max 5 seconds adjustment
) -> Tuple[int, int, str]:
    """
    Adjust segment boundaries to align with sentence boundaries.
    Returns (adjusted_start_ms, adjusted_end_ms, adjustment_info).
    """
    if not words:
        return start_ms, end_ms, "no_words"
    
    # Sentence-ending punctuation
    SENTENCE_END = {'.', '!', '?'}
    
    # Find words in the segment
    segment_words = [w for w in words if w.start_ms >= start_ms - max_adjust_ms and w.end_ms <= end_ms + max_adjust_ms]
    if not segment_words:
        return start_ms, end_ms, "no_nearby_words"
    
    adjusted_start = start_ms
    adjusted_end = end_ms
    adjustments = []
    
    # ADJUST START: Find nearest sentence start (word after sentence-ending punctuation)
    # Look backwards from start_ms to find a sentence boundary
    words_before_start = [w for w in segment_words if w.end_ms <= start_ms + 1000]  # Words ending near/before start
    for i in range(len(words_before_start) - 1, -1, -1):
        word = words_before_start[i]
        # Check if this word ends with sentence punctuation
        if any(word.word.rstrip().endswith(p) for p in SENTENCE_END):
            # Next word is the sentence start
            if i + 1 < len(words_before_start):
                next_word = words_before_start[i + 1]
                if abs(next_word.start_ms - start_ms) <= max_adjust_ms:
                    adjusted_start = next_word.start_ms
                    adjustments.append(f"start_snapped:{start_ms}->{adjusted_start}")
                    break
            else:
                # Check words after start
                words_after = [w for w in segment_words if w.start_ms > start_ms]
                if words_after and abs(words_after[0].start_ms - start_ms) <= max_adjust_ms:
                    adjusted_start = words_after[0].start_ms
                    adjustments.append(f"start_snapped:{start_ms}->{adjusted_start}")
                    break
            break
    
    # ADJUST END: Find nearest sentence end (word with sentence-ending punctuation)
    words_near_end = [w for w in segment_words if w.start_ms >= end_ms - 2000]  # Words starting near end
    for word in words_near_end:
        if any(word.word.rstrip().endswith(p) for p in SENTENCE_END):
            if abs(word.end_ms - end_ms) <= max_adjust_ms:
                adjusted_end = word.end_ms
                adjustments.append(f"end_snapped:{end_ms}->{adjusted_end}")
                break
    
    # If we couldn't find a sentence end forward, try looking backward
    if adjusted_end == end_ms:
        words_before_end = [w for w in segment_words if w.end_ms <= end_ms + 500]
        for word in reversed(words_before_end):
            if any(word.word.rstrip().endswith(p) for p in SENTENCE_END):
                if abs(word.end_ms - end_ms) <= max_adjust_ms:
                    adjusted_end = word.end_ms
                    adjustments.append(f"end_snapped_back:{end_ms}->{adjusted_end}")
                    break
    
    adjustment_info = "|".join(adjustments) if adjustments else "no_adjustment"
    return adjusted_start, adjusted_end, adjustment_info


SEGMENTATION_PROMPT = """You are analyzing a podcast transcript to identify SHORT, FOCUSED segments (atomic ideas).

CRITICAL: SKIP NON-CONTENT
First, identify and EXCLUDE:
- Sponsor reads / advertisements ("this episode brought to you by", "use code", "thanks to our sponsors")
- Show intros with theme music descriptions
- Outros, credits, calls to subscribe/review
- Extended banter with no substance

Only segment ACTUAL CONTENT discussions.

SEGMENT LENGTH: TARGET 1-3 MINUTES
- Prefer shorter, focused segments over long rambling ones
- A 10-minute discussion should become 3-5 segments, not 1
- Break at natural topic shifts, even within a conversation
- Each segment = ONE focused idea or topic

TOPIC TAGGING: PRIMARY + SECONDARY
For each segment, identify:
- primary_topic: The MAIN thing being discussed (2-5 words, specific)
- secondary_topics: OTHER things mentioned/discussed (entities, examples, tangents)

Example: A segment primarily about "AI replacing jobs" might mention hotels, airlines, customer service
- primary_topic: "AI job displacement fears"
- secondary_topics: ["hotel industry automation", "airline customer service", "call center jobs"]

This allows searching for "hotel" to find segments where hotels were discussed, even if not the main topic.

CONTENT TYPE LABELING
Mark each segment:
- "content" = actual discussion (will be indexed)
- "ad" = sponsor read, advertisement (will be skipped)
- "intro" = show intro, theme music, housekeeping (will be skipped)  
- "outro" = closing remarks, subscribe prompts (will be skipped)

QUOTE ANCHORS (CRITICAL - READ CAREFULLY)
You must define WHERE each segment starts and ends using exact quotes from the transcript.

starts_with = The FIRST words of a SENTENCE where this topic BEGINS (MUST start at sentence boundary!)
ends_with = The LAST words of a SENTENCE spoken BEFORE they move to a DIFFERENT topic

SENTENCE BOUNDARIES ARE MANDATORY:
- starts_with MUST be the beginning of a sentence (after a period, question mark, or clear topic transition)
- NEVER pick words from the middle of a sentence
- Look for phrases like "So...", "Now...", "Let me...", "The thing about...", "What I think is..."

IMPORTANT: These quotes define the BOUNDARIES of the ENTIRE discussion, not just one sentence!

WRONG approach (mid-sentence start):
- Topic is "Why Acquired podcast succeeded" (discussed for 2 minutes)
- starts_with: "we've received a bunch of" ← BAD! This is mid-sentence!
- ends_with: "99% of podcasts do not" 
- Result: Clip starts awkwardly mid-sentence, confusing the listener!

CORRECT approach (sentence boundary start):
- Topic is "Why Acquired podcast succeeded" (discussed for 2 minutes)  
- starts_with: "So let me tell you why" ← GOOD! Starts a new sentence/thought
- ends_with: "and that's the real secret." ← GOOD! Ends at sentence boundary
- Result: Clean entry point, listener hears complete thought!

Think of it like this:
- starts_with = "Press play here"
- ends_with = "Stop playback here"
- The listener should hear the COMPLETE discussion of this topic

Copy 4-6 words EXACTLY as they appear in the transcript (punctuation, capitalization may vary).

Return JSON array:
[
  {{
    "content_type": "content",
    "primary_topic": "Specific main topic in 2-5 words",
    "secondary_topics": ["other thing mentioned", "another entity discussed", "example given"],
    "summary": "2-3 sentences capturing the SPECIFIC insight. Include names, numbers, concrete details.",
    "density_score": 0.8,
    "starts_with": "So let me explain why",
    "ends_with": "and that changed everything for us"
  }},
  {{
    "content_type": "content",
    "primary_topic": "Warren Buffett investment philosophy",
    "secondary_topics": ["Berkshire Hathaway", "value investing", "long-term thinking"],
    "summary": "Discussion of Buffett's approach to identifying undervalued companies and holding them indefinitely.",
    "density_score": 0.9,
    "starts_with": "When you look at Buffett",
    "ends_with": "that's his entire strategy in a nutshell"
  }},
  {{
    "content_type": "ad",
    "primary_topic": "Sponsor: Athletic Greens",
    "secondary_topics": [],
    "summary": "Advertisement for Athletic Greens supplement.",
    "density_score": 0.0,
    "starts_with": "This episode is brought to",
    "ends_with": "now back to the show"
  }}
]

REMEMBER: starts_with and ends_with must capture the FULL topic discussion, typically 1-3 minutes of audio. If your segment would be less than 30 seconds, you probably picked quotes that are too close together!

BAD TAGS: "AI", "technology", "business", "interview", "discussion"
GOOD TAGS: "GPT-4 capabilities demo", "hotel cleanliness standards", "Y Combinator interview tips"

TRANSCRIPT:
{transcript}

Return ONLY valid JSON, no other text."""


def resolve_timestamps(
    transcript_words: List[TranscriptWord],
    starts_with: str,
    ends_with: str
) -> Tuple[Optional[int], Optional[int], str]:
    """
    Find timestamps by locating quote anchors in the word-level transcript.
    
    Args:
        transcript_words: List of TranscriptWord with word, start_ms, end_ms
        starts_with: Quote anchor for segment start
        ends_with: Quote anchor for segment end
        
    Returns:
        (start_ms, end_ms, status) where status is 'exact', 'fuzzy', or 'failed'
    """
    if not transcript_words or not starts_with or not ends_with:
        return None, None, "failed"
    
    # Build full transcript text with word positions
    words_list = [w.word for w in transcript_words]
    full_text = " ".join(words_list).lower()
    
    # Normalize the quotes
    starts_with_lower = starts_with.lower().strip()
    ends_with_lower = ends_with.lower().strip()
    
    # Try exact match first
    start_ms, start_status = _find_quote_position(transcript_words, starts_with_lower, find_end=False)
    end_ms, end_status = _find_quote_position(transcript_words, ends_with_lower, find_end=True)
    
    if start_ms is not None and end_ms is not None:
        # Validate - end must be after start
        if end_ms > start_ms:
            status = "exact" if start_status == "exact" and end_status == "exact" else "fuzzy"
            return start_ms, end_ms, status
    
    # If we got start but not end, or vice versa, try fuzzy
    if start_ms is None:
        start_ms, start_status = _fuzzy_find_quote(transcript_words, starts_with_lower, find_end=False)
    if end_ms is None:
        end_ms, end_status = _fuzzy_find_quote(transcript_words, ends_with_lower, find_end=True)
    
    if start_ms is not None and end_ms is not None and end_ms > start_ms:
        return start_ms, end_ms, "fuzzy"
    
    return None, None, "failed"


def _normalize_word(word: str) -> str:
    """
    Aggressively normalize a word for matching.
    
    Removes all punctuation, lowercases, and handles common variations.
    """
    import re
    # Remove all non-alphanumeric characters
    word = re.sub(r'[^a-z0-9]', '', word.lower())
    return word


def _find_quote_position(
    words: List[TranscriptWord],
    quote: str,
    find_end: bool = False
) -> Tuple[Optional[int], str]:
    """
    Find position of a quote in the transcript.
    
    Args:
        words: Transcript words
        quote: Quote to find (lowercase)
        find_end: If True, return end_ms of last word; otherwise start_ms of first word
        
    Returns:
        (timestamp_ms, status) where status is 'exact' or 'failed'
    """
    quote_words = [_normalize_word(w) for w in quote.split() if _normalize_word(w)]
    if not quote_words:
        return None, "failed"
    
    # Pre-normalize all transcript words for faster comparison
    normalized_words = [_normalize_word(w.word) for w in words]
    
    # Slide through transcript looking for match
    for i in range(len(words) - len(quote_words) + 1):
        match = True
        for j, qw in enumerate(quote_words):
            if normalized_words[i + j] != qw:
                match = False
                break
        
        if match:
            if find_end:
                return words[i + len(quote_words) - 1].end_ms, "exact"
            else:
                return words[i].start_ms, "exact"
    
    return None, "failed"


def _fuzzy_find_quote(
    words: List[TranscriptWord],
    quote: str,
    find_end: bool = False,
    min_overlap: float = 0.5
) -> Tuple[Optional[int], str]:
    """
    Fuzzy match a quote allowing for partial matches.
    
    Uses multiple strategies:
    1. Sliding window with word-level matching (allowing gaps for filler words)
    2. Consecutive word sequence matching (ignoring filler words like "uh", "um")
    
    Args:
        words: Transcript words
        quote: Quote to find (lowercase)
        find_end: If True, return end_ms of last word; otherwise start_ms of first word
        min_overlap: Minimum fraction of quote words that must match
        
    Returns:
        (timestamp_ms, status) where status is 'fuzzy' or 'failed'
    """
    import difflib
    
    # Normalize quote words
    quote_words = [_normalize_word(w) for w in quote.split() if _normalize_word(w)]
    if not quote_words:
        return None, "failed"
    
    # Filter out common filler words for matching
    filler_words = {'uh', 'um', 'ah', 'like', 'you', 'know', 'yeah', 'so', 'well', 'i', 'mean'}
    content_quote_words = [w for w in quote_words if w not in filler_words]
    
    # Use content words if we have enough, otherwise use all
    if len(content_quote_words) >= 2:
        search_words = content_quote_words
    else:
        search_words = quote_words
    
    required_matches = max(2, int(len(search_words) * min_overlap))
    best_match_count = 0
    best_position = None
    best_window_end = None
    
    # Pre-normalize all transcript words
    normalized_words = [_normalize_word(w.word) for w in words]
    
    # Try different window sizes (allow some slack for filler words)
    max_window = min(len(search_words) + 10, len(words))
    for window_size in range(len(search_words), max_window):
        for i in range(len(words) - window_size + 1):
            window = normalized_words[i:i + window_size]
            
            # Count matching words (in order, allowing gaps)
            matches = 0
            window_idx = 0
            for qw in search_words:
                while window_idx < len(window):
                    if window[window_idx] == qw:
                        matches += 1
                        window_idx += 1
                        break
                    window_idx += 1
            
            if matches > best_match_count and matches >= required_matches:
                best_match_count = matches
                best_window_end = i + window_size - 1
                if find_end:
                    best_position = words[i + window_size - 1].end_ms
                else:
                    best_position = words[i].start_ms
    
    if best_position is not None:
        return best_position, "fuzzy"
    
    # Strategy 2: Use difflib SequenceMatcher for approximate string matching
    # Build text windows and compare
    window_text_size = len(' '.join(search_words)) + 20  # chars
    full_text_normalized = ' '.join(normalized_words)
    quote_text = ' '.join(search_words)
    
    best_ratio = 0.0
    best_idx = None
    
    # Slide through transcript text looking for best match
    for i in range(0, len(full_text_normalized) - len(quote_text), 10):  # step by 10 chars
        window_text = full_text_normalized[i:i + len(quote_text) + 20]
        ratio = difflib.SequenceMatcher(None, quote_text, window_text).ratio()
        if ratio > best_ratio and ratio > 0.6:  # 60% similarity threshold
            best_ratio = ratio
            best_idx = i
    
    if best_idx is not None:
        # Find the word index corresponding to this character position
        char_count = 0
        word_idx = 0
        for idx, nw in enumerate(normalized_words):
            if char_count >= best_idx:
                word_idx = idx
                break
            char_count += len(nw) + 1  # +1 for space
        
        if find_end:
            # Estimate end position
            end_idx = min(word_idx + len(search_words) + 3, len(words) - 1)
            return words[end_idx].end_ms, "fuzzy"
        else:
            return words[word_idx].start_ms, "fuzzy"
    
    return None, "failed"


def validate_segment_duration(start_ms: int, end_ms: int, min_sec: int = 30, max_sec: int = 300) -> Tuple[bool, str]:
    """
    Validate segment duration is within acceptable range.
    
    Args:
        start_ms, end_ms: Segment boundaries
        min_sec: Minimum duration in seconds (default 30)
        max_sec: Maximum duration in seconds (default 300 = 5 min)
        
    Returns:
        (is_valid, message)
    """
    duration_ms = end_ms - start_ms
    duration_sec = duration_ms / 1000
    
    if duration_sec < min_sec:
        return False, f"Too short: {duration_sec:.1f}s < {min_sec}s minimum"
    if duration_sec > max_sec:
        return False, f"Too long: {duration_sec:.1f}s > {max_sec}s maximum"
    
    return True, f"OK: {duration_sec:.1f}s"


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
    # For long transcripts, chunk into ~15 minute pieces
    CHUNK_DURATION_MS = 15 * 60 * 1000  # 15 minutes
    MAX_CHUNK_CHARS = 40000  # Safety limit
    
    total_duration = transcript.words[-1].end_ms if transcript.words else 0
    
    if total_duration <= CHUNK_DURATION_MS * 1.5:
        # Short enough to process in one go
        return await _segment_single_chunk(transcript, episode_title)
    
    # Chunk the transcript
    all_segments = []
    chunk_start_ms = 0
    
    while chunk_start_ms < total_duration:
        chunk_end_ms = min(chunk_start_ms + CHUNK_DURATION_MS, total_duration)
        
        # Get words for this chunk
        chunk_words = [w for w in transcript.words 
                      if w.start_ms >= chunk_start_ms and w.start_ms < chunk_end_ms]
        
        if not chunk_words:
            chunk_start_ms = chunk_end_ms
            continue
        
        # Create chunk transcript
        chunk_transcript = TranscriptResult(
            text=' '.join(w.word for w in chunk_words),
            words=chunk_words,
            duration_ms=chunk_end_ms - chunk_start_ms,
            speakers=[]
        )
        
        print(f"[DEBUG] Processing chunk {chunk_start_ms//60000}m - {chunk_end_ms//60000}m ({len(chunk_words)} words)")
        
        # Segment this chunk
        try:
            chunk_segments = await _segment_single_chunk(chunk_transcript, episode_title)
            all_segments.extend(chunk_segments)
            print(f"[DEBUG] Chunk yielded {len(chunk_segments)} segments")
        except Exception as e:
            print(f"[DEBUG] Chunk failed: {e}")
        
        chunk_start_ms = chunk_end_ms
    
    return all_segments


async def _segment_single_chunk(
    transcript: TranscriptResult,
    episode_title: str = "",
) -> List[SegmentResult]:
    """Segment a single chunk of transcript."""
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
    anomalies = []
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
            
            # Handle new format (primary_topic + secondary_topics) and old format (topic_tags)
            primary_topic = seg.get("primary_topic", "")
            secondary_topics = seg.get("secondary_topics", [])
            content_type = seg.get("content_type", "content")
            
            # Skip non-content segments (ads, intros, outros)
            if content_type != "content":
                print(f"[DEBUG] Skipping {content_type} segment: {primary_topic}")
                continue
            
            # Get quote anchors (new format) or fall back to timestamps (old format)
            starts_with = seg.get("starts_with", "")
            ends_with = seg.get("ends_with", "")
            
            if starts_with and ends_with:
                # Use quote-anchored timestamp resolution
                start_ms, end_ms, status = resolve_timestamps(
                    transcript.words, starts_with, ends_with
                )
                
                if start_ms is None or end_ms is None:
                    print(f"[DEBUG] Failed to resolve timestamps for segment {i}")
                    print(f"[DEBUG]   starts_with: '{starts_with}'")
                    print(f"[DEBUG]   ends_with: '{ends_with}'")
                    anomalies.append({
                        "segment_idx": i,
                        "primary_topic": primary_topic,
                        "starts_with": starts_with,
                        "ends_with": ends_with,
                        "error": "timestamp_resolution_failed"
                    })
                    continue
                
                print(f"[DEBUG] Segment {i}: resolved timestamps ({status}): {start_ms}ms - {end_ms}ms")
                
                # SNAP TO SENTENCE BOUNDARIES
                orig_start, orig_end = start_ms, end_ms
                start_ms, end_ms, snap_info = snap_to_sentence_boundaries(
                    start_ms, end_ms, transcript.words
                )
                if snap_info != "no_adjustment":
                    print(f"[DEBUG] Segment {i}: snapped to sentence boundaries: {snap_info}")
                
                # Validate duration
                is_valid, msg = validate_segment_duration(start_ms, end_ms)
                if not is_valid:
                    print(f"[DEBUG] Segment {i} duration anomaly: {msg}")
                    anomalies.append({
                        "segment_idx": i,
                        "primary_topic": primary_topic,
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "duration_sec": (end_ms - start_ms) / 1000,
                        "error": msg
                    })
                    # Still include it but flag it
            else:
                # Fall back to direct timestamps (old format / backward compat)
                start_ms = seg.get("start_ms", 0)
                end_ms = seg.get("end_ms", 0)
                print(f"[DEBUG] Segment {i}: using direct timestamps (legacy): {start_ms}ms - {end_ms}ms")
            
            # Extract transcript text for this segment
            segment_words = [
                w.word for w in transcript.words
                if w.start_ms >= start_ms and w.end_ms <= end_ms
            ]
            transcript_text = " ".join(segment_words)
            
            # Build topic_tags for backward compat: [primary] + secondary
            if primary_topic:
                topic_tags = [primary_topic] + secondary_topics
            else:
                topic_tags = seg.get("topic_tags", [])
            
            results.append(SegmentResult(
                start_ms=start_ms,
                end_ms=end_ms,
                summary=seg.get("summary", ""),
                topic_tags=topic_tags,
                density_score=float(seg.get("density_score", 0.5)),
                transcript_text=transcript_text,
                primary_topic=primary_topic,
                secondary_topics=secondary_topics,
                content_type=content_type,
            ))
        except Exception as e:
            print(f"[DEBUG] Error parsing segment {i}: {e}")
            print(f"[DEBUG] Segment data: {seg}")
            raise
    
    if anomalies:
        print(f"[DEBUG] {len(anomalies)} anomalies detected during parsing:")
        for a in anomalies:
            print(f"[DEBUG]   - {a}")
    
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
