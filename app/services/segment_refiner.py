"""
Segment Refiner: LLM-based boundary detection and snippet generation.

At index time:
1. Takes initial segments (from embedding-based clustering)
2. Asks LLM to find precise topic start
3. Generates specific snippet for each segment

Run once per episode. Results cached for instant search.
"""

import openai
from typing import List, Tuple, Optional
from dataclasses import dataclass
from app.config import settings


@dataclass
class RefinedSegment:
    """A segment with LLM-refined boundaries and snippet."""
    start_ms: int
    end_ms: int
    original_start_ms: int  # Before refinement
    text: str
    snippet: str  # LLM-generated specific summary
    sentence_indices: List[int]  # Which sentences this covers


def get_context_window(
    sentences: List,
    segment_start_idx: int,
    segment_end_idx: int,
    context_sentences: int = 5
) -> Tuple[str, int, int]:
    """
    Get transcript context around a segment for LLM analysis.
    
    Returns:
        (context_text, context_start_idx, context_end_idx)
    """
    # Include some sentences before and after
    ctx_start = max(0, segment_start_idx - context_sentences)
    ctx_end = min(len(sentences), segment_end_idx + context_sentences)
    
    lines = []
    for i in range(ctx_start, ctx_end):
        sent = sentences[i]
        # Mark the segment boundaries
        marker = ""
        if i == segment_start_idx:
            marker = " [SEGMENT START]"
        if i == segment_end_idx - 1:
            marker = " [SEGMENT END]"
        
        # Format timestamp
        mins = sent.start_ms // 60000
        secs = (sent.start_ms // 1000) % 60
        lines.append(f"[{mins}:{secs:02d}]{marker} {sent.text}")
    
    return "\n".join(lines), ctx_start, ctx_end


def refine_segment_boundary(
    sentences: List,
    segment_start_idx: int,
    segment_end_idx: int,
    model: str = "gpt-4o-mini"
) -> int:
    """
    Ask LLM to find where the topic actually begins.
    
    Returns:
        New start sentence index
    """
    context_text, ctx_start, ctx_end = get_context_window(
        sentences, segment_start_idx, segment_end_idx
    )
    
    prompt = f"""Analyze this podcast transcript excerpt. I've marked a segment that discusses a topic.

Your task: Find where this topic ACTUALLY begins. Look for:
- The question or statement that introduces the topic
- The natural conversation transition into this subject
- Don't cut into the middle of a thought

Transcript:
{context_text}

The segment currently starts at [SEGMENT START]. Should it start earlier to capture the topic introduction?

Reply with ONLY the timestamp (e.g., "2:45") where the topic truly begins. If the current start is correct, reply with the [SEGMENT START] timestamp."""

    client = openai.OpenAI(api_key=settings.openai_api_key)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You analyze podcast transcripts to find topic boundaries. Reply only with timestamps."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=20
        )
        
        timestamp_str = response.choices[0].message.content.strip()
        
        # Parse timestamp like "2:45" or "12:30"
        if ":" in timestamp_str:
            parts = timestamp_str.replace("[", "").replace("]", "").split(":")
            mins = int(parts[0])
            secs = int(parts[1])
            target_ms = (mins * 60 + secs) * 1000
            
            # Find sentence closest to this timestamp
            for i in range(ctx_start, segment_end_idx):
                if sentences[i].start_ms >= target_ms:
                    return max(ctx_start, i - 1)  # Return sentence just before
            
        return segment_start_idx  # Fallback to original
        
    except Exception as e:
        print(f"Error refining boundary: {e}")
        return segment_start_idx


def generate_specific_snippet(
    segment_text: str,
    model: str = "gpt-4o-mini"
) -> str:
    """
    Generate a specific, descriptive snippet for a segment.
    
    NOT: "Discussion about AI"
    YES: "Why chatbots dominate AI today and how agentic AI will change everything"
    """
    # Truncate if too long
    text_sample = segment_text[:3000] if len(segment_text) > 3000 else segment_text
    
    prompt = f"""Summarize this podcast segment in ONE specific sentence.

BE SPECIFIC:
- Include specific claims, names, numbers mentioned
- Describe WHAT is said, not just the topic
- Avoid vague phrases like "discussion about" or "talks about"

BAD: "Discussion about AI and technology"
GOOD: "Why ChatGPT-style chatbots are just the first phase of AI, and how AI agents will replace them within 2 years"

BAD: "The hosts talk about the economy"  
GOOD: "Analysis of why the dollar dropped 5% this week and what it means for inflation"

Segment:
{text_sample}

Write ONE specific summary sentence:"""

    client = openai.OpenAI(api_key=settings.openai_api_key)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You write specific, informative summaries. Never use vague language."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=100
        )
        
        snippet = response.choices[0].message.content.strip()
        # Remove quotes if present
        snippet = snippet.strip('"').strip("'")
        return snippet
        
    except Exception as e:
        print(f"Error generating snippet: {e}")
        return segment_text[:200] + "..."


def refine_segments(
    sentences: List,
    initial_segments: List[Tuple[int, int]],  # List of (start_idx, end_idx)
    model: str = "gpt-4o-mini"
) -> List[RefinedSegment]:
    """
    Refine all segments with LLM.
    
    Args:
        sentences: List of Sentence objects
        initial_segments: List of (start_sentence_idx, end_sentence_idx) tuples
        model: LLM model to use
    
    Returns:
        List of RefinedSegment with improved boundaries and snippets
    """
    refined = []
    
    for start_idx, end_idx in initial_segments:
        # Get original start time
        original_start_ms = sentences[start_idx].start_ms
        
        # Refine the start boundary
        new_start_idx = refine_segment_boundary(sentences, start_idx, end_idx, model)
        
        # Build segment text
        segment_sentences = sentences[new_start_idx:end_idx]
        segment_text = " ".join(s.text for s in segment_sentences)
        
        # Generate specific snippet
        snippet = generate_specific_snippet(segment_text, model)
        
        refined.append(RefinedSegment(
            start_ms=sentences[new_start_idx].start_ms,
            end_ms=sentences[end_idx - 1].end_ms,
            original_start_ms=original_start_ms,
            text=segment_text,
            snippet=snippet,
            sentence_indices=list(range(new_start_idx, end_idx))
        ))
        
        print(f"Segment refined: {original_start_ms//60000}:{(original_start_ms//1000)%60:02d} → {sentences[new_start_idx].start_ms//60000}:{(sentences[new_start_idx].start_ms//1000)%60:02d}")
        print(f"  Snippet: {snippet[:100]}...")
    
    return refined


def detect_topic_boundaries(
    sentences: List,
    similarity_threshold: float = 0.5,
    min_segment_sentences: int = 5
) -> List[Tuple[int, int]]:
    """
    Detect topic boundaries using embedding similarity.
    
    When similarity between adjacent sentences drops below threshold,
    that's likely a topic change.
    
    Returns:
        List of (start_idx, end_idx) tuples for each segment
    """
    import numpy as np
    
    if len(sentences) < 2:
        return [(0, len(sentences))]
    
    boundaries = [0]  # Start of first segment
    
    for i in range(1, len(sentences)):
        prev_emb = sentences[i-1].embedding
        curr_emb = sentences[i].embedding
        
        if prev_emb is None or curr_emb is None:
            continue
        
        # Compute cosine similarity
        prev_norm = prev_emb / np.linalg.norm(prev_emb)
        curr_norm = curr_emb / np.linalg.norm(curr_emb)
        similarity = np.dot(prev_norm, curr_norm)
        
        if similarity < similarity_threshold:
            # Check minimum segment length
            if i - boundaries[-1] >= min_segment_sentences:
                boundaries.append(i)
    
    # Create segment tuples
    segments = []
    for i in range(len(boundaries)):
        start = boundaries[i]
        end = boundaries[i + 1] if i + 1 < len(boundaries) else len(sentences)
        if end - start >= min_segment_sentences:
            segments.append((start, end))
    
    return segments


# Test function
if __name__ == "__main__":
    print("Segment refiner module loaded")
    print("Use refine_segments() to process segments with LLM")
