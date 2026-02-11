"""
Optimized topic segmentation using GPT-4o-mini.
Version 2.1: Commercial product-ready with optimized prompt.

Prompt engineering based on 2026-02-10 session findings:
1. Merge-biased (default don't split)
2. Multi-criteria (must pass all 4 tests)
3. Intent-focused (listener's perspective)
4. Quality-gated (substance check)
5. Chunked processing for long episodes
"""

import json
import re
import openai
from typing import List, Optional, Tuple, Set
from dataclasses import dataclass, field

from app.config import settings
from app.services.sentence_parser import Sentence
from app.services.ner_extraction import merge_entities, ExtractedEntities


def validate_summary_against_text(summary: str, full_text: str) -> Tuple[bool, List[str]]:
    """
    Validate that entities in summary actually appear in the transcript.
    
    Returns:
        (is_valid, list_of_hallucinated_terms)
    """
    # Normalize texts for comparison
    full_text_lower = full_text.lower()
    
    # Common summary style words to ignore (not entities)
    skip_words = {
        # Structural words
        'The', 'A', 'An', 'This', 'That', 'These', 'Those',
        'Discussion', 'Why', 'How', 'What', 'When', 'Where', 'Who',
        'Analysis', 'Overview', 'Introduction', 'Conclusion',
        'Exploration', 'Conversation', 'Reflections', 'Insights',
        'Examination', 'Review', 'Commentary', 'Debate', 'Talk',
        # Common descriptors
        'U', 'S', 'US', 'UK', 'AI', 'CEO', 'GDP', 'IPO',
    }
    
    # Extract potential proper nouns/entities from summary
    words = summary.split()
    potential_entities: Set[str] = set()
    
    for i, word in enumerate(words):
        # Clean punctuation but keep apostrophes for possessives
        clean_word = re.sub(r'[^\w\s\'-]', '', word)
        # Remove trailing 's for possessives
        if clean_word.endswith("'s"):
            clean_word = clean_word[:-2]
        elif clean_word.endswith("s'"):
            clean_word = clean_word[:-2]
        
        if not clean_word or len(clean_word) < 2:
            continue
            
        # Check if capitalized
        if clean_word[0].isupper():
            if clean_word not in skip_words:
                potential_entities.add(clean_word)
    
    # Also extract multi-word entities (e.g., "Elon Musk", "Model S")
    entity_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
    multi_word = re.findall(entity_pattern, summary)
    potential_entities.update(multi_word)
    
    # Check each entity against full text
    hallucinated = []
    for entity in potential_entities:
        entity_lower = entity.lower()
        
        # Handle possessive variations in text too
        # e.g., "Epstein" should match "epstein's" in text
        found = False
        
        # Direct match
        if entity_lower in full_text_lower:
            found = True
        
        # Check for possessive form in text
        if not found and (entity_lower + "'s") in full_text_lower:
            found = True
        if not found and (entity_lower + "s") in full_text_lower:
            found = True
            
        # For multi-word entities, check if ANY word is present
        if not found:
            words_in_entity = entity_lower.split()
            # Require at least one significant word (>3 chars) to match
            if any(w in full_text_lower for w in words_in_entity if len(w) > 3):
                found = True
        
        if not found:
            hallucinated.append(entity)
    
    return (len(hallucinated) == 0, hallucinated)


def fix_hallucinated_summary(
    summary: str, 
    hallucinated_terms: List[str],
    full_text: str,
    model: str = "gpt-4o-mini"
) -> str:
    """
    Ask LLM to fix a summary by removing hallucinated terms.
    """
    client = openai.OpenAI(api_key=settings.openai_api_key)
    
    fix_prompt = f"""The following summary contains terms that don't appear in the actual transcript.

SUMMARY: {summary}

HALLUCINATED TERMS (not in transcript): {', '.join(hallucinated_terms)}

ACTUAL TRANSCRIPT:
{full_text[:2000]}...

Rewrite the summary to accurately describe the transcript content WITHOUT using the hallucinated terms.
Use only names, companies, and products that actually appear in the transcript.
If the transcript doesn't name something specifically, describe it generically.

Return ONLY the corrected summary sentence, nothing else."""

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You fix summaries to remove hallucinated content. Be accurate to source text."},
            {"role": "user", "content": fix_prompt}
        ],
        temperature=0.2,
        max_tokens=200
    )
    
    return response.choices[0].message.content.strip()


@dataclass
class Topic:
    """A topic segment with summary and metadata."""
    topic_id: str
    episode_id: str
    summary: str
    full_text: str
    start_ms: int
    end_ms: int
    sentence_start_idx: int
    sentence_end_idx: int
    people: List[str] = field(default_factory=list)
    companies: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)


# The optimized prompt for topic segmentation
SEGMENTATION_PROMPT = '''You are segmenting a podcast transcript into MAJOR topic sections.

## When to create a new segment:

Default: DON'T split. Only create a new segment when ALL of these are true:

1. **Subject shift**: The conversation moves to a clearly different subject, person, or question — not just a follow-up point or tangent.

2. **Standalone value**: A listener could bookmark this segment on its own. It's a complete discussion, not a fragment.

3. **Search distinction**: Someone searching for this topic would NOT want it bundled with the adjacent section. They're looking for THIS specifically.

4. **Sufficient substance**: The topic has enough depth to stand alone (typically 3+ minutes). Brief mentions or asides should stay with their parent topic.

If you're unsure whether to split — don't. One cohesive segment is better than two fragments.

## Segment balance:
- Each segment should be roughly 3-10 minutes
- No segment should be over 15 minutes unless it's truly ONE continuous topic
- If you find yourself creating a segment over 15 minutes, double-check: are there natural breakpoints you missed?

## For each segment, provide:
- **start_sentence_idx**: The index of the first sentence (0-indexed)
- **end_sentence_idx**: The index of the last sentence (inclusive)
- **summary**: ONE sentence describing the main discussion.
  - Be SPECIFIC: include names, claims, numbers when they appear in the transcript
  - **CRITICAL: ONLY mention people, companies, or products that ACTUALLY APPEAR in the transcript text above**
  - Do NOT add context from your training data. Do NOT infer entities not explicitly mentioned.
  - If speakers say "the company" without naming it, write "the company" — do NOT guess which company.
  BAD: "Elon Musk discusses Tesla's future" (if neither "Elon" nor "Tesla" appears in text)
  GOOD: "Discussion about shutting down Model S and X production lines" (uses actual words from transcript)
- **keywords**: 2-4 keyword tags that APPEAR in the transcript text

## Quality check before submitting:
- Review each summary: Does every name/company in the summary appear in the transcript? If not, remove it.
- Review segment lengths: Any over 15 minutes that could be split?
- Review overall count: A 60-minute podcast typically has 5-10 segments

## Transcript (sentences {start_idx} to {end_idx}):
{transcript}

Return ONLY valid JSON: {{"segments": [{{...}}, ...]}}'''


def segment_sentences(
    sentences: List[Sentence],
    episode_id: str,
    model: str = "gpt-4o-mini",
    max_sentences_per_chunk: int = 200
) -> List[Topic]:
    """
    Segment sentences into topics using LLM.
    
    For long episodes, processes in overlapping chunks and merges results.
    
    Args:
        sentences: List of parsed sentences with timestamps
        episode_id: Episode identifier
        model: OpenAI model to use
        max_sentences_per_chunk: Max sentences per LLM call
        
    Returns:
        List of Topic objects
    """
    if not sentences:
        return []
    
    # For short episodes, process in one go
    if len(sentences) <= max_sentences_per_chunk:
        return _segment_chunk(sentences, 0, episode_id, model)
    
    # For long episodes, process in chunks with overlap
    all_topics = []
    chunk_size = max_sentences_per_chunk
    overlap = 20  # Overlap to catch topics spanning chunks
    
    chunk_start = 0
    chunk_num = 0
    
    while chunk_start < len(sentences):
        chunk_end = min(chunk_start + chunk_size, len(sentences))
        chunk_sentences = sentences[chunk_start:chunk_end]
        
        print(f"  Processing chunk {chunk_num + 1}: sentences {chunk_start}-{chunk_end-1}")
        
        chunk_topics = _segment_chunk(
            chunk_sentences, 
            chunk_start,  # Offset for global indexing
            episode_id, 
            model
        )
        
        # Merge topics from this chunk
        # Handle overlap: if last topic from prev chunk overlaps with first of this chunk, merge
        if all_topics and chunk_topics:
            last_topic = all_topics[-1]
            first_topic = chunk_topics[0]
            
            # If they overlap in time, merge them
            if first_topic.sentence_start_idx <= last_topic.sentence_end_idx + overlap:
                # Merge: extend last topic to include first topic
                merged = _merge_topics(last_topic, first_topic, sentences)
                all_topics[-1] = merged
                chunk_topics = chunk_topics[1:]  # Remove merged topic
        
        all_topics.extend(chunk_topics)
        
        # Move to next chunk
        chunk_start = chunk_end - overlap
        chunk_num += 1
        
        if chunk_start >= len(sentences) - overlap:
            break
    
    # Renumber topic IDs
    for i, topic in enumerate(all_topics):
        topic.topic_id = f"{episode_id}_t{i}"
    
    return all_topics


def _segment_chunk(
    sentences: List[Sentence],
    global_offset: int,
    episode_id: str,
    model: str
) -> List[Topic]:
    """Segment a single chunk of sentences."""
    
    # Build transcript with sentence indices
    transcript_lines = []
    for i, sent in enumerate(sentences):
        global_idx = global_offset + i
        time_str = _format_time(sent.start_ms)
        # Clean and truncate text
        text = sent.text.replace('&gt;', '>').replace('&lt;', '<')
        text = text[:300] + "..." if len(text) > 300 else text
        transcript_lines.append(f"[{global_idx}] [{time_str}] {text}")
    
    transcript = "\n".join(transcript_lines)
    
    prompt = SEGMENTATION_PROMPT.format(
        transcript=transcript,
        start_idx=global_offset,
        end_idx=global_offset + len(sentences) - 1
    )
    
    client = openai.OpenAI(api_key=settings.openai_api_key)
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": (
                    "You segment podcast transcripts into major topics. "
                    "Be conservative — fewer, better segments. "
                    "Aim for 3-10 minute segments. "
                    "Reply with valid JSON only."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        response_format={"type": "json_object"}
    )
    
    raw = response.choices[0].message.content
    parsed = json.loads(raw)
    
    segments_data = parsed.get("segments", [])
    
    # Build Topic objects
    topics = []
    for i, seg in enumerate(segments_data):
        start_idx = seg.get("start_sentence_idx", global_offset)
        end_idx = seg.get("end_sentence_idx", global_offset + len(sentences) - 1)
        
        # Convert to local indices for this chunk
        local_start = max(0, start_idx - global_offset)
        local_end = min(len(sentences) - 1, end_idx - global_offset)
        
        # Get sentences in this segment
        segment_sentences = sentences[local_start:local_end + 1]
        
        if not segment_sentences:
            continue
        
        # Build full text (clean HTML entities)
        full_text = " ".join(s.text for s in segment_sentences)
        full_text = full_text.replace('&gt;', '>').replace('&lt;', '<')
        
        # Merge entities from all sentences
        entities = merge_entities([
            ExtractedEntities(
                people=s.people,
                companies=s.companies,
                topics=s.topics
            )
            for s in segment_sentences
        ])
        
        # Clean entity lists (remove HTML artifacts)
        def clean_list(lst):
            return [x for x in lst if '&gt;' not in x and '&lt;' not in x and len(x) > 1]
        
        # Validate summary against actual transcript content
        summary = seg.get("summary", "")
        is_valid, hallucinated = validate_summary_against_text(summary, full_text)
        
        if not is_valid:
            print(f"    ⚠️  Hallucination detected in segment {i}: {hallucinated}")
            print(f"       Original: {summary[:80]}...")
            # Attempt to fix the summary
            try:
                summary = fix_hallucinated_summary(summary, hallucinated, full_text, model)
                print(f"       Fixed: {summary[:80]}...")
                # Re-validate
                is_valid_2, hallucinated_2 = validate_summary_against_text(summary, full_text)
                if not is_valid_2:
                    print(f"    ⚠️  Still has issues after fix: {hallucinated_2}")
            except Exception as e:
                print(f"    ❌ Failed to fix summary: {e}")
        
        topics.append(Topic(
            topic_id=f"{episode_id}_t{i}",
            episode_id=episode_id,
            summary=summary,
            full_text=full_text,
            start_ms=segment_sentences[0].start_ms,
            end_ms=segment_sentences[-1].end_ms,
            sentence_start_idx=start_idx,
            sentence_end_idx=end_idx,
            people=clean_list(entities.people)[:10],
            companies=clean_list(entities.companies)[:10],
            keywords=seg.get("keywords", []) + clean_list(entities.topics)[:5]
        ))
    
    return topics


def _merge_topics(topic1: Topic, topic2: Topic, all_sentences: List[Sentence]) -> Topic:
    """Merge two overlapping topics."""
    start_idx = min(topic1.sentence_start_idx, topic2.sentence_start_idx)
    end_idx = max(topic1.sentence_end_idx, topic2.sentence_end_idx)
    
    # Get all sentences in merged range
    segment_sentences = all_sentences[start_idx:end_idx + 1]
    full_text = " ".join(s.text for s in segment_sentences)
    
    # Merge entities
    all_people = set(topic1.people + topic2.people)
    all_companies = set(topic1.companies + topic2.companies)
    all_keywords = list(set(topic1.keywords + topic2.keywords))
    
    # Use the longer summary
    summary = topic1.summary if len(topic1.summary) > len(topic2.summary) else topic2.summary
    
    return Topic(
        topic_id=topic1.topic_id,
        episode_id=topic1.episode_id,
        summary=summary,
        full_text=full_text,
        start_ms=segment_sentences[0].start_ms,
        end_ms=segment_sentences[-1].end_ms,
        sentence_start_idx=start_idx,
        sentence_end_idx=end_idx,
        people=list(all_people)[:10],
        companies=list(all_companies)[:10],
        keywords=all_keywords[:10]
    )


def _format_time(ms: int) -> str:
    """Format milliseconds as MM:SS."""
    total_seconds = ms // 1000
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes}:{seconds:02d}"


if __name__ == "__main__":
    from app.services.sentence_parser import parse_vtt_to_sentences
    from pathlib import Path
    
    # Find a VTT file to test
    vtt_files = list(Path("audio").glob("*.vtt"))
    if not vtt_files:
        print("No VTT files found in audio/")
        exit(1)
    
    vtt_path = vtt_files[0]
    episode_id = vtt_path.stem.replace(".en", "")
    
    print(f"Testing topic segmentation with: {vtt_path}")
    print(f"Episode ID: {episode_id}")
    print("-" * 60)
    
    # Parse sentences
    print("Parsing sentences...")
    sentences = parse_vtt_to_sentences(str(vtt_path))
    print(f"Total sentences: {len(sentences)}")
    
    # Segment into topics
    print("\nSegmenting into topics...")
    topics = segment_sentences(sentences, episode_id)
    
    print(f"\nFound {len(topics)} topics:")
    print("=" * 60)
    
    total_duration = 0
    for topic in topics:
        duration_min = (topic.end_ms - topic.start_ms) / 60000
        total_duration += duration_min
        flag = "⚠️" if duration_min > 15 else "✅"
        print(f"\n{flag} Topic: {topic.summary[:80]}...")
        print(f"   Time: {_format_time(topic.start_ms)} - {_format_time(topic.end_ms)} ({duration_min:.1f} min)")
        print(f"   Sentences: {topic.sentence_start_idx} - {topic.sentence_end_idx}")
        if topic.people:
            print(f"   People: {topic.people[:5]}")
        if topic.companies:
            print(f"   Companies: {topic.companies[:5]}")
        if topic.keywords:
            print(f"   Keywords: {topic.keywords[:5]}")
    
    print(f"\n{'='*60}")
    print(f"Total episode duration: {total_duration:.1f} min")
    print(f"Average segment duration: {total_duration/len(topics):.1f} min")
