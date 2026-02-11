"""
Topic segmentation v3: Commercial-grade with natural boundary refinement.

Key improvements over v2:
1. Natural start detection: LLM picks where topic intro REALLY begins
2. Natural end detection: LLM picks where topic REALLY concludes
3. Context window: Shows 3 sentences before/after boundaries for better picks
4. Ending quality: Avoids cutting off mid-thought

Based on v2.1 with added boundary refinement step.
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
    full_text_lower = full_text.lower()
    
    skip_words = {
        'The', 'A', 'An', 'This', 'That', 'These', 'Those',
        'Discussion', 'Why', 'How', 'What', 'When', 'Where', 'Who',
        'Analysis', 'Overview', 'Introduction', 'Conclusion',
        'Exploration', 'Conversation', 'Reflections', 'Insights',
        'Examination', 'Review', 'Commentary', 'Debate', 'Talk',
        'U', 'S', 'US', 'UK', 'AI', 'CEO', 'GDP', 'IPO',
    }
    
    words = summary.split()
    potential_entities: Set[str] = set()
    
    for word in words:
        clean_word = re.sub(r'[^\w\s\'-]', '', word)
        if clean_word.endswith("'s"):
            clean_word = clean_word[:-2]
        elif clean_word.endswith("s'"):
            clean_word = clean_word[:-2]
        
        if not clean_word or len(clean_word) < 2:
            continue
            
        if clean_word[0].isupper() and clean_word not in skip_words:
            potential_entities.add(clean_word)
    
    entity_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
    multi_word = re.findall(entity_pattern, summary)
    potential_entities.update(multi_word)
    
    hallucinated = []
    for entity in potential_entities:
        entity_lower = entity.lower()
        found = False
        
        if entity_lower in full_text_lower:
            found = True
        if not found and (entity_lower + "'s") in full_text_lower:
            found = True
        if not found and (entity_lower + "s") in full_text_lower:
            found = True
        if not found:
            words_in_entity = entity_lower.split()
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
    """Ask LLM to fix a summary by removing hallucinated terms."""
    client = openai.OpenAI(api_key=settings.openai_api_key)
    
    fix_prompt = f"""The following summary contains terms that don't appear in the actual transcript.

SUMMARY: {summary}

HALLUCINATED TERMS (not in transcript): {', '.join(hallucinated_terms)}

ACTUAL TRANSCRIPT:
{full_text[:2000]}...

Rewrite the summary to accurately describe the transcript content WITHOUT using the hallucinated terms.
Use only names, companies, and products that actually appear in the transcript.

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


# Enhanced prompt with natural boundary detection
SEGMENTATION_PROMPT_V3 = '''You are segmenting a podcast transcript into MAJOR topic sections with precise boundaries.

## When to create a new segment:

Default: DON'T split. Only create a new segment when ALL of these are true:

1. **Subject shift**: The conversation moves to a clearly different subject, person, or question — not just a follow-up point or tangent.

2. **Standalone value**: A listener could bookmark this segment on its own. It's a complete discussion, not a fragment.

3. **Search distinction**: Someone searching for this topic would NOT want it bundled with the adjacent section.

4. **Sufficient substance**: The topic has enough depth to stand alone (typically 3+ minutes).

## Segment balance:
- Each segment should be roughly 3-10 minutes
- No segment should be over 15 minutes unless it's truly ONE continuous topic

## CRITICAL: Natural boundary selection

For EACH segment, you must pick boundaries that create a COMPLETE listening experience:

### START boundary rules:
- Include the question or prompt that introduces the topic
- Include transition phrases ("Speaking of...", "Let's talk about...", "So what about...")
- A listener pressing play should immediately understand what's being discussed
- DON'T start mid-answer or mid-thought

### END boundary rules:
- Include the conclusion or wrap-up of the thought
- Include natural pauses or transitions OUT of the topic
- End on a complete thought, not mid-sentence
- Prefer ending on: conclusions, summaries, transition phrases, or natural pauses
- DON'T cut off while someone is still making their point

## For each segment, provide:
- **start_sentence_idx**: The NATURAL starting point (may be 1-3 sentences before the "meat" of the topic)
- **end_sentence_idx**: The NATURAL ending point (may be 1-2 sentences after the main point concludes)
- **summary**: ONE specific sentence describing the main discussion
  - Be SPECIFIC: include names, claims, numbers when they appear
  - ONLY mention entities that ACTUALLY APPEAR in the transcript text
- **keywords**: 2-4 keyword tags that APPEAR in the transcript

## Quality check before submitting:
1. Would a listener pressing play at start_sentence_idx understand the context?
2. Would stopping at end_sentence_idx feel like a natural pause?
3. Does every name/company in the summary appear in the transcript?
4. Are there any 15+ minute segments that could be split?

## Transcript (sentences {start_idx} to {end_idx}):
{transcript}

Return ONLY valid JSON: {{"segments": [{{...}}, ...]}}'''


# Boundary refinement prompt - for fine-tuning after initial segmentation
BOUNDARY_REFINEMENT_PROMPT = '''You are refining segment boundaries for podcast clips. For each segment, I'll show you:
- 3 sentences BEFORE the detected start (context)
- The detected boundary sentence
- 3 sentences AFTER the detected end (context)

Your job: Pick the OPTIMAL start and end for a seamless listening experience.

## Rules for START:
- "before_3", "before_2", "before_1": Choose if the topic intro/question starts earlier
- "boundary": The detected start is already optimal
- Pick the earliest sentence where the topic is clearly being introduced

## Rules for END:
- "boundary": The detected end is already optimal  
- "after_1", "after_2", "after_3": Choose if there's a better conclusion point
- Pick the latest sentence that still belongs to THIS topic before the next begins

{segments_context}

Return JSON array with refined boundaries:
[
  {{"segment": 0, "start": "before_2", "end": "after_1"}},
  {{"segment": 1, "start": "boundary", "end": "boundary"}},
  ...
]'''


def segment_sentences(
    sentences: List[Sentence],
    episode_id: str,
    model: str = "gpt-4o-mini",
    max_sentences_per_chunk: int = 200,
    refine_boundaries: bool = True
) -> List[Topic]:
    """
    Segment sentences into topics using LLM with boundary refinement.
    
    Args:
        sentences: List of parsed sentences with timestamps
        episode_id: Episode identifier
        model: OpenAI model to use
        max_sentences_per_chunk: Max sentences per LLM call
        refine_boundaries: Whether to run boundary refinement pass
        
    Returns:
        List of Topic objects with optimized start/end times
    """
    if not sentences:
        return []
    
    # Phase 1: Initial segmentation
    if len(sentences) <= max_sentences_per_chunk:
        topics = _segment_chunk(sentences, 0, episode_id, model)
    else:
        topics = _segment_long_episode(sentences, episode_id, model, max_sentences_per_chunk)
    
    if not topics:
        return []
    
    # Phase 2: Boundary refinement (optional but recommended)
    if refine_boundaries and len(topics) > 0:
        topics = _refine_boundaries(topics, sentences, model)
    
    # Phase 3: Fix conjunction starts (topics starting with But/And/So)
    if len(topics) > 0:
        topics = _fix_conjunction_starts(topics, sentences)
    
    # Renumber topic IDs
    for i, topic in enumerate(topics):
        topic.topic_id = f"{episode_id}_t{i}"
    
    return topics


def _segment_long_episode(
    sentences: List[Sentence],
    episode_id: str,
    model: str,
    max_sentences_per_chunk: int
) -> List[Topic]:
    """Process long episodes in overlapping chunks."""
    all_topics = []
    chunk_size = max_sentences_per_chunk
    overlap = 20
    
    chunk_start = 0
    chunk_num = 0
    
    while chunk_start < len(sentences):
        chunk_end = min(chunk_start + chunk_size, len(sentences))
        chunk_sentences = sentences[chunk_start:chunk_end]
        
        print(f"  Processing chunk {chunk_num + 1}: sentences {chunk_start}-{chunk_end-1}")
        
        chunk_topics = _segment_chunk(
            chunk_sentences, 
            chunk_start,
            episode_id, 
            model
        )
        
        # Handle overlap merging
        if all_topics and chunk_topics:
            last_topic = all_topics[-1]
            first_topic = chunk_topics[0]
            
            if first_topic.sentence_start_idx <= last_topic.sentence_end_idx + overlap:
                merged = _merge_topics(last_topic, first_topic, sentences)
                all_topics[-1] = merged
                chunk_topics = chunk_topics[1:]
        
        all_topics.extend(chunk_topics)
        
        chunk_start = chunk_end - overlap
        chunk_num += 1
        
        if chunk_start >= len(sentences) - overlap:
            break
    
    return all_topics


def _segment_chunk(
    sentences: List[Sentence],
    global_offset: int,
    episode_id: str,
    model: str
) -> List[Topic]:
    """Segment a single chunk of sentences with enhanced prompt."""
    
    # Build transcript with sentence indices
    transcript_lines = []
    for i, sent in enumerate(sentences):
        global_idx = global_offset + i
        time_str = _format_time(sent.start_ms)
        text = sent.text.replace('&gt;', '>').replace('&lt;', '<')
        text = text[:300] + "..." if len(text) > 300 else text
        transcript_lines.append(f"[{global_idx}] [{time_str}] {text}")
    
    transcript = "\n".join(transcript_lines)
    
    prompt = SEGMENTATION_PROMPT_V3.format(
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
                    "You segment podcast transcripts into major topics with precise boundaries. "
                    "Focus on creating complete listening experiences — natural starts and endings. "
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
        
        local_start = max(0, start_idx - global_offset)
        local_end = min(len(sentences) - 1, end_idx - global_offset)
        
        segment_sentences = sentences[local_start:local_end + 1]
        
        if not segment_sentences:
            continue
        
        full_text = " ".join(s.text for s in segment_sentences)
        full_text = full_text.replace('&gt;', '>').replace('&lt;', '<')
        
        entities = merge_entities([
            ExtractedEntities(
                people=s.people,
                companies=s.companies,
                topics=s.topics
            )
            for s in segment_sentences
        ])
        
        def clean_list(lst):
            return [x for x in lst if '&gt;' not in x and '&lt;' not in x and len(x) > 1]
        
        # Validate and fix hallucinations
        summary = seg.get("summary", "")
        is_valid, hallucinated = validate_summary_against_text(summary, full_text)
        
        if not is_valid:
            print(f"    ⚠️  Hallucination detected in segment {i}: {hallucinated}")
            try:
                summary = fix_hallucinated_summary(summary, hallucinated, full_text, model)
                print(f"       Fixed: {summary[:60]}...")
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


def _refine_boundaries(
    topics: List[Topic],
    sentences: List[Sentence],
    model: str,
    context_size: int = 3
) -> List[Topic]:
    """
    Refine topic boundaries by showing context sentences to LLM.
    
    This is a second pass that fine-tunes where each segment starts/ends
    based on surrounding context.
    """
    print(f"  Refining boundaries for {len(topics)} topics...")
    
    # Build context for each topic
    segments_context_parts = []
    
    for i, topic in enumerate(topics):
        start_idx = topic.sentence_start_idx
        end_idx = topic.sentence_end_idx
        
        # Context before start
        before_lines = []
        for j in range(context_size, 0, -1):
            ctx_idx = start_idx - j
            if 0 <= ctx_idx < len(sentences):
                sent = sentences[ctx_idx]
                before_lines.append(f"    [before_{j}] [{_format_time(sent.start_ms)}] {sent.text[:150]}")
        
        # The boundary sentence (start)
        if 0 <= start_idx < len(sentences):
            sent = sentences[start_idx]
            start_line = f"    [boundary] [{_format_time(sent.start_ms)}] {sent.text[:150]}"
        else:
            start_line = "    [boundary] (missing)"
        
        # Context after end
        after_lines = []
        for j in range(1, context_size + 1):
            ctx_idx = end_idx + j
            if 0 <= ctx_idx < len(sentences):
                sent = sentences[ctx_idx]
                after_lines.append(f"    [after_{j}] [{_format_time(sent.start_ms)}] {sent.text[:150]}")
        
        # The boundary sentence (end)
        if 0 <= end_idx < len(sentences):
            sent = sentences[end_idx]
            end_line = f"    [boundary] [{_format_time(sent.start_ms)}] {sent.text[:150]}"
        else:
            end_line = "    [boundary] (missing)"
        
        segment_block = f"""=== SEGMENT {i}: "{topic.summary[:60]}..." ===
  START CONTEXT:
{chr(10).join(before_lines)}
{start_line}
  
  END CONTEXT:
{end_line}
{chr(10).join(after_lines)}"""
        
        segments_context_parts.append(segment_block)
    
    segments_context = "\n\n".join(segments_context_parts)
    
    prompt = BOUNDARY_REFINEMENT_PROMPT.format(segments_context=segments_context)
    
    client = openai.OpenAI(api_key=settings.openai_api_key)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You refine podcast segment boundaries for optimal listening experience. "
                        "Pick natural start and end points. Reply with valid JSON only."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        
        raw = response.choices[0].message.content
        parsed = json.loads(raw)
        
        # Handle both {"refinements": [...]} and bare [...]
        if isinstance(parsed, dict):
            refinements = parsed.get("refinements", parsed.get("segments", list(parsed.values())[0] if parsed else []))
        else:
            refinements = parsed
        
        # Apply refinements
        adjustments_made = 0
        for ref in refinements:
            seg_idx = ref.get("segment", -1)
            if seg_idx < 0 or seg_idx >= len(topics):
                continue
            
            topic = topics[seg_idx]
            
            # Adjust start
            start_choice = ref.get("start", "boundary")
            if start_choice.startswith("before_"):
                try:
                    n = int(start_choice.split("_")[1])
                    new_start_idx = max(0, topic.sentence_start_idx - n)
                    if new_start_idx != topic.sentence_start_idx:
                        topic.sentence_start_idx = new_start_idx
                        topic.start_ms = sentences[new_start_idx].start_ms
                        adjustments_made += 1
                except (ValueError, IndexError):
                    pass
            
            # Adjust end
            end_choice = ref.get("end", "boundary")
            if end_choice.startswith("after_"):
                try:
                    n = int(end_choice.split("_")[1])
                    new_end_idx = min(len(sentences) - 1, topic.sentence_end_idx + n)
                    if new_end_idx != topic.sentence_end_idx:
                        topic.sentence_end_idx = new_end_idx
                        topic.end_ms = sentences[new_end_idx].end_ms
                        # Update full_text with new range
                        segment_sentences = sentences[topic.sentence_start_idx:topic.sentence_end_idx + 1]
                        topic.full_text = " ".join(s.text for s in segment_sentences)
                        adjustments_made += 1
                except (ValueError, IndexError):
                    pass
        
        print(f"    ✅ Made {adjustments_made} boundary adjustments")
        
    except Exception as e:
        print(f"    ⚠️ Boundary refinement failed: {e}")
    
    return topics


def _fix_conjunction_starts(
    topics: List[Topic],
    sentences: List[Sentence]
) -> List[Topic]:
    """
    Fix topics that start with conjunctions (mid-sentence starts).
    
    If a topic's first sentence starts with But/And/So/However/etc.,
    move the boundary back to include context.
    """
    CONJUNCTIONS = {
        'but', 'and', 'so', 'however', 'therefore', 'thus', 'yet',
        'because', 'since', 'although', 'though', 'while', 'whereas',
        'or', 'nor', 'for', 'also', 'plus', 'then', 'now'
    }
    
    adjustments = 0
    
    for topic in topics:
        start_idx = topic.sentence_start_idx
        if start_idx < 0 or start_idx >= len(sentences):
            continue
            
        first_sentence = sentences[start_idx].text.strip()
        first_word = first_sentence.split()[0].lower().rstrip('.,!?') if first_sentence else ''
        
        # Check if starts with conjunction
        if first_word in CONJUNCTIONS:
            # Move back up to 3 sentences to find context
            new_start = start_idx
            for back in range(1, 4):
                check_idx = start_idx - back
                if check_idx < 0:
                    break
                # Don't cross into previous topic
                prev_topic_end = topics[topics.index(topic) - 1].sentence_end_idx if topics.index(topic) > 0 else -1
                if check_idx <= prev_topic_end:
                    break
                new_start = check_idx
                # Stop if this sentence doesn't start with conjunction
                check_text = sentences[check_idx].text.strip()
                check_word = check_text.split()[0].lower().rstrip('.,!?') if check_text else ''
                if check_word not in CONJUNCTIONS:
                    break
            
            if new_start != start_idx:
                topic.sentence_start_idx = new_start
                topic.start_ms = sentences[new_start].start_ms
                # Rebuild full_text
                segment_sentences = sentences[topic.sentence_start_idx:topic.sentence_end_idx + 1]
                topic.full_text = " ".join(s.text for s in segment_sentences)
                adjustments += 1
    
    if adjustments > 0:
        print(f"    ✅ Fixed {adjustments} conjunction starts")
    
    return topics


def _merge_topics(topic1: Topic, topic2: Topic, all_sentences: List[Sentence]) -> Topic:
    """Merge two overlapping topics."""
    start_idx = min(topic1.sentence_start_idx, topic2.sentence_start_idx)
    end_idx = max(topic1.sentence_end_idx, topic2.sentence_end_idx)
    
    segment_sentences = all_sentences[start_idx:end_idx + 1]
    full_text = " ".join(s.text for s in segment_sentences)
    
    all_people = set(topic1.people + topic2.people)
    all_companies = set(topic1.companies + topic2.companies)
    all_keywords = list(set(topic1.keywords + topic2.keywords))
    
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
    
    vtt_files = list(Path("audio").glob("*.vtt"))
    if not vtt_files:
        print("No VTT files found in audio/")
        exit(1)
    
    vtt_path = vtt_files[0]
    episode_id = vtt_path.stem.replace(".en", "")
    
    print(f"Testing v3 segmentation with: {vtt_path}")
    print(f"Episode ID: {episode_id}")
    print("-" * 60)
    
    sentences = parse_vtt_to_sentences(str(vtt_path))
    print(f"Total sentences: {len(sentences)}")
    
    print("\nSegmenting with boundary refinement...")
    topics = segment_sentences(sentences, episode_id, refine_boundaries=True)
    
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
        if topic.keywords:
            print(f"   Keywords: {topic.keywords[:5]}")
    
    print(f"\n{'='*60}")
    print(f"Total episode duration: {total_duration:.1f} min")
    print(f"Average segment duration: {total_duration/len(topics):.1f} min")
