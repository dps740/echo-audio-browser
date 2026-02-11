"""
Parse VTT files into sentences with timestamps and NER entities.
Combines VTT parsing with spaCy NER extraction.
"""

import re
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass, field

from app.services.ner_extraction import extract_entities_batch, ExtractedEntities


@dataclass
class Sentence:
    """A sentence with timestamp and extracted entities."""
    text: str
    start_ms: int
    end_ms: int
    people: List[str] = field(default_factory=list)
    companies: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)


def parse_vtt_to_sentences(vtt_path: str, pause_threshold_ms: int = 800) -> List[Sentence]:
    """
    Parse VTT file into sentences with timestamps and NER entities.
    
    Args:
        vtt_path: Path to VTT file
        pause_threshold_ms: Minimum pause between words to indicate sentence boundary
        
    Returns:
        List of Sentence objects with timestamps and entities
    """
    vtt_path = Path(vtt_path)
    if not vtt_path.exists():
        raise FileNotFoundError(f"VTT file not found: {vtt_path}")
    
    content = vtt_path.read_text(encoding='utf-8')
    
    # Parse VTT cues - format: "HH:MM:SS.mmm --> HH:MM:SS.mmm"
    # followed by text lines
    cue_pattern = re.compile(
        r'(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})'
    )
    
    # Extract all cues with timestamps
    words_with_times = []
    lines = content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        match = cue_pattern.match(line)
        
        if match:
            start_time = _parse_timestamp(match.group(1))
            end_time = _parse_timestamp(match.group(2))
            
            # Collect text until next timestamp or empty line
            text_lines = []
            i += 1
            while i < len(lines) and lines[i].strip() and not cue_pattern.match(lines[i].strip()):
                text_line = lines[i].strip()
                # Remove VTT formatting tags
                text_line = re.sub(r'<[^>]+>', '', text_line)
                if text_line and not text_line.startswith('align:'):
                    text_lines.append(text_line)
                i += 1
            
            if text_lines:
                text = ' '.join(text_lines)
                words_with_times.append({
                    'text': text,
                    'start_ms': start_time,
                    'end_ms': end_time
                })
        else:
            i += 1
    
    if not words_with_times:
        return []
    
    # Group into sentences based on pauses and punctuation
    sentences_raw = []
    current_sentence = {
        'text': '',
        'start_ms': words_with_times[0]['start_ms'],
        'end_ms': words_with_times[0]['end_ms']
    }
    
    for j, word in enumerate(words_with_times):
        # Check for pause (sentence boundary)
        if j > 0:
            pause = word['start_ms'] - words_with_times[j-1]['end_ms']
            prev_text = current_sentence['text']
            
            # Sentence boundary conditions:
            # 1. Long pause
            # 2. Previous text ends with sentence-ending punctuation
            is_sentence_end = (
                pause > pause_threshold_ms or
                prev_text.rstrip().endswith(('.', '!', '?'))
            )
            
            if is_sentence_end and current_sentence['text'].strip():
                sentences_raw.append(current_sentence)
                current_sentence = {
                    'text': '',
                    'start_ms': word['start_ms'],
                    'end_ms': word['end_ms']
                }
        
        # Add word to current sentence
        if current_sentence['text']:
            current_sentence['text'] += ' ' + word['text']
        else:
            current_sentence['text'] = word['text']
        current_sentence['end_ms'] = word['end_ms']
    
    # Don't forget the last sentence
    if current_sentence['text'].strip():
        sentences_raw.append(current_sentence)
    
    # Filter out very short sentences (likely noise)
    sentences_raw = [s for s in sentences_raw if len(s['text'].split()) >= 3]
    
    if not sentences_raw:
        return []
    
    # Extract entities in batch (efficient)
    texts = [s['text'] for s in sentences_raw]
    entities_list = extract_entities_batch(texts)
    
    # Combine into Sentence objects
    sentences = []
    for raw, entities in zip(sentences_raw, entities_list):
        sentences.append(Sentence(
            text=raw['text'],
            start_ms=raw['start_ms'],
            end_ms=raw['end_ms'],
            people=entities.people,
            companies=entities.companies,
            topics=entities.topics
        ))
    
    return sentences


def _parse_timestamp(ts: str) -> int:
    """Parse VTT timestamp to milliseconds."""
    # Format: HH:MM:SS.mmm
    parts = ts.split(':')
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds_parts = parts[2].split('.')
    seconds = int(seconds_parts[0])
    millis = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
    
    return (hours * 3600 + minutes * 60 + seconds) * 1000 + millis


if __name__ == "__main__":
    from pathlib import Path
    
    # Find a VTT file to test
    vtt_files = list(Path("audio").glob("*.vtt"))
    if not vtt_files:
        print("No VTT files found in audio/")
        exit(1)
    
    vtt_path = vtt_files[0]
    print(f"Testing with: {vtt_path}")
    print("-" * 60)
    
    sentences = parse_vtt_to_sentences(str(vtt_path))
    
    print(f"Total sentences: {len(sentences)}")
    print(f"\nFirst 5 sentences:")
    
    for i, sent in enumerate(sentences[:5]):
        print(f"\n{i+1}. [{sent.start_ms/1000:.1f}s - {sent.end_ms/1000:.1f}s]")
        print(f"   Text: {sent.text[:80]}...")
        if sent.people:
            print(f"   People: {sent.people}")
        if sent.companies:
            print(f"   Companies: {sent.companies}")
        if sent.topics:
            print(f"   Topics: {sent.topics}")
    
    # Count total entities
    all_people = set()
    all_companies = set()
    all_topics = set()
    
    for sent in sentences:
        all_people.update(sent.people)
        all_companies.update(sent.companies)
        all_topics.update(sent.topics)
    
    print(f"\n" + "=" * 60)
    print(f"Entity summary:")
    print(f"  Unique people: {len(all_people)}")
    print(f"  Unique companies: {len(all_companies)}")
    print(f"  Unique topics: {len(all_topics)}")
