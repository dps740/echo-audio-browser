"""Parse MFA TextGrid output to extract word-level timestamps."""

import re
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class WordTimestamp:
    """A word with its precise timestamp."""
    word: str
    start: float  # seconds
    end: float    # seconds
    
    @property
    def start_ms(self) -> int:
        """Start time in milliseconds."""
        return int(self.start * 1000)
    
    @property
    def end_ms(self) -> int:
        """End time in milliseconds."""
        return int(self.end * 1000)
    
    @property
    def duration_ms(self) -> int:
        """Duration in milliseconds."""
        return self.end_ms - self.start_ms


@dataclass 
class PhoneTimestamp:
    """A phoneme with its timestamp (optional, for fine-grained analysis)."""
    phone: str
    start: float
    end: float


def parse_textgrid(textgrid_path: str) -> List[WordTimestamp]:
    """
    Parse a Praat TextGrid file and extract word-level timestamps.
    
    MFA TextGrid format has tiers for "words" and "phones".
    We extract the "words" tier.
    
    Args:
        textgrid_path: Path to .TextGrid file
    
    Returns:
        List of WordTimestamp objects
    """
    with open(textgrid_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if it's short format or long format
    if '"IntervalTier"' in content or '"TextTier"' in content:
        return _parse_long_format(content)
    else:
        return _parse_short_format(content)


def _parse_long_format(content: str) -> List[WordTimestamp]:
    """Parse long (full) TextGrid format."""
    words = []
    
    # Find the words tier
    # Pattern: item [N]: ... class = "IntervalTier" ... name = "words"
    tier_pattern = r'item \[\d+\]:.*?class = "IntervalTier".*?name = "words".*?intervals: size = (\d+)(.*?)(?=item \[|$)'
    tier_match = re.search(tier_pattern, content, re.DOTALL | re.IGNORECASE)
    
    if not tier_match:
        # Try alternate format
        tier_pattern = r'name = "words".*?intervals: size = (\d+)(.*?)(?=name = "|$)'
        tier_match = re.search(tier_pattern, content, re.DOTALL | re.IGNORECASE)
    
    if not tier_match:
        print("Could not find 'words' tier in TextGrid")
        return words
    
    tier_content = tier_match.group(2)
    
    # Extract each interval
    # Pattern: intervals [N]: xmin = 0.0 xmax = 0.5 text = "word"
    interval_pattern = r'intervals \[\d+\]:\s*xmin = ([\d.]+)\s*xmax = ([\d.]+)\s*text = "([^"]*)"'
    
    for match in re.finditer(interval_pattern, tier_content):
        start = float(match.group(1))
        end = float(match.group(2))
        word = match.group(3).strip()
        
        # Skip empty intervals (silence/pauses)
        if word and word != "":
            words.append(WordTimestamp(
                word=word,
                start=start,
                end=end
            ))
    
    return words


def _parse_short_format(content: str) -> List[WordTimestamp]:
    """Parse short TextGrid format."""
    words = []
    lines = content.strip().split('\n')
    
    i = 0
    in_words_tier = False
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Look for "words" tier
        if '"words"' in line or "'words'" in line:
            in_words_tier = True
            i += 1
            continue
        
        # Next tier starts
        if in_words_tier and ('"phones"' in line or '"' in line and 'Tier' not in lines[i-1] if i > 0 else False):
            in_words_tier = False
        
        if in_words_tier:
            # Try to parse interval: xmin, xmax, text
            try:
                if line.replace('.', '').replace('-', '').isdigit():
                    xmin = float(line)
                    i += 1
                    xmax = float(lines[i].strip())
                    i += 1
                    text = lines[i].strip().strip('"').strip("'")
                    
                    if text and text != "":
                        words.append(WordTimestamp(
                            word=text,
                            start=xmin,
                            end=xmax
                        ))
            except (ValueError, IndexError):
                pass
        
        i += 1
    
    return words


def parse_textgrid_with_phones(textgrid_path: str) -> tuple[List[WordTimestamp], List[PhoneTimestamp]]:
    """
    Parse both words and phones tiers from TextGrid.
    
    Returns:
        (words, phones) tuple
    """
    words = parse_textgrid(textgrid_path)
    phones = []  # TODO: implement phone parsing if needed
    return words, phones


def words_to_transcript(words: List[WordTimestamp]) -> str:
    """Convert word timestamps back to plain text."""
    return ' '.join(w.word for w in words)


def find_word_at_time(words: List[WordTimestamp], time_ms: int) -> Optional[WordTimestamp]:
    """Find the word being spoken at a given timestamp."""
    time_sec = time_ms / 1000.0
    for word in words:
        if word.start <= time_sec <= word.end:
            return word
    return None


def find_words_in_range(
    words: List[WordTimestamp], 
    start_ms: int, 
    end_ms: int
) -> List[WordTimestamp]:
    """Find all words within a time range."""
    start_sec = start_ms / 1000.0
    end_sec = end_ms / 1000.0
    return [w for w in words if w.start >= start_sec and w.end <= end_sec]


def find_phrase_timestamps(
    words: List[WordTimestamp],
    phrase: str,
    fuzzy: bool = True
) -> Optional[tuple[int, int]]:
    """
    Find the start and end timestamps for a phrase.
    
    Args:
        words: List of word timestamps
        phrase: Phrase to find
        fuzzy: If True, allow partial/fuzzy matching
    
    Returns:
        (start_ms, end_ms) if found, None otherwise
    """
    phrase_words = phrase.lower().split()
    if not phrase_words:
        return None
    
    word_texts = [w.word.lower() for w in words]
    
    # Exact match first
    for i in range(len(words) - len(phrase_words) + 1):
        if word_texts[i:i+len(phrase_words)] == phrase_words:
            return (words[i].start_ms, words[i+len(phrase_words)-1].end_ms)
    
    # Fuzzy match: find best substring match
    if fuzzy:
        best_match = None
        best_score = 0
        
        for i in range(len(words) - len(phrase_words) + 1):
            window = word_texts[i:i+len(phrase_words)]
            score = sum(1 for a, b in zip(window, phrase_words) if a == b or a.startswith(b) or b.startswith(a))
            
            if score > best_score and score >= len(phrase_words) * 0.7:  # 70% match threshold
                best_score = score
                best_match = (words[i].start_ms, words[i+len(phrase_words)-1].end_ms)
        
        return best_match
    
    return None


def export_to_json(words: List[WordTimestamp]) -> list:
    """Export word timestamps to JSON-serializable format."""
    return [
        {
            "word": w.word,
            "start": w.start,
            "end": w.end,
            "start_ms": w.start_ms,
            "end_ms": w.end_ms
        }
        for w in words
    ]


def load_from_json(data: list) -> List[WordTimestamp]:
    """Load word timestamps from JSON data."""
    return [
        WordTimestamp(
            word=item["word"],
            start=item["start"],
            end=item["end"]
        )
        for item in data
    ]


# Try to use the textgrid library if available
try:
    import textgrid as tg_lib
    
    def parse_textgrid_lib(textgrid_path: str) -> List[WordTimestamp]:
        """Parse TextGrid using the textgrid library (more robust)."""
        try:
            grid = tg_lib.TextGrid.fromFile(textgrid_path)
            words_tier = None
            
            # Find words tier
            for tier in grid.tiers:
                if tier.name.lower() == 'words':
                    words_tier = tier
                    break
            
            if words_tier is None:
                print("No 'words' tier found, using first IntervalTier")
                for tier in grid.tiers:
                    if hasattr(tier, 'intervals'):
                        words_tier = tier
                        break
            
            if words_tier is None:
                return []
            
            words = []
            for interval in words_tier.intervals:
                if interval.mark and interval.mark.strip():
                    words.append(WordTimestamp(
                        word=interval.mark.strip(),
                        start=interval.minTime,
                        end=interval.maxTime
                    ))
            
            return words
            
        except Exception as e:
            print(f"textgrid lib parse failed: {e}, falling back to regex")
            return parse_textgrid(textgrid_path)
    
    # Use library version if available
    parse_textgrid = parse_textgrid_lib

except ImportError:
    # textgrid library not installed, use regex parser
    pass
