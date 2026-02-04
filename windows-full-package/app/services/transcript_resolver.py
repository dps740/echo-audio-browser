"""Transcript resolver - fetch free transcripts before paying for transcription."""

import re
import urllib.request
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class ResolvedTranscript:
    """A transcript with timestamps."""
    segments: List[dict]  # [{timestamp, ms, text, speaker}]
    source: str  # "website", "rss", "youtube", "whisper", "deepgram"
    duration_ms: int


def parse_timestamp_to_ms(ts: str) -> int:
    """Convert HH:MM:SS or MM:SS to milliseconds."""
    parts = ts.split(':')
    if len(parts) == 3:
        h, m, s = map(int, parts)
        return (h * 3600 + m * 60 + s) * 1000
    elif len(parts) == 2:
        m, s = map(int, parts)
        return (m * 60 + s) * 1000
    return 0


async def resolve_transcript(
    episode_url: str,
    podcast_name: str = "",
    episode_title: str = "",
) -> Optional[ResolvedTranscript]:
    """
    Try to get a transcript for free before falling back to paid services.
    
    Resolution order:
    1. Podcast website (Lex Fridman, etc.)
    2. RSS transcript tag
    3. YouTube captions
    4. Whisper (local)
    5. Deepgram (paid)
    """
    
    # Try website scraping for known podcasts
    if "lexfridman" in podcast_name.lower() or "lex fridman" in episode_title.lower():
        transcript = await fetch_lex_fridman_transcript(episode_title)
        if transcript:
            return transcript
    
    # TODO: Add more podcast-specific scrapers
    # TODO: Add RSS transcript tag checking
    # TODO: Add YouTube caption fetching
    
    return None


async def fetch_lex_fridman_transcript(episode_title: str) -> Optional[ResolvedTranscript]:
    """Fetch transcript from lexfridman.com."""
    
    # Convert title to URL slug
    # "Dario Amodei: Anthropic CEO" -> "dario-amodei-transcript"
    slug = episode_title.lower()
    slug = re.sub(r'[:#|].*', '', slug)  # Remove subtitle
    slug = re.sub(r'[^a-z0-9\s-]', '', slug)
    slug = re.sub(r'\s+', '-', slug.strip())
    slug = slug.rstrip('-')
    
    url = f"https://lexfridman.com/{slug}-transcript"
    
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=15) as response:
            html = response.read().decode('utf-8')
        
        return parse_lex_transcript(html, url)
    except Exception as e:
        # Try without -transcript suffix (some older episodes)
        try:
            url2 = f"https://lexfridman.com/{slug}"
            req = urllib.request.Request(url2, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=15) as response:
                html = response.read().decode('utf-8')
            return parse_lex_transcript(html, url2)
        except:
            pass
    
    return None


def parse_lex_transcript(html: str, source_url: str) -> Optional[ResolvedTranscript]:
    """Parse Lex Fridman transcript HTML."""
    
    # Pattern: (HH:MM:SS) followed by text
    pattern = r'\((\d{1,2}:\d{2}:\d{2})\)\s*([^(]+?)(?=\(\d{1,2}:\d{2}:\d{2}\)|$)'
    matches = re.findall(pattern, html, re.DOTALL)
    
    if not matches:
        return None
    
    segments = []
    for ts, text in matches:
        # Clean HTML entities and tags
        clean_text = re.sub(r'<[^>]+>', ' ', text)
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        clean_text = clean_text.replace('&#8217;', "'")
        clean_text = clean_text.replace('&#8220;', '"')
        clean_text = clean_text.replace('&#8221;', '"')
        clean_text = clean_text.replace('&#8211;', '-')
        clean_text = clean_text.replace('&amp;', '&')
        
        # Extract speaker if present
        speaker = None
        speaker_match = re.match(r'^(Lex Fridman|[A-Z][a-z]+ [A-Z][a-z]+)\s*$', clean_text)
        if speaker_match:
            speaker = speaker_match.group(1)
            continue  # Skip speaker-only lines
        
        if len(clean_text) > 10:
            segments.append({
                'timestamp': ts,
                'ms': parse_timestamp_to_ms(ts),
                'text': clean_text,
                'speaker': speaker,
            })
    
    if not segments:
        return None
    
    return ResolvedTranscript(
        segments=segments,
        source=f"website:{source_url}",
        duration_ms=segments[-1]['ms'] if segments else 0,
    )


def transcript_to_text_with_times(transcript: ResolvedTranscript) -> str:
    """Convert transcript to text format for LLM segmentation."""
    lines = []
    for seg in transcript.segments:
        mins = seg['ms'] // 60000
        lines.append(f"[{mins}m] {seg['text']}")
    return "\n".join(lines)
