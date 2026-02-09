"""VTT transcript parsing — word extraction and sentence grouping."""

import re
from typing import List
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Sentence:
    """A single sentence/utterance from the transcript."""
    text: str
    start_ms: int
    end_ms: int
    embedding: Optional[np.ndarray] = field(default=None, repr=False)


def parse_vtt_to_words(vtt_content: str) -> List[dict]:
    """
    Parse VTT content to word-level timestamps.

    YouTube VTT uses <timestamp><c> word</c> format for word-level timing.
    """
    words = []

    for line in vtt_content.split('\n'):
        if '-->' not in line and not line.startswith('WEBVTT') and line.strip():
            matches = re.findall(
                r'<(\d{2}:\d{2}:\d{2}\.\d{3})><c>\s*([^<]+)</c>', line
            )
            for ts, word in matches:
                h, m, s = ts.split(':')
                ms = int(float(h) * 3600000 + float(m) * 60000 + float(s) * 1000)
                words.append({
                    'text': word.strip(),
                    'start_ms': ms,
                    'end_ms': ms + 200  # Estimate — VTT only gives start times
                })

    words.sort(key=lambda w: w['start_ms'])
    return words


def words_to_sentences(
    words: List[dict],
    pause_threshold_ms: int = 800
) -> List[Sentence]:
    """
    Group words into sentences based on pauses.

    A gap > pause_threshold_ms between consecutive words indicates a
    sentence boundary.
    """
    if not words:
        return []

    sentences = []
    current_words = [words[0]]

    for i in range(1, len(words)):
        gap = words[i]['start_ms'] - words[i - 1]['end_ms']

        if gap > pause_threshold_ms:
            text = ' '.join(w['text'] for w in current_words)
            sentences.append(Sentence(
                text=text,
                start_ms=current_words[0]['start_ms'],
                end_ms=current_words[-1]['end_ms']
            ))
            current_words = [words[i]]
        else:
            current_words.append(words[i])

    # Last sentence
    if current_words:
        text = ' '.join(w['text'] for w in current_words)
        sentences.append(Sentence(
            text=text,
            start_ms=current_words[0]['start_ms'],
            end_ms=current_words[-1]['end_ms']
        ))

    return sentences
