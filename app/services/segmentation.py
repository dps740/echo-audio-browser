"""
Segmentation: embed sentences, detect topic boundaries, generate snippets.

Pipeline:
  1. Embed all sentences (1 batched OpenAI call)
  2. Find topic boundaries via similarity drops (local computation)
  3. Merge short segments
  4. Batch LLM call: pick natural starts + generate snippets (1 call)
"""

import json
import numpy as np
import openai
from typing import List
from dataclasses import dataclass

from app.config import settings
from app.services.vtt_parser import Sentence


@dataclass
class Segment:
    """A topic segment with timestamp range and sentence indices."""
    start_idx: int
    end_idx: int  # exclusive
    start_ms: int
    end_ms: int
    sentences: List[Sentence]

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms

    @property
    def duration_s(self) -> float:
        return self.duration_ms / 1000

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.sentences)


@dataclass
class IndexedSegment:
    """A segment ready for storage — has snippet and final timestamps."""
    start_ms: int
    end_ms: int
    snippet: str
    text: str


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def get_embeddings(texts: List[str], batch_size: int = 100) -> np.ndarray:
    """
    Embed texts via OpenAI text-embedding-3-small.

    Cost: ~$0.02/1M tokens. A 90-min episode has ~700 sentences × ~20 tokens
    = 14,000 tokens = ~$0.0003.
    """
    client = openai.OpenAI(api_key=settings.openai_api_key)
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        all_embeddings.extend(e.embedding for e in response.data)

    return np.array(all_embeddings)


# ---------------------------------------------------------------------------
# Boundary detection (percentile-based, from segmentation_v3)
# ---------------------------------------------------------------------------

def find_boundaries(
    embeddings: np.ndarray,
    sentences: List[Sentence],
    max_segment_ms: int = 300_000,   # 5 min max before forced split
    percentile: float = 10.0         # bottom 10% similarities → boundaries
) -> List[int]:
    """
    Find topic boundaries by detecting drops in adjacent-sentence similarity.

    Returns list of sentence indices where a new segment starts.
    """
    if len(embeddings) < 2:
        return []

    # Normalise
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.maximum(norms, 1e-10)

    # Adjacent cosine similarities
    sims = np.array([
        float(np.dot(normed[i], normed[i + 1]))
        for i in range(len(normed) - 1)
    ])

    threshold = np.percentile(sims, percentile)

    boundaries = []
    last_boundary_ms = sentences[0].start_ms

    for i in range(1, len(sentences)):
        duration_since = sentences[i].start_ms - last_boundary_ms

        # Force split if segment too long
        if duration_since > max_segment_ms:
            boundaries.append(i)
            last_boundary_ms = sentences[i].start_ms
            continue

        # Natural boundary at similarity drop
        if sims[i - 1] < threshold:
            boundaries.append(i)
            last_boundary_ms = sentences[i].start_ms

    return boundaries


def _build_segments(
    sentences: List[Sentence],
    boundaries: List[int]
) -> List[Segment]:
    """Create Segment objects from sentence list + boundary indices."""
    indices = [0] + boundaries + [len(sentences)]
    segments = []

    for i in range(len(indices) - 1):
        s, e = indices[i], indices[i + 1]
        if e > s:
            seg_sentences = sentences[s:e]
            segments.append(Segment(
                start_idx=s,
                end_idx=e,
                start_ms=seg_sentences[0].start_ms,
                end_ms=seg_sentences[-1].end_ms,
                sentences=seg_sentences
            ))

    return segments


def _merge_short_segments(
    segments: List[Segment],
    min_duration_ms: int = 15_000
) -> List[Segment]:
    """Merge segments shorter than min_duration into their neighbour."""
    if not segments:
        return []

    merged = [segments[0]]

    for seg in segments[1:]:
        if seg.duration_ms < min_duration_ms:
            prev = merged[-1]
            merged[-1] = Segment(
                start_idx=prev.start_idx,
                end_idx=seg.end_idx,
                start_ms=prev.start_ms,
                end_ms=seg.end_ms,
                sentences=prev.sentences + seg.sentences
            )
        else:
            merged.append(seg)

    # Handle short first segment
    if len(merged) > 1 and merged[0].duration_ms < min_duration_ms:
        second = merged[1]
        merged[1] = Segment(
            start_idx=merged[0].start_idx,
            end_idx=second.end_idx,
            start_ms=merged[0].start_ms,
            end_ms=second.end_ms,
            sentences=merged[0].sentences + second.sentences
        )
        merged = merged[1:]

    return merged


# ---------------------------------------------------------------------------
# Batch LLM: natural starts + snippets (1 call for all segments)
# ---------------------------------------------------------------------------

def _format_time(ms: int) -> str:
    m, s = divmod(ms // 1000, 60)
    return f"{m}:{s:02d}"


def generate_snippets_with_natural_starts(
    sentences: List[Sentence],
    segments: List[Segment],
    episode_title: str,
    context_sentences: int = 3,
    model: str = "gpt-4o-mini"
) -> List[IndexedSegment]:
    """
    Single LLM call that, for every segment:
      1. Picks the natural listening start (topic introduction)
      2. Generates a specific one-sentence snippet

    Returns list of IndexedSegment with resolved timestamps.
    """
    # Build per-segment blocks with context
    blocks = []
    for i, seg in enumerate(segments):
        ctx_start = max(0, seg.start_idx - context_sentences)

        lines = []
        for j in range(ctx_start, seg.start_idx):
            s = sentences[j]
            label = f"before_{seg.start_idx - j}"
            lines.append(f"  [{label}] [{_format_time(s.start_ms)}] {s.text}")

        seg_text = seg.text
        if len(seg_text) > 2000:
            seg_text = seg_text[:2000] + "..."

        lines.append(
            f"  [boundary] [{_format_time(seg.start_ms)}] {seg_text}"
        )

        blocks.append(f"=== SEGMENT {i + 1} ===\n" + "\n".join(lines))

    prompt = f"""You are processing {len(segments)} segments from the podcast episode "{episode_title}".

For each segment, I detected an approximate topic boundary using embedding
similarity.  The [before_N] lines show sentences just before the boundary —
the topic introduction may actually start there (a question, transition
phrase, or new thought).

For each segment, return TWO things:

1. **start** — where a listener should begin hearing this clip.
   Pick "before_1", "before_2", "before_3", or "boundary" (if the detected
   boundary is already a natural start).

2. **snippet** — ONE specific sentence describing what this segment discusses.
   BE SPECIFIC: include names, numbers, claims.
   BAD: "Discussion about AI"
   GOOD: "Why ChatGPT-style chatbots are just the first phase of AI applications"

{chr(10).join(blocks)}

Return ONLY a JSON object with a "segments" key containing an array of {len(segments)} objects.
Example: {{"segments": [{{"start": "before_2", "snippet": "..."}}, ...]}}"""

    client = openai.OpenAI(api_key=settings.openai_api_key)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You identify natural topic starts in podcasts and "
                        "write specific one-sentence summaries. "
                        "Reply with valid JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)

        # Accept either {"segments": [...]} or a bare list
        if isinstance(parsed, dict):
            items = (
                parsed.get("segments")
                or parsed.get("results")
                or list(parsed.values())[0]
            )
        elif isinstance(parsed, list):
            items = parsed
        else:
            items = []

    except Exception as e:
        print(f"LLM batch call failed, falling back to raw text snippets: {e}")
        items = [
            {"start": "boundary", "snippet": seg.text[:150] + "..."}
            for seg in segments
        ]

    # Resolve natural starts → actual timestamps
    results: List[IndexedSegment] = []
    for i, seg in enumerate(segments):
        item = (
            items[i] if i < len(items)
            else {"start": "boundary", "snippet": seg.text[:150] + "..."}
        )
        snippet = item.get("snippet", seg.text[:150] + "...")
        start_key = item.get("start", "boundary")

        # Map "before_N" → sentence index
        resolved_start_idx = seg.start_idx
        if start_key.startswith("before_"):
            try:
                n = int(start_key.split("_")[1])
                resolved_start_idx = max(0, seg.start_idx - n)
            except (ValueError, IndexError):
                pass

        resolved_start_ms = sentences[resolved_start_idx].start_ms

        full_text = " ".join(
            s.text for s in sentences[resolved_start_idx:seg.end_idx]
        )

        results.append(IndexedSegment(
            start_ms=resolved_start_ms,
            end_ms=seg.end_ms,
            snippet=snippet.strip('"').strip("'"),
            text=full_text
        ))

    return results


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def index_episode(
    vtt_content: str,
    episode_title: str,
    max_segment_ms: int = 300_000,
    min_segment_ms: int = 15_000,
    boundary_percentile: float = 10.0,
    model: str = "gpt-4o-mini"
) -> List[IndexedSegment]:
    """
    Full indexing pipeline: VTT → sentences → boundaries → snippets.

    Returns a list of IndexedSegment ready for storage.
    API calls: 3 total (sentence embeddings, batch LLM, snippet embeddings in storage).
    """
    from app.services.vtt_parser import parse_vtt_to_words, words_to_sentences

    # 1. Parse VTT
    words = parse_vtt_to_words(vtt_content)
    if not words:
        return []

    sentences = words_to_sentences(words)
    if len(sentences) < 2:
        return []

    # 2. Embed sentences (1 API call)
    embeddings = get_embeddings([s.text for s in sentences])
    for i, sent in enumerate(sentences):
        sent.embedding = embeddings[i]

    # 3. Detect boundaries (local)
    boundaries = find_boundaries(
        embeddings, sentences,
        max_segment_ms=max_segment_ms,
        percentile=boundary_percentile
    )

    # 4. Build + merge segments
    segments = _build_segments(sentences, boundaries)
    segments = _merge_short_segments(segments, min_segment_ms)

    if not segments:
        return []

    # 5. Batch LLM: natural starts + snippets (1 API call)
    indexed = generate_snippets_with_natural_starts(
        sentences, segments, episode_title, model=model
    )

    return indexed
