# Echo Audio Browser - Ingestion Pipeline

## Overview
Topic-first podcast search. Accurate word-level timestamps via MFA forced alignment.

**Key Insight:** MP3 files have seek issues (VBR/metadata problems). Convert to WAV for accurate timestamps and playback.

---

## End-to-End Flow

### 1. DOWNLOAD
**Input:** YouTube URL or RSS feed URL  
**Output:** `episode.mp3` + `transcript.txt`

```bash
# YouTube (preferred - free transcript)
yt-dlp -x --audio-format mp3 -o "episode.mp3" "https://youtube.com/..."
yt-dlp --write-auto-sub --sub-lang en --skip-download -o "transcript" "https://youtube.com/..."

# RSS-only (no transcript available)
# Download MP3 from feed, will need Whisper in step 3
```

**Time:** ~1-2 min (download speed dependent)

---

### 2. CONVERT TO WAV
**Input:** `episode.mp3`  
**Output:** `episode.wav` (16kHz mono, required for MFA)

```bash
ffmpeg -i episode.mp3 -ar 16000 -ac 1 episode.wav
```

**Time:** ~30 seconds for 1-hour episode

---

### 3. TRANSCRIBE (if needed)
**Input:** `episode.wav` (only if no YouTube transcript)  
**Output:** `transcript.txt`

```bash
# Using faster-whisper (GPU)
faster-whisper episode.wav --model small --output_format txt
```

**Time:** ~2-5 min on GTX 1050 Ti  
**Skip if:** YouTube transcript available

---

### 4. PREPARE FOR MFA
**Input:** `episode.wav` + `transcript.txt`  
**Output:** MFA input directory structure

```bash
mkdir -p mfa_input/episode
cp episode.wav mfa_input/episode/
cp transcript.txt mfa_input/episode/episode.txt  # Must match wav filename
```

MFA expects:
```
mfa_input/
└── episode/
    ├── episode.wav
    └── episode.txt
```

---

### 5. ALIGN WITH MFA
**Input:** MFA input directory  
**Output:** `episode.TextGrid` with word-level timestamps

```bash
mfa align mfa_input/ english_mfa english_mfa mfa_output/
```

**Time:** ~3 min for 1-hour episode  
**Output:** `mfa_output/episode/episode.TextGrid`

---

### 6. PARSE TEXTGRID
**Input:** `episode.TextGrid`  
**Output:** JSON with word timestamps

```python
import textgrid

tg = textgrid.TextGrid.fromFile('episode.TextGrid')
words_tier = tg.getFirst('words')

words = []
for interval in words_tier:
    if interval.mark:  # Skip empty intervals
        words.append({
            'word': interval.mark,
            'start': interval.minTime,
            'end': interval.maxTime
        })

# Output: [{'word': 'obviously', 'start': 765.23, 'end': 765.89}, ...]
```

---

### 7. SEGMENT WITH LLM
**Input:** Word timestamps + full transcript  
**Output:** Topic segments with accurate start/end times

```python
# Send to LLM (Claude/GPT)
prompt = """
Given this transcript with word timestamps, identify 1-3 minute topic segments.
For each segment, provide:
- topic: Brief description
- start_word: First word of segment (exact match)
- end_word: Last word of segment (exact match)
- summary: 1-2 sentence summary

RULES:
- Start at sentence boundaries (after periods, not mid-sentence)
- Each segment should be 1-3 minutes
- Topics should be semantically coherent
"""

# LLM returns segments with start_word/end_word
# We look up exact timestamps from our word list
```

**Time:** ~5-10 sec per segment (API call)  
**Cost:** ~$0.02-0.03 per episode

---

### 8. STORE IN CHROMADB
**Input:** Segments with timestamps  
**Output:** Searchable vector database

```python
collection.add(
    ids=[segment_id],
    documents=[segment['summary']],
    metadatas=[{
        'episode_id': episode_id,
        'topic': segment['topic'],
        'start_time': segment['start'],
        'end_time': segment['end'],
        'duration': segment['end'] - segment['start']
    }]
)
```

---

### 9. SERVE AUDIO
**Input:** User search query → matched segment  
**Output:** Audio clip or full file with accurate seeking

**Option A: Pre-clip segments (best UX)**
```bash
ffmpeg -i episode.wav -ss 765.23 -to 832.45 -c copy segment_001.wav
```

**Option B: Serve full WAV, seek in browser**
```html
<audio src="/audio/episode.wav" id="player">
<script>
  player.currentTime = 765.23;  // Accurate on WAV!
  player.play();
</script>
```

**Option C: Convert clips to MP3 for smaller files**
```bash
ffmpeg -i segment.wav -codec:a libmp3lame -b:a 128k segment.mp3
```

---

## Complete Pipeline Summary

```
┌─────────────┐
│  YouTube    │──→ MP3 + Transcript (free)
│  or RSS     │──→ MP3 only (needs Whisper)
└─────────────┘
       │
       ▼
┌─────────────┐
│ Convert     │──→ WAV (16kHz mono)
│ ffmpeg      │    ~30 sec
└─────────────┘
       │
       ▼
┌─────────────┐
│ Transcribe  │──→ transcript.txt
│ (if needed) │    ~2-5 min (Whisper)
└─────────────┘    SKIP if YouTube transcript
       │
       ▼
┌─────────────┐
│ Align       │──→ TextGrid
│ MFA         │    ~3 min
└─────────────┘    Word-level timestamps!
       │
       ▼
┌─────────────┐
│ Parse       │──→ JSON word timestamps
│ textgrid.py │    <1 sec
└─────────────┘
       │
       ▼
┌─────────────┐
│ Segment     │──→ Topic segments
│ LLM         │    ~$0.03, ~10 sec
└─────────────┘
       │
       ▼
┌─────────────┐
│ Store       │──→ ChromaDB
│ + Index     │    Searchable!
└─────────────┘
       │
       ▼
┌─────────────┐
│ Serve       │──→ WAV clips or full file
│ Web Player  │    Accurate playback!
└─────────────┘
```

---

## Time & Cost Per Episode (1 hour)

| Step | Time | Cost |
|------|------|------|
| Download | 1-2 min | Free |
| Convert to WAV | 30 sec | Free |
| Transcribe | 0 or 2-5 min | Free (YT) or GPU |
| MFA Align | 3 min | Free |
| Parse | <1 sec | Free |
| Segment (LLM) | 10 sec | $0.03 |
| Store | <1 sec | Free |
| **TOTAL** | **~5-10 min** | **$0.03** |

---

## Storage

| Format | Size (1 hr) | Use |
|--------|-------------|-----|
| MP3 | ~60 MB | Archive only |
| WAV (16kHz mono) | ~115 MB | Alignment + playback |
| WAV clips | ~2 MB each | Pre-cut segments |
| MP3 clips | ~0.5 MB each | Compressed segments |

**Recommendation:** Store WAV for processing, serve MP3 clips for bandwidth.

---

## Fallback: No YouTube Transcript

For podcasts only on RSS (no YouTube):
1. Download MP3 from RSS feed
2. Convert to WAV
3. **Transcribe with Whisper** (adds 2-5 min)
4. Continue with MFA alignment as normal

The pipeline handles both cases - just skip step 3 when YouTube transcript exists.

---

## Dependencies

**On David's Desktop (transcription machine):**
- yt-dlp
- ffmpeg
- faster-whisper (for non-YouTube fallback)
- MFA (Montreal Forced Aligner)
- Python: textgrid

**On Server (cloud):**
- ChromaDB
- FastAPI
- Anthropic/OpenAI API (for segmentation)

---

## Files

- `PIPELINE.md` - This document
- `app/services/ingest.py` - Main ingestion logic
- `app/services/segmentation.py` - LLM segmentation
- `app/services/mfa_align.py` - MFA wrapper (to create)
- `app/services/textgrid_parser.py` - Parse MFA output (to create)
