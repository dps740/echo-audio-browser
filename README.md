# Echo - Topic-First Audio Browser

**Status:** V3 Pipeline Complete  
Browse podcasts by **topic**, not episode. Search "AI safety" and get clips of relevant segments from across your library.

## Architecture (V3)

```
YouTube Video
    ↓
Download: Audio + VTT transcript (yt-dlp)
    ↓
Convert to WAV (ffmpeg, 16kHz mono)
    ↓
Parse VTT → word-level timestamps
    ↓
Sentence segmentation (800ms pause threshold)
    ↓
Embed sentences (text-embedding-3-small)
    ↓
Detect topic boundaries (embedding similarity)
    ↓
LLM refinement (gpt-4o-mini):
  - Find true topic start
  - Generate SPECIFIC snippet
    ↓
Store segments + serve MP3 clips
```

**Key insight:** YouTube VTT timestamps are accurate when clips extracted from WAV. No MFA/Whisper needed.

## Quick Start

```bash
# 1. Clone and setup
git clone https://github.com/dps740/echo-audio-browser
cd echo-audio-browser
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env - add OPENAI_API_KEY

# 3. Run
python -m uvicorn app.main:app --host 0.0.0.0 --port 8765

# 4. Index an episode
curl -X POST "http://localhost:8765/v3/segment-refined/VIDEO_ID"

# 5. Search
curl "http://localhost:8765/v3/search-refined/VIDEO_ID?q=AI"
```

## API Endpoints (V3)

| Endpoint | Purpose |
|----------|---------|
| `POST /v3/segment-refined/{video_id}` | Index episode with LLM refinement |
| `GET /v3/search-refined/{video_id}?q=` | Search with pre-computed snippets |
| `GET /v3/clip/{video_id}?start_ms=&end_ms=` | Generate MP3 clip |

## Cost

- **Indexing:** ~$0.02 per episode (embeddings + LLM refinement)
- **Search:** Free (uses pre-computed data)

## Files

| File | Purpose |
|------|---------|
| `PIPELINE.md` | Full pipeline documentation |
| `app/services/segmentation_v3.py` | VTT parsing, sentence segmentation |
| `app/services/segment_refiner.py` | LLM boundary + snippet generation |
| `app/services/search_v3.py` | Cluster-based search |
| `app/services/clip_extractor.py` | WAV → MP3 clip extraction |
| `app/routers/test_v3.py` | V3 API endpoints |

## Pre-segmenting (Before API)

1. Download audio + VTT: `yt-dlp -x --write-auto-sub URL`
2. Convert to WAV: `ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav`
3. Place in `audio/` folder

## License

MIT
