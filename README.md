# Echo - Topic-First Audio Browser

**Status:** MVP Testing  
Browse podcasts by **topic**, not episode. Search "AI safety" and get a playlist of relevant segments from across your library.

## Quick Start (Windows)

```bash
# 1. Clone and setup
git clone https://github.com/dps740/echo-audio-browser
cd echo-audio-browser
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env - add your OPENAI_API_KEY

# 3. Run
start_server.bat
# Or: python -m uvicorn app.main:app --port 8765

# 4. Open browser
http://localhost:8765/player
```

## Environment Variables

```bash
# Required
OPENAI_API_KEY=sk-...          # For segmentation + embeddings

# Optional (defaults work fine)
EMBEDDING_MODEL=text-embedding-3-small
SEGMENTATION_MODEL=gpt-4o-mini
USE_OPENAI_EMBEDDINGS=true
CHROMA_PERSIST_DIR=./chroma_data
```

## Project Structure

```
echo-audio-browser/
├── app/                    # FastAPI server
│   ├── main.py            # App entry point
│   ├── config.py          # Settings
│   ├── routers/           # API endpoints
│   │   ├── youtube_ingest.py   # Ingest, re-analyze, repair
│   │   ├── library.py     # Browse podcasts/episodes
│   │   └── playlists.py   # Search → playlist
│   └── services/          # Business logic
│       ├── segmentation.py     # LLM segmentation
│       ├── hybrid_search.py    # Semantic + keyword search
│       └── vectordb.py         # ChromaDB operations
├── static/
│   └── index.html         # Web UI (Dashboard, Library, Ingest, Player)
├── scripts/               # Utility scripts
│   ├── echo_ingest.py     # CLI batch ingest
│   └── ingest.bat         # Windows batch wrapper
├── chroma_data/           # Vector DB storage (gitignored)
├── start_server.bat       # Windows launcher
├── ECHO_PROJECT.md        # Detailed architecture docs
└── requirements.txt
```

## Features

- **YouTube ingestion** - Paste URL or select from curated podcasts
- **LLM segmentation** - GPT-4o-mini identifies topic boundaries (2-10 min segments)
- **Hybrid search** - Semantic + keyword matching
- **Audio player** - Stream with seek, crossfade between segments
- **Re-analyze** - Re-run full pipeline on existing content

## API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /player` | Web UI |
| `GET /library/overview` | Dashboard stats |
| `GET /playlists/topic/{topic}` | Search → playlist |
| `POST /ingest/youtube/url` | Ingest single video |
| `POST /ingest/youtube/deep-reindex` | Re-run LLM segmentation |
| `POST /ingest/youtube/repair` | Fix corrupted collections |

## Cost

~$0.03-0.05 per episode (GPT-4o-mini segmentation + embeddings)

## Docs

See `ECHO_PROJECT.md` for full architecture, pipeline details, and troubleshooting.

## License

MIT
