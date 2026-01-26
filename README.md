# Echo - Topic-First Audio Browser

**Status:** MVP Development  
**Philosophy:** Anti-Summarizer - deliver high-fidelity audio segments, not text summaries

## Core Concept

Echo lets users browse podcasts by **topic** rather than episode. Instead of "Episode 234 of Lex Fridman", users see "AI Safety discussions" and get a curated playlist of relevant segments from multiple podcasts.

### Key Innovation: Virtual Stitching

- No audio re-hosting (legal/cost advantage)
- Stream directly from original source URLs
- Client seeks to specific timestamps
- Zero storage costs for audio

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   RSS Feeds     │────▶│   Transcription  │────▶│   Segmentation  │
│   (podcasts)    │     │   (Deepgram)     │     │   (LLM)         │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Web Player    │◀────│   Manifest API   │◀────│   Vector DB     │
│   (client)      │     │   (FastAPI)      │     │   (ChromaDB)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Add your API keys

# Run the server
python -m uvicorn app.main:app --reload

# Access API docs
open http://localhost:8000/docs
```

## Project Structure

```
echo-audio-browser/
├── app/
│   ├── main.py              # FastAPI app
│   ├── config.py            # Settings
│   ├── models.py            # Pydantic models
│   ├── routers/
│   │   ├── feeds.py         # RSS feed management
│   │   ├── episodes.py      # Episode ingestion
│   │   ├── segments.py      # Segment search
│   │   └── playlists.py     # Manifest generation
│   ├── services/
│   │   ├── transcription.py # Deepgram/Whisper
│   │   ├── segmentation.py  # LLM chunking
│   │   ├── vectordb.py      # ChromaDB
│   │   └── rss.py           # Feed parsing
│   └── utils/
├── tests/
├── static/                   # Simple web player
├── requirements.txt
└── .env.example
```

## API Endpoints

- `POST /feeds` - Add podcast RSS feed
- `GET /feeds` - List subscribed feeds
- `POST /episodes/{id}/ingest` - Transcribe & segment episode
- `GET /segments/search?q=topic` - Search segments by topic
- `GET /playlists/topic/{topic}` - Get playback manifest for topic
- `GET /playlists/daily-mix` - Personalized daily mix

## Environment Variables

```
DEEPGRAM_API_KEY=your_key
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
DATABASE_URL=sqlite:///./echo.db
CHROMA_PERSIST_DIR=./chroma_data
```

## Development Roadmap

### MVP (Current)
- [ ] RSS feed ingestion
- [ ] Transcription (Deepgram)
- [ ] LLM segmentation
- [ ] Vector search
- [ ] Manifest API
- [ ] Simple web player

### V1
- [ ] User accounts
- [ ] Personalization
- [ ] Mobile apps (iOS/Android)
- [ ] Dynamic ad detection

### V2
- [ ] Audio fingerprinting for timestamp drift
- [ ] Multi-language support
- [ ] Creator analytics

## License

MIT
