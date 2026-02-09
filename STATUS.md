# Echo Audio Browser - Project Status
**Last Updated:** 2026-02-09 05:25 UTC

## Overview
Topic-first podcast browser with semantic search across episodes.

## Current Architecture

### V4 Segment System (Active)
- **Collection:** `v4_segments` in ChromaDB
- **Embedding source:** LLM-generated snippets (not raw transcripts)
- **Snippet format:** 2-3 sentence summaries of what each segment discusses
- **Search:** Semantic search on snippets, returns clips

### Indexed Content
| Episode ID | Title | Segments | Status |
|------------|-------|----------|--------|
| BvhFuEp55X0 | Joe Rogan Experience | 160 | ✅ |
| mlYSvE3XZcQ | Episode | 149 | ✅ |
| Wp1Wn6QkkKU | Episode | 114 | ✅ |
| UPfN2G0RyQM | Episode | 113 | ✅ |
| sY7WI669CFQ | Episode | 110 | ✅ |
| w2BqPnVKVo4 | Episode | 84 | ✅ |
| gXY1kx7zlkk | Episode | 72 | ✅ |
| NWZZEa9BURw | Episode | 65 | ✅ |
| wTiHheA40nI | Episode | 60 | ✅ |
| kX7zW2TIriM | Episode | 33 | ✅ |
| xNk2QgZ12Rk | Episode | 19 | ✅ |
| _0dxe6Sci6Y | Episode | 7 | ✅ |
| EV7WhVT270Q | - | - | ❌ VTT format |
| Z-FRe5AKmCU | - | - | ❌ VTT format |

**Total:** 12 episodes, 986 segments

### API Endpoints
- `GET /v4/search?q=<query>&top_k=N` - Semantic search across all episodes
- `GET /v4/stats` - Collection statistics
- `POST /v3/segment-refined/<video_id>` - Index new episode with V4 snippets
- `GET /library/overview` - Browse all content
- `GET /playlists/topic/<topic>` - Generate playlist for topic

### File Structure
```
echo-audio-browser/
├── app/
│   ├── main.py
│   ├── routers/
│   │   ├── v4_segments.py      # V4 search & indexing
│   │   ├── library.py          # Library browsing (V4)
│   │   ├── playlists.py        # Playlist generation (V4)
│   │   └── ...
│   └── services/
│       └── clip_extractor.py   # Extract MP3 clips
├── audio/
│   ├── *.wav                   # Full audio files
│   ├── *.vtt                   # Transcripts
│   └── clips/                  # Pre-extracted segment clips
├── chroma_data/                # ChromaDB persistence
└── static/
    └── index.html              # Frontend UI
```

### Known Issues
1. **VTT Parser:** Two episodes have different VTT format (no `<c>` word-level tags)
2. **Browser quirk:** Some clips show "Format error" despite being valid MP3
3. **Titles:** Episode titles not fetched during indexing

### Next Steps
1. Fix VTT parser for simple format (handle `- ` prefix)
2. Add YouTube title fetching during ingest
3. Improve server stability under load
4. Consider clip caching strategy

## Running the Server
```bash
cd ~/clawd/projects/echo-audio-browser
source venv/bin/activate
python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8765
```

## Tunnel (for external access)
```bash
cloudflared tunnel --url http://localhost:8765
```
