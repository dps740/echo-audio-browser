# Echo Audio Browser - Scale-Up Session Handoff

**Created:** 2026-02-11 17:45 UTC
**Purpose:** Everything needed to scale Echo from 16 → 50 episodes
**Previous Session:** Fixed search bugs, improved boundaries, ready for scale

---

## 🚀 Quick Start

### Current State
- **16 episodes** indexed (All-In Podcast)
- **438 topics**, **10,217 sentences** in Typesense
- **V3 segmentation** with boundary refinement + conjunction fix
- **Clip extraction** working (WAV → MP3 on demand)
- **Live URL:** Check `TUNNEL_URL.txt` for current cloudflared URL

### Start the Server
```bash
cd ~/clawd/projects/echo-audio-browser
tmux kill-session -t echo 2>/dev/null
tmux new-session -d -s echo "cd ~/clawd/projects/echo-audio-browser && source venv/bin/activate && export OPENAI_API_KEY=\$(cat ~/.clawd/.api-keys/openai.key) && python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8766 --workers 1"
```

### Start the Tunnel
```bash
nohup cloudflared tunnel --url http://localhost:8766 > tunnel.log 2>&1 &
sleep 8
grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" tunnel.log | head -1 > TUNNEL_URL.txt
cat TUNNEL_URL.txt
```

---

## 📋 Task: Scale to 50 Episodes

### See: `SCALE_UP_PLAN.md` for detailed plan

### High-Level Steps
1. **Select episodes** — Mix of All-In + other podcasts (Lex, Ferriss, etc.)
2. **Download content** — yt-dlp for audio + captions
3. **Convert to WAV** — Required for accurate clip timestamps
4. **Run indexing pipeline** — V3 segmentation (~5-8 min per episode)
5. **Validate quality** — Test searches, verify clips

---

## 🔧 Key Files & Architecture

### Indexing Pipeline
```
app/services/
├── pipeline_v3.py          # Main indexing entry point
├── topic_segmentation_v3.py # V3 segmentation with:
│   ├── Boundary refinement (LLM picks optimal start/end)
│   ├── Conjunction fix (handles "But/And/So" starts)
│   └── Hallucination detection (validates summaries)
├── sentence_parser.py      # VTT → sentences with NER
├── typesense_indexer.py    # Index to Typesense
└── clip_extractor.py       # WAV → MP3 clip extraction
```

### Search Pipeline
```
app/services/search_l3.py   # Level 3 commercial-grade search:
├── Phase 1: Query understanding (LLM expands query)
├── Phase 2: Hybrid retrieval (BM25 + vector)
├── Phase 3: LLM reranking (returns topic_id, not index)
└── Phase 4: Categorization (ABOUT / Also discusses / Related)
```

### Frontend
```
static/index.html           # Single-page app
├── Search → calls /v2/search/l3
├── Results → ABOUT (yellow), Also discusses (cyan), Related
└── Audio → Redirects to /clip/{episode}?start_ms=X&end_ms=Y
```

---

## 🐛 Bugs Fixed This Session

### 1. Summary Mismatch Bug
**Problem:** LLM returned wrong indices during reranking, causing mismatched summaries
**Fix:** Changed to return `topic_id` instead of index in reranking
**File:** `app/services/search_l3.py` (lines 335-410)

### 2. Audio Timestamp Bug
**Problem:** Frontend used full MP3 with seek (unreliable), not extracted clips
**Fix:** 
- Backend returns `/clip/{episode}?start_ms=X&end_ms=Y`
- `/clip` endpoint generates clip on demand, redirects to audio file
**Files:** `app/services/search_l3.py`, `app/routers/search.py`, `static/index.html`

### 3. Mid-Sentence Start Bug
**Problem:** Topics could start with "But", "So", etc. without context
**Fix:** Added `_fix_conjunction_starts()` post-processing step
**File:** `app/services/topic_segmentation_v3.py`

---

## 📁 Directory Structure

```
~/clawd/projects/echo-audio-browser/
├── audio/                  # Audio files
│   ├── *.wav              # Source (accurate timestamps)
│   ├── *.mp3              # Fallback playback
│   ├── *.en.vtt           # Transcripts
│   └── clips/             # Generated MP3 clips (cache)
├── data/
│   └── topic_embeddings.json  # Cached OpenAI embeddings
├── app/
│   ├── main.py            # FastAPI app
│   ├── routers/           # API endpoints
│   └── services/          # Business logic
├── static/
│   └── index.html         # Frontend
├── SCALE_UP_PLAN.md       # Detailed scaling plan
├── HANDOFF_SESSION.md     # Previous handoff (V3 features)
└── TUNNEL_URL.txt         # Current public URL
```

---

## 🔑 Credentials & Config

### OpenAI API Key
```bash
cat ~/.clawd/.api-keys/openai.key
```

### Typesense
- **Host:** localhost:8108
- **API Key:** See `.env` file
- **Collections:** `topics`, `sentences`

### Environment
```bash
source venv/bin/activate
export OPENAI_API_KEY=$(cat ~/.clawd/.api-keys/openai.key)
```

---

## 📊 Commands Reference

### Download an Episode
```bash
VIDEO_ID="xxxxxxxxxxx"
yt-dlp -f 'bestaudio[ext=m4a]' -o "audio/${VIDEO_ID}.m4a" --write-auto-subs --sub-lang en "https://youtube.com/watch?v=${VIDEO_ID}"
ffmpeg -i "audio/${VIDEO_ID}.m4a" -ar 44100 -ac 2 "audio/${VIDEO_ID}.wav"
ffmpeg -i "audio/${VIDEO_ID}.m4a" -ab 128k "audio/${VIDEO_ID}.mp3"
```

### Index an Episode
```bash
cd ~/clawd/projects/echo-audio-browser
source venv/bin/activate
export OPENAI_API_KEY=$(cat ~/.clawd/.api-keys/openai.key)
PYTHONPATH=. python3 app/services/pipeline_v3.py audio/VIDEO_ID.en.vtt
```

### Index All Episodes
```bash
for vtt in audio/*.en.vtt; do
    echo "Processing: $vtt"
    PYTHONPATH=. python3 app/services/pipeline_v3.py "$vtt"
    sleep 2
done
```

### Test Search (CLI)
```bash
PYTHONPATH=. python3 -c "
from app.services.search_l3 import search_l3
result = search_l3('your query here', limit=5)
for r in result['about']:
    print(f'{r[\"summary\"][:80]}...')
"
```

### Clear Clip Cache
```bash
rm -rf audio/clips/*.mp3
```

### Check Typesense Stats
```bash
curl -s "http://localhost:8108/collections/topics" -H "X-TYPESENSE-API-KEY: $(grep TYPESENSE_API_KEY .env | cut -d= -f2)" | python3 -m json.tool | grep num_documents
```

---

## ⚠️ Known Limitations

1. **VTT Quality** — YouTube auto-captions can have errors; affects segmentation
2. **Embedding Load Time** — ~5 seconds on first search to load embeddings
3. **Clip Generation** — First play of a clip has ~2-3s delay for ffmpeg
4. **Memory** — Search loads all embeddings; may need optimization at scale

---

## 🎯 Success Criteria for Scale-Up

- [ ] 50 episodes downloaded and converted
- [ ] All episodes indexed with V3 pipeline
- [ ] Search returns relevant results across all content
- [ ] Audio clips play correctly
- [ ] Performance acceptable (< 10s search)
- [ ] Disk space under control

---

## 📞 Questions?

If stuck, check:
1. `SCALE_UP_PLAN.md` — Detailed execution plan
2. `HANDOFF_SESSION.md` — V3 feature documentation
3. `ARCHITECTURE_REVIEW.md` — System design overview
4. Server logs: `tmux capture-pane -t echo -p -S -100`

---

*Ready for scale-up execution!*
