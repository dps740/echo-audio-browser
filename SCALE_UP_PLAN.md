# Echo Scale-Up Plan: 16 → 50 Podcasts

**Goal:** Expand from 16 episodes to 50 episodes for realistic PoC demonstration
**Target:** Production-quality search across ~50 hours of podcast content

---

## Phase 1: Content Selection (30 min)

### Criteria for Episode Selection
- **Quality:** Clear audio, good transcription
- **Variety:** Mix of topics, guests, episode lengths
- **Recency:** Prefer recent episodes (2024-2025) for relevance
- **Uniqueness:** Diverse topics to test search breadth

### Recommended Sources
1. **All-In Podcast** — Already have 16, add 10-15 more recent episodes
2. **Lex Fridman** — Long-form, high-quality transcripts (5-10 episodes)
3. **Tim Ferriss** — Varied guests, good topic diversity (5-10 episodes)
4. **Joe Rogan (select)** — High-profile guests, varied topics (5-10 episodes)
5. **Other tech podcasts** — Acquired, My First Million, etc.

### Episode List Template
| # | Podcast | Episode Title | YouTube ID | Duration | Status |
|---|---------|---------------|------------|----------|--------|
| 1 | All-In | Episode XXX | xxxxxxxxxx | 2h 15m | Pending |
| ... | ... | ... | ... | ... | ... |

---

## Phase 2: Content Acquisition (2-3 hours)

### For Each Episode:
```bash
# 1. Download audio + captions
yt-dlp -f 'bestaudio[ext=m4a]' -o 'audio/%(id)s.%(ext)s' --write-auto-subs --sub-lang en "https://youtube.com/watch?v=VIDEO_ID"

# 2. Convert to WAV (for accurate clip extraction)
ffmpeg -i audio/VIDEO_ID.m4a -ar 44100 -ac 2 audio/VIDEO_ID.wav

# 3. Convert to MP3 (fallback playback)
ffmpeg -i audio/VIDEO_ID.m4a -ab 128k audio/VIDEO_ID.mp3

# 4. Rename VTT
mv "audio/VIDEO_ID.en.vtt" "audio/VIDEO_ID.en.vtt"
```

### Batch Script
Create `scripts/download_episode.sh`:
```bash
#!/bin/bash
VIDEO_ID=$1
echo "Downloading $VIDEO_ID..."
yt-dlp -f 'bestaudio[ext=m4a]' -o "audio/${VIDEO_ID}.m4a" --write-auto-subs --sub-lang en "https://youtube.com/watch?v=${VIDEO_ID}"
ffmpeg -i "audio/${VIDEO_ID}.m4a" -ar 44100 -ac 2 "audio/${VIDEO_ID}.wav"
ffmpeg -i "audio/${VIDEO_ID}.m4a" -ab 128k "audio/${VIDEO_ID}.mp3"
echo "Done: $VIDEO_ID"
```

---

## Phase 3: Indexing Pipeline (4-6 hours for 50 episodes)

### Indexing Time Estimates
- Per episode: ~5-8 minutes (depends on length)
- 50 episodes: ~4-6 hours total
- LLM API cost: ~$2-5 (gpt-4o-mini for segmentation)

### Batch Indexing Script
```bash
#!/bin/bash
cd ~/clawd/projects/echo-audio-browser
source venv/bin/activate
export OPENAI_API_KEY=$(cat ~/.clawd/.api-keys/openai.key)

for vtt in audio/*.en.vtt; do
    echo "Processing: $vtt"
    PYTHONPATH=. python3 app/services/pipeline_v3.py "$vtt"
    sleep 2  # Rate limit protection
done
```

### Sequential Processing (Recommended)
- Process one episode at a time
- Log progress to file
- Resume capability if interrupted

---

## Phase 4: Quality Validation (1-2 hours)

### Test Queries
After indexing, verify search quality with diverse queries:
- Tech: "AI", "Bitcoin", "Tesla", "OpenAI"
- People: "Elon Musk", "Sam Altman", "Naval"
- Topics: "investing", "startups", "health", "politics"
- Abstract: "advice for founders", "future of work"

### Validation Checklist
- [ ] Search returns results for all test queries
- [ ] ABOUT/Also discusses/Related classification is accurate
- [ ] Audio clips play correctly (start/end alignment)
- [ ] No duplicate results
- [ ] Summaries match actual content
- [ ] Response time < 10 seconds

---

## Phase 5: Infrastructure Considerations

### Storage Requirements
| Content | Per Episode | 50 Episodes |
|---------|-------------|-------------|
| WAV | ~170 MB | ~8.5 GB |
| MP3 | ~85 MB | ~4.2 GB |
| VTT | ~1 MB | ~50 MB |
| Clips (cache) | ~50 MB | ~2.5 GB |
| **Total** | ~300 MB | **~15 GB** |

### Typesense Scaling
- Current: 438 topics, 10K sentences
- Expected: ~1,400 topics, ~33K sentences
- Memory: Should handle fine on current instance

### Embedding Cache
- Current: 438 embeddings in `data/topic_embeddings.json`
- Expected: ~1,400 embeddings (~50 MB JSON file)
- Consider: Move to SQLite or vector DB for faster loading

---

## Phase 6: Monitoring & Optimization

### Performance Metrics to Track
- Search latency (target: < 5 seconds)
- Clip generation time (target: < 3 seconds)
- Memory usage during search
- Typesense query performance

### Potential Optimizations
1. **Pre-generate popular clips** — Cache during off-hours
2. **Embedding pre-loading** — Load on startup, keep in memory
3. **Search result caching** — Cache common queries (5 min TTL)
4. **Parallel indexing** — Index multiple episodes simultaneously (careful with rate limits)

---

## Timeline Summary

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| 1. Content Selection | 30 min | Episode list |
| 2. Content Acquisition | 2-3 hours | yt-dlp, storage |
| 3. Indexing | 4-6 hours | OpenAI API |
| 4. Validation | 1-2 hours | Completed indexing |
| 5. Infrastructure | As needed | - |
| 6. Optimization | Ongoing | - |
| **Total** | **8-12 hours** | |

---

## Success Criteria

1. **50 episodes indexed** with V3 segmentation
2. **Search quality** maintained or improved
3. **Audio alignment** accurate (clips start at natural points)
4. **Performance** acceptable (< 10s search, < 3s clip load)
5. **Storage** within available disk space

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| LLM rate limits | Indexing delays | Add delays, batch wisely |
| Disk space | Can't store all content | Monitor usage, compress old clips |
| Search latency | Poor UX | Optimize embeddings loading |
| Transcription quality | Bad segmentation | Manual review of problem episodes |

---

*Created: 2026-02-11*
*Status: Ready for execution*
