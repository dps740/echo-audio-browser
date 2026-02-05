# Echo Audio Browser — Business & Architecture Plan

## What It Is
Topic-first podcast search engine. Search across podcasts by topic, get a curated playlist of the best segments. "Google for podcast content."

## Cost Per Episode (AI Processing)
| Step | Cost |
|------|------|
| LLM segmentation (gpt-4o-mini) | ~$0.007 |
| Key term extraction (gpt-4o-mini) | ~$0.003 |
| OpenAI embeddings (text-embedding-3-small) | ~$0.0002 |
| **Total per episode** | **~$0.01-0.02** |

At 100 episodes: ~$1-2. At 1,000 episodes: ~$10-20.

## Pricing Model
- **Consumer:** $3-5/month, unlimited search & playback
- **B2B/API (future):** $0.25/episode ingest + $0.001/search query, or $99-499/mo tiered
- **Prosumer/Creator (future):** $29/mo — "make your podcast library searchable"

## No GPU Required
Everything is API-based (OpenAI). Runs on any laptop or cheap VPS. Local GPU only useful for Whisper transcription (nice-to-have, not needed — YouTube captions are free).

---

## Infrastructure Tiers

### MVP: Local (Current)
- Runs on laptop/desktop
- Cost: $0/mo hosting, ~$0.02/episode AI
- Users: 1 (you)

### Tier 1: Small Cloud (10-100 users)
| Item | Monthly Cost |
|------|-------------|
| VPS (DigitalOcean/Hetzner) | $20 |
| S3/R2 audio storage (500GB) | $8 |
| OpenAI API (embeddings + segmentation) | $5-15 |
| Domain + SSL | $1 |
| Auth (Clerk/Auth0 free tier) | $0 |
| **Total** | **~$35-45/mo** |

**Break-even:** 7-15 users @ $3-5/mo

### Tier 2: Serious Cloud (1,000-10,000 users)
| Item | Monthly Cost |
|------|-------------|
| App servers (2-3 instances) | $60-100 |
| Managed vector DB (Pinecone/Qdrant) | $70-150 |
| CDN + audio streaming (Cloudflare R2) | $30-80 |
| OpenAI API | $50-200 |
| PostgreSQL (user data, billing) | $15-30 |
| Stripe fees (~3%) | 3% of revenue |
| Auth + monitoring | $25 |
| **Total** | **~$250-600/mo** |

**Break-even:** 50-200 users @ $3-5/mo
**At 1,000 users @ $5/mo:** $5,000 revenue, ~$4,500 profit
**At 5,000 users @ $5/mo:** $25,000 revenue, ~$24,000 profit

### Tier 3: Scale (25,000+ users)
| Item | Monthly Cost |
|------|-------------|
| Kubernetes / ECS | $300-800 |
| Vector DB cluster | $300-500 |
| CDN + bandwidth (TB scale) | $200-500 |
| AI costs | $500-2,000 |
| Managed DB + cache | $100-200 |
| Ops/monitoring/logging | $100 |
| **Total** | **~$1,500-4,000/mo** |

**Break-even:** 300-800 users @ $5/mo
**At 25,000 users @ $5/mo:** $125k/mo revenue, 97% margin

---

## Concurrency Notes
User counts above are **total subscribers**, not concurrent.
- Typical concurrent = 5-10% of total subscribers
- 100 subscribers → ~5-10 concurrent (Tier 1 VPS handles easily)
- 1,000 subscribers → ~50-100 concurrent (still Tier 1)
- 10,000 subscribers → ~500-1,000 concurrent (Tier 2)

Bottleneck is audio streaming bandwidth, not compute. CDN (Cloudflare free tier) solves this cheaply.

---

## Architecture: Current (MVP)

```
[Browser] → [FastAPI on localhost:8765]
                ├── /player (static HTML)
                ├── /playlists/* (hybrid search)
                ├── /ingest/* (YouTube + LLM pipeline)
                ├── /audio/* (static files)
                └── ChromaDB (local, persistent)
                        └── OpenAI embeddings (API)
```

## Architecture: Cloud Deployment

```
[Browser] → [CDN/Cloudflare]
                ├── Static assets (cached)
                ├── Audio files → [R2/S3 bucket]
                └── API → [FastAPI on VPS]
                            ├── Auth middleware
                            ├── Hybrid search engine
                            ├── Ingest pipeline (background jobs)
                            └── Vector DB (ChromaDB/Pinecone)
                                    └── OpenAI embeddings (API)
```

---

## Search Architecture (Implemented Feb 2026)

### Indexing Pipeline (per episode)
1. YouTube captions extracted via yt-dlp
2. LLM segmentation (gpt-4o-mini) → semantic segments with summaries, topic tags, density scores
3. Key term extraction (gpt-4o-mini) → named entities, concepts
4. Enriched document built: TOPIC + SUMMARY + KEY TERMS + TRANSCRIPT
5. OpenAI embeddings (text-embedding-3-small)
6. Stored in ChromaDB with full metadata

### Search Pipeline (per query)
1. Semantic search → top 50 candidates from ChromaDB
2. Keyword filter & boost → exact matches in transcript, tags, key terms, summary
3. No-match penalty → results without any keyword match get score × 0.3
4. Synonym expansion → "AGI" also matches "artificial general intelligence", etc.
5. Diversity filter → max 3 per episode, max 4 per podcast
6. Quality threshold → minimum combined score
7. Return top N sorted by relevance

### Key Endpoint
- `POST /ingest/reindex` — re-processes all existing content through new pipeline

---

## Testing Checklist (Feb 5, 2026)
- [ ] Pull latest from GitHub
- [ ] Add `OPENAI_API_KEY=sk-...` to `.env` in project root
- [ ] Start server, verify player loads
- [ ] Run re-index: `POST http://localhost:8765/ingest/reindex`
- [ ] Search "AGI" — should return actual AGI-related clips
- [ ] Search "AI" — should return AI-related clips, not random
- [ ] Verify crossfade transitions between clips
- [ ] Test with various queries to assess relevance quality
- [ ] Note any issues for iteration

---

## Key Margins Insight
This is a **97%+ margin business at scale**. AI costs per episode are pennies. Infrastructure costs scale sub-linearly. The moat is UX and search quality, not tech cost. The hard part is distribution, not infrastructure.
