# Echo Business Model

*Last updated: 2026-02-06*

## Product

Podcast search tool. Index podcast transcripts → users search by topic → play relevant clips.

## Revenue Model

**Subscription:** $10/user/month (assumed starting point)

---

## Cost Structure

### API Costs (OpenAI)

| Operation | Cost |
|-----------|------|
| Ingestion (per episode) | **$0.03** |
| Search (per query) | **$0.005** |

**Breakdown:**
- Ingestion: GPT-4o-mini segmentation (~160k tokens in, ~4k out) + embeddings
- Search: Query expansion + relevance filtering (~25k tokens)
- Transcription: **Free** (YouTube transcripts or local Whisper)

### Infrastructure Costs

| Scale | Monthly Cost |
|-------|--------------|
| MVP (10-50 users) | $30-40/mo |
| Growth (100-500 users) | $50-100/mo |
| Scale (1,000+ users) | $150+/mo |

Components: VPS, database, vector DB, audio storage, CDN

---

## Monthly P&L Projections

**Assumptions:**
- $10/user/month
- 50 searches/user/month
- New episodes indexed scales with users

| Users | Revenue | Infra | Ingestion | Searches | **Profit** | **Margin** |
|-------|---------|-------|-----------|----------|------------|------------|
| 10 | $100 | $30 | $1.50 | $2.50 | **$66** | 66% |
| 50 | $500 | $40 | $3 | $12.50 | **$444** | 89% |
| 100 | $1,000 | $50 | $6 | $25 | **$919** | 92% |
| 500 | $5,000 | $100 | $15 | $125 | **$4,760** | 95% |
| 1,000 | $10,000 | $150 | $30 | $250 | **$9,570** | 96% |

**Key insight:** API costs are tiny (~3% of revenue at scale). Main costs are time and marketing.

---

## Content Strategy

### Default: Self-Hosted Clips
- Best UX: instant playback, no ads
- Store audio clips (30s-3min segments)
- Higher engagement, better experience

### Fallback: YouTube Embed
- Zero copyright risk (linking ≠ infringement)
- Embed at exact timestamp: `youtube.com/embed/VIDEO_ID?start=125`
- Video option comes free for podcasts with YouTube versions

### Copyright Approach

**Risk assessment:** Low-moderate

**Why risk is acceptable:**
1. Most podcasters want discovery — we drive listeners to them
2. DMCA = takedown notice first, not lawsuit
3. Easy compliance — remove content or switch to embed
4. Podcasts on YouTube are already public
5. We monetize search, not the content itself

**If complaint received:**
1. Immediately remove self-hosted clips for that show
2. Switch to YouTube embed mode (keep it indexed, change playback source)
3. Respond to DMCA within required timeframe
4. Document the takedown

**Content we index:**
- Public podcasts with YouTube presence (safest)
- Popular shows where discovery benefits creator
- Avoid: Paywalled content, exclusive subscriber content

---

## Video Option

Many podcasts have YouTube video versions. When available:
- Offer video playback option
- Same timestamp logic
- Better for visual content (charts, demos, reactions)
- Uses YouTube embed (zero hosting cost)

---

## Unit Economics Summary

| Metric | Value |
|--------|-------|
| CAC target | <$30 (3 months payback) |
| LTV (12mo, 5% churn) | ~$100 |
| Gross margin at scale | 95%+ |
| Break-even users | 5-10 |

The economics are excellent. Primary investments needed:
1. Development time
2. Content acquisition/indexing
3. Marketing and user acquisition
