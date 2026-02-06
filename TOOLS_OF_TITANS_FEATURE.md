# Echo — "Tools of Titans" Auto-Generation

**Status:** Concept  
**Added:** 2026-02-06  
**Inspiration:** Tim Ferriss's "Tools of Titans" — compiled routines, habits, tools, and wisdom from 200+ podcast guests into a searchable reference.

---

## The Vision

Turn any podcast library into an auto-generated "Tools of Titans" — structured, searchable compilation of actionable personal intel from guests.

**Query examples:**
- "What morning routines have guests mentioned?"
- "What books have been recommended and why?"
- "What supplements/health protocols do people use?"
- "What productivity tools do successful people rely on?"
- "What advice would guests give their younger selves?"

---

## Tools of Titans Categories

Based on Ferriss's book structure + expanded:

### 🌅 Routines & Rituals
- **Morning routines** — Wake time, first activities, rituals
- **Evening routines** — Wind-down, sleep prep
- **Work routines** — Deep work habits, scheduling
- **Weekly rituals** — Review practices, recurring activities
- **Travel routines** — How they stay productive on the road

### 💪 Health & Body
- **Exercise protocols** — Specific workouts, frequency, preferences
- **Diet/nutrition** — What they eat, avoid, timing
- **Supplements** — What they take and why
- **Sleep optimization** — Hours, hacks, tools
- **Recovery practices** — Sauna, cold plunge, massage, etc.
- **Biometrics tracked** — What they measure

### 🧠 Mind & Performance
- **Meditation/mindfulness** — Practice details
- **Journaling** — What kind, prompts, frequency
- **Learning habits** — How they consume information
- **Focus techniques** — How they maintain concentration
- **Stress management** — What they do when overwhelmed

### 🛠 Tools & Tech
- **Productivity apps** — What software they rely on
- **Hardware** — Devices, setups, gear
- **Books** — Recommendations with context (why this book?)
- **Podcasts/newsletters** — What they consume
- **Services** — Coaches, assistants, services they use

### 💼 Work & Career
- **Hiring insights** — What they look for
- **Management practices** — How they lead
- **Decision frameworks** — How they make choices
- **Negotiation tactics** — What works for them
- **Time management** — How they allocate attention

### 💰 Money & Investing
- **Investment philosophy** — How they think about money
- **Specific allocations** — Where they put capital
- **Financial habits** — Saving, spending patterns
- **Money mistakes** — What they'd do differently

### 🎯 Life Philosophy
- **Core beliefs** — What principles guide them
- **Advice to younger self** — What they'd tell 20-year-old them
- **Biggest lessons** — Hard-won wisdom
- **Regrets** — What they'd change
- **Definition of success** — How they measure it
- **Fears** — What they worry about

### 🤝 Relationships & Network
- **How they met key people** — Origin stories
- **Networking approach** — How they build relationships
- **Mentors** — Who influenced them
- **Communication habits** — How they stay in touch

### ⚡ Quotes & Sound Bites
- **Memorable phrases** — Quotable moments
- **Contrarian takes** — Surprising opinions
- **Predictions** — What they think will happen

---

## Extraction Approach

### Per-Segment Analysis
Run extraction prompt looking for Tools of Titans categories:

```
Analyze this podcast segment for "Tools of Titans" style personal insights.

Look for SPECIFIC, ACTIONABLE personal practices mentioned by the speaker:
- Routines (morning, evening, work)
- Health practices (exercise, diet, supplements, sleep)
- Tools & technology they use
- Books/resources they recommend
- Productivity techniques
- Decision frameworks
- Life advice or lessons learned
- Specific habits with details

For each insight found, extract:
- Category (routine/health/tool/book/advice/etc.)
- Speaker (if identifiable)
- The specific practice/recommendation
- Context (why they do it, what problem it solves)
- Specificity level (vague mention vs. detailed protocol)

Only extract CONCRETE practices, not general discussion about topics.
Return null if no specific personal practices mentioned.
```

### Example Output
```json
{
  "category": "morning_routine",
  "speaker": "Andrew Ross Sorkin",
  "insight": "Wakes at 4:30 AM, works on Dealbook newsletter before hosting CNBC 6-9 AM",
  "context": "Balances live TV with written content by splitting the morning",
  "specificity": "high",
  "segment_id": "xxx"
}
```

```json
{
  "category": "supplement",
  "speaker": "Guest on brain health episode",
  "insight": "Takes omega-3 DHA specifically for brain structure maintenance",
  "context": "Dementia prevention protocol",
  "specificity": "medium",
  "segment_id": "yyy"
}
```

---

## Query Interface

### Category Browsing
"Show me all [category] mentions"
→ Returns all extracted insights for that category, grouped by speaker

### Cross-Category Search  
"What does [Person] do?" 
→ All insights attributed to that speaker across categories

### Comparison
"Compare morning routines across guests"
→ Side-by-side view of different approaches

### Frequency Analysis
"What's the most commonly recommended book?"
→ Aggregation across all book recommendations

### Context-Rich Results
Each result links back to the source segment with timestamp for full context

---

## UI Concept

### "Titans View" 
A browsable interface organized like the book:
- Left sidebar: Categories
- Main view: Cards with insights
- Each card: Speaker photo, insight summary, "Listen" button to jump to source
- Filter by: Guest, podcast, recency, specificity

### "Build Your Own Titans"
- Select which podcasts to include
- Select which categories interest you
- Generate personalized compilation
- Export as PDF/document

---

## Current State vs. Target

| Capability | Current | Target |
|------------|---------|--------|
| Find topic discussions | ✅ | ✅ |
| Find specific routines | ⚠️ Only if tagged | ✅ Structured extraction |
| "All book recommendations" | ❌ | ✅ Category aggregation |
| "What does X do?" | ❌ | ✅ Speaker-centric view |
| Compare across guests | ❌ | ✅ Cross-reference |
| Export compilation | ❌ | ✅ Generate document |

---

## Implementation Priority

**Phase 1: Extraction Layer**
- Add Tools of Titans extraction to segment processing
- Store structured insights alongside segments

**Phase 2: Category Search**
- Enable "show all [category]" queries
- Basic aggregation and listing

**Phase 3: Speaker View**
- "What does [Person] do?" across categories
- Speaker profiles with compiled insights

**Phase 4: Comparison & Export**
- Side-by-side comparisons
- Generate "mini Titans" documents from library

---

## Why This Matters

Tim Ferriss spent years manually compiling Tools of Titans from his interviews. 

With the right extraction layer, Echo could:
1. Generate this automatically from any podcast library
2. Keep it updated as new episodes arrive
3. Make it searchable in ways a book can't be
4. Personalize based on user interests

**The podcast becomes a living, queryable knowledge base of human practices and wisdom.**
