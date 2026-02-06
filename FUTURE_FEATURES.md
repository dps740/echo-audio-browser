# Echo — Future Features

## Actionable Intelligence Layer

**Status:** Concept (not yet implemented)
**Added:** 2026-02-06

### The Problem

Current Echo finds *discussions about topics*. But podcasts contain **actionable insights** buried in conversation:
- Chamath mentions investing in copper → copper up 26% months later
- Founder shares what actually worked for fundraising
- Expert predicts regulatory change before it happens

Current tagging captures "Tech Investing and Founders" when it should capture **"Chamath: Long Copper (infrastructure thesis, Oct 2024)"**

### The Solution

Second-pass LLM extraction on segments asking: *"Is there an actionable insight here? If so, extract it structured."*

Most segments return nothing. The gems get captured as structured data.

### Output Format

```json
{
  "type": "investment_thesis",
  "speaker": "Chamath",
  "asset": "Copper", 
  "direction": "Long",
  "thesis": "Infrastructure spending, supply constraints, EV demand",
  "conviction": "High",
  "timeframe": "12-18 months",
  "date": "2024-10-15",
  "segment_id": "all-in-ep-xxx_seg_4"
}
```

### What This Enables

- **Search:** "Show me all commodity calls from last 6 months"
- **Track:** "Which predictions played out?" (with price data integration)
- **Alert:** "Notify me when hosts make specific trade calls"
- **Filter:** By speaker, asset class, conviction level, thesis type

---

## Actionable Insight Categories

Beyond investments, other **actionable insight types** worth extracting:

### 1. Investment Theses
- Asset recommendations (long/short)
- Sector calls
- Macro predictions
- "I'm putting money into X because Y"

### 2. Predictions & Forecasts
- Market predictions (with timeframe)
- Technology adoption timelines
- Political/regulatory forecasts
- "By 2026, X will happen because Y"

### 3. Career & Business Advice
- Hiring insights ("We look for X in candidates")
- Fundraising tactics ("What actually worked was...")
- Negotiation strategies
- "If I were starting over, I would..."

### 4. Book/Resource Recommendations
- Specific books mentioned with context
- Tools and software recommendations
- Courses, newsletters, people to follow
- "The book that changed how I think about X"

### 5. Contrarian Takes
- "Everyone thinks X but actually Y"
- Unpopular opinions with reasoning
- Counter-consensus views
- Useful for identifying non-obvious opportunities

### 6. Lessons from Failure
- "We tried X and it failed because Y"
- Post-mortems and what they'd do differently
- Expensive mistakes to avoid
- "The biggest mistake I made was..."

### 7. Insider Knowledge
- Industry dynamics not publicly discussed
- How things actually work vs. perception
- Regulatory/political insights
- "What most people don't understand about X"

### 8. Health & Longevity
- Specific protocols mentioned
- Supplement/intervention recommendations
- "I started doing X and noticed Y"
- Expert recommendations with evidence

### 9. Relationship & Network Insights  
- How successful people met key contacts
- Networking strategies that worked
- "The introduction that changed everything"

### 10. Decision Frameworks
- Mental models mentioned
- How experts make specific decisions
- "When I face X situation, I always Y"

---

## Implementation Notes

### Approach
1. Run extraction prompt on each segment
2. Most return empty (no actionable insight)
3. Store structured insights in separate collection
4. Enable filtered search across insight types

### Extraction Prompt (Draft)
```
Analyze this podcast segment for actionable insights.

An actionable insight is a specific, concrete piece of information that someone could ACT on — not general discussion or opinion.

Look for:
- Investment theses (specific assets, directions, reasoning)
- Predictions with timeframes
- Specific advice based on experience
- Book/resource recommendations with context
- Contrarian takes with reasoning
- Lessons from specific failures

If there's an actionable insight, extract:
- Type (investment/prediction/advice/resource/contrarian/lesson)
- Speaker (if identifiable)
- The specific insight
- Supporting reasoning
- Confidence/conviction level
- Timeframe (if applicable)

If no actionable insight, return null.
```

### Cost Consideration
Running extraction on all segments = expensive. Options:
- Run on new segments only (going forward)
- Run on high-density segments only (density_score > 0.7)
- Run on-demand when user searches specific categories

---

## Current State (2026-02-06)

**Search "investments" returns:**
- Psychology of investing discussions
- General healthcare AI content  
- Interview intros

**Gap:** No structured extraction of actual investment theses or actionable recommendations.

**Next step:** Decide if/when to implement extraction layer.
