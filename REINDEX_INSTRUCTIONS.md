# Echo Audio Browser - Reindex Instructions

**Copy-paste this to start reindexing in a new session:**

---

## Task: Reindex Echo Episodes

Index all 15 podcast episodes with the V3 persistence fix. Run sequentially (one at a time), report after each batch of 5.

**Location:** `~/clawd/projects/echo-audio-browser`
**Server:** Start with `cd ~/clawd/projects/echo-audio-browser && source venv/bin/activate && python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8765 &`
**Endpoint:** `POST /v3/segment-refined/{video_id}`

**Episodes to index (15 total):**
```
BvhFuEp55X0  (JRE)
UPfN2G0RyQM  (JRE)
EV7WhVT270Q  (Lex) - may fail, caption-level VTT
Z-FRe5AKmCU  (Lex) - may fail, caption-level VTT
kX7zW2TIriM  (My First Million)
xNk2QgZ12Rk  (My First Million)
sY7WI669CFQ  (Sean Carroll)
mlYSvE3XZcQ  (Sean Carroll)
NWZZEa9BURw  (Knowledge Project)
Wp1Wn6QkkKU  (Knowledge Project)
e4PBgfI4LW0  (Tim Ferriss) - ALREADY DONE
_0dxe6Sci6Y  (Tim Ferriss)
gXY1kx7zlkk  (All-In)
w2BqPnVKVo4  (All-In)
wTiHheA40nI  (All-In)
```

**Rules:**
1. ONE episode at a time (no parallel, no sub-agents)
2. Wait for each to complete before starting next
3. Log results to `INDEXING_TIMES.md` after each episode
4. Report to David after every 5 episodes
5. If an episode fails, note it and move on (don't retry infinitely)

**Command per episode:**
```bash
time curl -s -X POST "http://localhost:8765/v3/segment-refined/{VIDEO_ID}" | jq -r '.episode + ": " + (.total_segments | tostring) + " segments"'
```

**Expected total time:** 65-85 minutes

---

**Verification after completion:**
```bash
cd ~/clawd/projects/echo-audio-browser && source venv/bin/activate && python3 -c "
import chromadb
client = chromadb.PersistentClient(path='./chroma_data')
col = client.get_collection('v3_segments')
print(f'Total indexed: {len(col.get()[\"ids\"])} episodes')
"
```
