=============================================================
ECHO - Topic-First Audio Browser
Windows Setup Guide
=============================================================

WHAT IT DOES
------------
Search podcasts by topic, not episode. Type "AI safety" 
and get a playlist of 60-second clips from Lex Fridman, 
Sam Harris, Huberman, etc. — right where the topic starts.

Everything runs locally on your PC. No accounts, no cloud.


STEP 1: INSTALL PYTHON (skip if you have it)
---------------------------------------------
1. Go to https://www.python.org/downloads/
2. Download Python 3.11 or newer
3. Run installer
4. IMPORTANT: Check "Add Python to PATH" during install
5. Click "Install Now"

To verify: open Command Prompt, type: python --version


STEP 2: INSTALL DEPENDENCIES
-----------------------------
Open Command Prompt (Win+R, type "cmd", press Enter):

    pip install yt-dlp requests fastapi uvicorn chromadb pydantic-settings httpx


STEP 3: START THE SERVER
------------------------
Double-click "start_server.bat"

Wait until you see: "Uvicorn running on http://127.0.0.1:8765"

Leave this window open!


STEP 4: ADD PODCASTS
--------------------
Open a NEW Command Prompt window.
Navigate to this folder:

    cd C:\path\to\echo-windows-full-package

See available podcasts:

    python echo_batch_ingest.py

Ingest all curated podcasts (20 latest episodes each):

    python echo_batch_ingest.py --all --limit 20

Or just one channel:

    python echo_batch_ingest.py --channel @lexfridman --limit 10

Or a single video:

    python echo_ingest.py "https://www.youtube.com/watch?v=VIDEO_ID"

Audio files download to the "audio" folder (~50-100 MB each).
Captions are extracted and indexed automatically.
Progress is saved — you can stop and restart anytime.


STEP 5: SEARCH AND PLAY
------------------------
Open browser: http://localhost:8765/player

Search for any topic. Click a result to play.
Audio plays from your local files — instant, no buffering.


FOLDER STRUCTURE
----------------
echo-windows-full-package/
├── start_server.bat         ← Double-click to start
├── echo_ingest.py           ← Single video ingestion
├── echo_batch_ingest.py     ← Batch ingestion (channels)
├── ingest_progress.json     ← Auto-saved progress
├── audio/                   ← Downloaded MP3 files
├── chroma_data/             ← Search index database
├── app/                     ← Server code
└── static/                  ← Web player


STORAGE
-------
Each podcast episode: ~50-100 MB audio + ~1 KB index data
20 episodes from 20 channels (400 episodes): ~20-40 GB
Start small (--limit 5) and add more as needed.


GPU TRANSCRIPTION (OPTIONAL)
----------------------------
If YouTube captions aren't available for some videos:

1. Install: pip install faster-whisper
2. Use: python echo_ingest.py "URL" --whisper

Your GTX 1050 Ti will handle the "base" model well.


TROUBLESHOOTING
---------------
"python not found"     → Reinstall Python, check "Add to PATH"
"pip not found"        → Same as above  
Server won't start     → Check port 8765 isn't in use
No captions found      → Try --whisper flag
Ingest fails           → Check internet connection, try again

=============================================================
