=============================================================
ECHO - Topic-First Audio Browser
Windows Installation Guide
=============================================================

STEP 1: INSTALL PYTHON
-----------------------
1. Go to https://www.python.org/downloads/
2. Download Python 3.11 or newer
3. Run installer
4. IMPORTANT: Check "Add Python to PATH" during install
5. Click "Install Now"


STEP 2: INSTALL YT-DLP
-----------------------
Open Command Prompt (press Win+R, type "cmd", press Enter)
Type this command and press Enter:

    pip install yt-dlp requests


STEP 3: START THE SERVER
------------------------
1. Double-click "start_server.bat"
2. Wait for it to say "Uvicorn running"
3. Open your browser to: http://localhost:8765/player


STEP 4: ADD A YOUTUBE VIDEO
---------------------------
Open a NEW Command Prompt window (keep the server running!)
Navigate to this folder:

    cd C:\path\to\echo-windows-package

Run the ingest script with a YouTube URL:

    python echo_ingest.py "https://www.youtube.com/watch?v=VIDEO_ID"

Example with Lex Fridman:

    python echo_ingest.py "https://www.youtube.com/watch?v=Mde2q7GFCrw"


STEP 5: SEARCH AND PLAY
-----------------------
1. Go back to http://localhost:8765/player
2. Search for a topic (e.g., "AI safety")
3. Click a result to play


OPTIONAL: USE YOUR GPU FOR TRANSCRIPTION
----------------------------------------
If YouTube captions aren't available or you want better accuracy:

1. Install CUDA from NVIDIA (if not already installed)
2. Run: pip install faster-whisper
3. Use the --whisper flag:

    python echo_ingest.py "https://www.youtube.com/watch?v=VIDEO_ID" --whisper


TROUBLESHOOTING
---------------
- "python not found" → Reinstall Python, check "Add to PATH"
- "pip not found" → Same as above
- Server won't start → Make sure port 8765 isn't in use
- No captions found → Try --whisper flag


=============================================================
