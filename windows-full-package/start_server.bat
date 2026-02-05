@echo off
echo ============================================
echo    ECHO - Topic-First Audio Browser
echo ============================================
echo.
echo Installing dependencies (first time only)...
pip install fastapi uvicorn chromadb pydantic-settings httpx openai --quiet 2>nul
echo.
echo Starting server...
echo.
echo   Player: http://localhost:8765/player
echo   API:    http://localhost:8765/docs
echo.
echo   Keep this window open!
echo   Open a new cmd window to add podcasts.
echo ============================================
echo.
echo Opening browser...
start http://localhost:8765/player
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
pause
