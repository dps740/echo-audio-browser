@echo off
echo Starting Echo Server...
echo.
echo Installing dependencies (first time only)...
pip install fastapi uvicorn chromadb pydantic-settings httpx openai anthropic --quiet
echo.
echo Starting server at http://localhost:8765
echo Open http://localhost:8765/player in your browser
echo.
python -m uvicorn app.main:app --host 127.0.0.1 --port 8765
pause
