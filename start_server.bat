@echo off
echo Starting Echo Audio Browser...
echo.
echo Open http://localhost:8765/player in your browser
echo.
python -m uvicorn app.main:app --port 8765 --host 127.0.0.1
pause
