@echo off
echo Starting Echo Audio Browser...
echo.
start http://localhost:8765/player
echo Opening browser to http://localhost:8765/player
echo.
python -m uvicorn app.main:app --port 8765 --host 127.0.0.1
pause
