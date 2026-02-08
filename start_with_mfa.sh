#!/bin/bash
cd /home/ubuntu/clawd/projects/echo-audio-browser
export PATH="/home/ubuntu/.deno/bin:/home/ubuntu/miniconda3/envs/mfa/bin:$PATH"
./venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8765
