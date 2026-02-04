# Echo Windows Ingest Client

## Setup

1. Install Python 3.10+ from https://python.org
2. Install yt-dlp: `pip install yt-dlp requests`
3. (Optional) For Whisper: `pip install faster-whisper` or `pip install openai-whisper`
4. Edit `echo_ingest.py` and set `SERVER_URL` to your Echo server

## Usage

### Using YouTube captions (fast, free):
```
python echo_ingest.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Using local Whisper (accurate, uses GPU):
```
python echo_ingest.py "https://www.youtube.com/watch?v=VIDEO_ID" --whisper --model base
```

Models: tiny, base, small, medium, large (larger = more accurate, slower)

### Specify server:
```
python echo_ingest.py "URL" --server https://your-server.com
```

## GPU Acceleration

For faster-whisper with your GTX 1050 Ti:
1. Install CUDA 11.x from NVIDIA
2. `pip install faster-whisper`

The `base` model runs well on 4GB VRAM. For better accuracy, try `small`.
