# Desktop Setup for Whisper Transcription

## Requirements
- Python 3.10+
- NVIDIA GPU with CUDA (GTX 1050 Ti or better)
- yt-dlp for downloading

## Install faster-whisper with CUDA

```bash
# Install CUDA toolkit (if not already)
# Windows: Download from https://developer.nvidia.com/cuda-downloads

# Install faster-whisper with CUDA support
pip install faster-whisper

# Verify CUDA is working
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Install yt-dlp

```bash
# Windows
pip install yt-dlp

# Or download exe from: https://github.com/yt-dlp/yt-dlp/releases
```

## Model Sizes (VRAM Usage)

| Model  | VRAM (int8) | Speed | Accuracy |
|--------|-------------|-------|----------|
| tiny   | ~1 GB       | 32x   | Basic    |
| base   | ~1 GB       | 16x   | Good     |
| small  | ~2 GB       | 6x    | Better   |
| medium | ~5 GB       | 2x    | Great    |
| large  | ~10 GB      | 1x    | Best     |

**For GTX 1050 Ti (4GB VRAM):** Use `small` or `base` with int8.

## Running

The Echo UI will:
1. Download audio via yt-dlp
2. Load faster-whisper with CUDA + int8
3. Transcribe and save to `app/transcripts/`

Check the console for:
```
[Whisper] Loading base model with CUDA int8...
[Whisper] Transcribing with cuda (int8): ...
```

If CUDA fails, it falls back to CPU automatically.
