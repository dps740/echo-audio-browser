"""Echo Audio Browser - FastAPI Application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from app.config import settings
from app.routers import indexing, search, playlists, library, download, search_v2

app = FastAPI(
    title=settings.app_name,
    description="Topic-First Audio Browser - Browse podcasts by topic, not episode",
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(indexing.router)
app.include_router(search.router)
app.include_router(search_v2.router)  # New Typesense-based search
app.include_router(playlists.router)
app.include_router(library.router)
app.include_router(download.router)


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": settings.app_name,
        "version": "0.2.0",
        "description": "Topic-First Audio Browser",
        "docs": "/docs",
        "endpoints": {
            "index": "/index/{video_id} - Index a VTT transcript",
            "search": "/search?q=... - Semantic search across all episodes",
            "playlists": "/playlists - Generate playback manifests",
            "library": "/library - Browse indexed content",
            "download": "/ingest/youtube - Download from YouTube",
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


# Mount audio files directory
audio_dir = os.path.join(os.path.dirname(__file__), "..", "audio")
if not os.path.exists(audio_dir):
    audio_dir = os.path.join(os.path.dirname(__file__), "audio")
    os.makedirs(audio_dir, exist_ok=True)
app.mount("/audio", StaticFiles(directory=audio_dir), name="audio")

# Mount static files for web player
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/player")
    async def player():
        """Serve the web player."""
        return FileResponse(os.path.join(static_dir, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
