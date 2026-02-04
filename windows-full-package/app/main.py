"""Echo Audio Browser - FastAPI Application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from app.config import settings
from app.routers import feeds, segments, playlists, ingest


# Create app
app = FastAPI(
    title=settings.app_name,
    description="Topic-First Audio Browser - Browse podcasts by topic, not episode",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware (allow all for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(feeds.router)
app.include_router(segments.router)
app.include_router(playlists.router)
app.include_router(ingest.router)


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": settings.app_name,
        "version": "0.1.0",
        "description": "Topic-First Audio Browser",
        "docs": "/docs",
        "endpoints": {
            "feeds": "/feeds - Manage podcast subscriptions",
            "segments": "/segments - Search and ingest segments",
            "playlists": "/playlists - Generate playback manifests",
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


# Mount static files for web player (if exists)
static_dir = os.path.join(os.path.dirname(__file__), "..", "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    
    @app.get("/player")
    async def player():
        """Serve the web player."""
        return FileResponse(os.path.join(static_dir, "index.html"))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
