"""Echo Audio Browser - FastAPI Application."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from app.config import settings
from app.routers import feeds, segments, playlists, ingest, library, youtube_ingest, test_v3, v4_segments


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
app.include_router(library.router)
app.include_router(youtube_ingest.router)
app.include_router(test_v3.router)
app.include_router(v4_segments.router)


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


@app.get("/debug/search")
async def debug_search(q: str = "AI"):
    """Debug search - test ChromaDB query directly and report any errors."""
    import chromadb
    import traceback
    
    results = {"query": q, "steps": []}
    
    # Step 1: Open ChromaDB
    try:
        client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        results["steps"].append({"step": "open_chromadb", "status": "ok", "path": settings.chroma_persist_dir})
    except Exception as e:
        results["steps"].append({"step": "open_chromadb", "status": "error", "error": str(e)})
        return results
    
    # Step 2: Get collection
    try:
        collection = client.get_or_create_collection("segments")
        count = collection.count()
        results["steps"].append({"step": "get_collection", "status": "ok", "count": count})
    except Exception as e:
        results["steps"].append({"step": "get_collection", "status": "error", "error": str(e)})
        return results
    
    # Step 3: Peek at data
    try:
        peek = collection.peek(limit=2)
        sample_meta = peek["metadatas"][0] if peek["metadatas"] else None
        results["steps"].append({
            "step": "peek_data",
            "status": "ok",
            "sample_ids": peek["ids"][:2],
            "sample_metadata": sample_meta,
            "has_documents": bool(peek.get("documents") and peek["documents"][0]),
        })
    except Exception as e:
        results["steps"].append({"step": "peek_data", "status": "error", "error": str(e)})
    
    # Step 4: Query WITHOUT where filter
    try:
        qr = collection.query(
            query_texts=[q],
            n_results=5,
            include=["metadatas", "distances"]
        )
        hits = len(qr["ids"][0]) if qr["ids"] else 0
        results["steps"].append({
            "step": "query_no_filter",
            "status": "ok",
            "hits": hits,
            "top_distances": qr["distances"][0][:3] if qr.get("distances") and qr["distances"][0] else [],
            "top_episodes": [m.get("episode_title", "?") for m in qr["metadatas"][0][:3]] if hits > 0 else [],
        })
    except Exception as e:
        results["steps"].append({
            "step": "query_no_filter",
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()[-500:],
        })
    
    # Step 5: Query WITH density filter
    try:
        qr2 = collection.query(
            query_texts=[q],
            n_results=5,
            where={"density_score": {"$gte": 0.3}},
            include=["metadatas", "distances"]
        )
        hits2 = len(qr2["ids"][0]) if qr2["ids"] else 0
        results["steps"].append({
            "step": "query_with_density_filter",
            "status": "ok",
            "hits": hits2,
        })
    except Exception as e:
        results["steps"].append({
            "step": "query_with_density_filter",
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()[-500:],
        })
    
    # Step 6: Test the full playlist path
    try:
        from app.services.vectordb import search_segments
        segs = await search_segments(query=q, limit=10, min_density=0.0)
        results["steps"].append({
            "step": "vectordb_search_segments",
            "status": "ok",
            "hits": len(segs),
        })
    except Exception as e:
        results["steps"].append({
            "step": "vectordb_search_segments",
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc()[-500:],
        })
    
    return results


# Mount audio files directory
# Mount the root audio directory (where actual files are)
audio_dir = os.path.join(os.path.dirname(__file__), "..", "audio")
if not os.path.exists(audio_dir):
    audio_dir = os.path.join(os.path.dirname(__file__), "audio")
    os.makedirs(audio_dir, exist_ok=True)
app.mount("/audio", StaticFiles(directory=audio_dir), name="audio")

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
    uvicorn.run(app, host="0.0.0.0", port=8765)
