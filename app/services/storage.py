"""ChromaDB storage and search for V4 snippet-embedded segments."""

import chromadb
import openai
from typing import List, Optional

from app.config import settings
from app.services.segmentation import IndexedSegment

# Singleton ChromaDB client
_chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)


def _get_collection():
    """Get or create the v4_segments collection."""
    return _chroma_client.get_or_create_collection(
        name="v4_segments",
        metadata={"description": "Snippet-embedded segments for semantic search"}
    )


def _embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed texts with OpenAI text-embedding-3-small."""
    client = openai.OpenAI(api_key=settings.openai_api_key)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [e.embedding for e in response.data]


def _embed_query(query: str) -> List[float]:
    """Embed a single search query."""
    return _embed_texts([query])[0]


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_episode(
    video_id: str,
    episode_title: str,
    podcast_name: str,
    segments: List[IndexedSegment]
):
    """
    Save indexed segments to ChromaDB.

    Embeds all snippets in one batched API call, then upserts.
    Deletes any existing segments for the episode first.
    """
    collection = _get_collection()

    # Delete existing segments for this episode
    try:
        existing = collection.get(where={"video_id": video_id})
        if existing['ids']:
            collection.delete(ids=existing['ids'])
    except Exception:
        pass

    snippets = [seg.snippet for seg in segments]
    embeddings = _embed_texts(snippets)

    ids = [f"{video_id}_{i}" for i in range(len(segments))]

    metadatas = [
        {
            "video_id": video_id,
            "episode_title": episode_title,
            "podcast_name": podcast_name,
            "segment_index": i,
            "start_ms": seg.start_ms,
            "end_ms": seg.end_ms,
            "duration_s": round((seg.end_ms - seg.start_ms) / 1000, 1),
        }
        for i, seg in enumerate(segments)
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=snippets,
        metadatas=metadatas
    )

    print(f"Saved {len(segments)} segments for {video_id} ({episode_title})")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search(
    query: str,
    top_k: int = 10,
    video_id: Optional[str] = None
) -> List[dict]:
    """
    Search segments by semantic similarity to query.

    Returns list of dicts with snippet, score, video_id, timestamps, etc.
    """
    collection = _get_collection()
    query_embedding = _embed_query(query)

    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"]
    }

    if video_id:
        query_params["where"] = {"video_id": video_id}

    results = collection.query(**query_params)

    segments = []
    for i in range(len(results['ids'][0])):
        # ChromaDB default is L2 distance. For unit-normalised vectors:
        # cosine_similarity = 1 - (L2_distance² / 2)
        distance = results['distances'][0][i]
        similarity = 1 - (distance ** 2 / 2)

        segments.append({
            "id": results['ids'][0][i],
            "snippet": results['documents'][0][i],
            "score": round(similarity, 3),
            **results['metadatas'][0][i]
        })

    return segments


# ---------------------------------------------------------------------------
# Library / management
# ---------------------------------------------------------------------------

def delete_episode(video_id: str) -> int:
    """Delete all segments for an episode. Returns count deleted."""
    collection = _get_collection()
    existing = collection.get(where={"video_id": video_id})
    if existing['ids']:
        collection.delete(ids=existing['ids'])
        return len(existing['ids'])
    return 0


def get_episode_segments(video_id: str) -> List[dict]:
    """Get all segments for an episode, sorted by time."""
    collection = _get_collection()
    results = collection.get(
        where={"video_id": video_id},
        include=["metadatas", "documents"],
    )

    segments = []
    for i, meta in enumerate(results["metadatas"]):
        segments.append({
            "segment_id": results["ids"][i],
            "snippet": results["documents"][i] if results["documents"] else "",
            **meta,
        })

    segments.sort(key=lambda s: s.get("start_ms", 0))
    return segments


def get_all_metadata() -> List[dict]:
    """Get metadata for all segments (for library overview)."""
    collection = _get_collection()
    results = collection.get(include=["metadatas"])
    return results["metadatas"]


def get_total_count() -> int:
    """Get total number of segments in the collection."""
    return _get_collection().count()
