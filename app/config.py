"""Configuration settings for Echo."""

from pydantic_settings import BaseSettings
from typing import Optional
import chromadb.utils.embedding_functions as embedding_functions


class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    # App
    app_name: str = "Echo Audio Browser"
    debug: bool = False
    
    # Database
    database_url: str = "sqlite:///./echo.db"
    
    # Vector DB
    chroma_persist_dir: str = "./chroma_data"
    
    # Transcription
    deepgram_api_key: Optional[str] = None
    whisper_model: str = "base"  # For local Whisper fallback
    
    # LLM
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    segmentation_model: str = "gpt-4o-mini"  # or "claude-3-haiku"
    
    # Embeddings
    embedding_model: str = "text-embedding-3-small"  # OpenAI embedding model
    use_openai_embeddings: bool = True  # Use OpenAI embeddings vs ChromaDB default
    
    # Segmentation settings
    min_segment_duration_sec: int = 120  # 2 minutes
    max_segment_duration_sec: int = 600  # 10 minutes
    target_segments_per_episode: int = 10
    
    # Search settings
    default_search_limit: int = 20
    min_density_score: float = 0.3
    
    class Config:
        env_file = ".env"


settings = Settings()


def get_embedding_function():
    """
    Get the ChromaDB embedding function based on config.
    Returns OpenAI embeddings if configured, otherwise ChromaDB default.
    """
    if settings.use_openai_embeddings and settings.openai_api_key:
        return embedding_functions.OpenAIEmbeddingFunction(
            api_key=settings.openai_api_key,
            model_name=settings.embedding_model
        )
    else:
        # Use ChromaDB default (all-MiniLM-L6-v2)
        return embedding_functions.DefaultEmbeddingFunction()
