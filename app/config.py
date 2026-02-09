"""Configuration settings for Echo."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    app_name: str = "Echo Audio Browser"
    debug: bool = False

    # Vector DB
    chroma_persist_dir: str = "./chroma_data"

    # LLM + Embeddings (OpenAI)
    openai_api_key: Optional[str] = None
    segmentation_model: str = "gpt-4o-mini"

    class Config:
        env_file = ".env"


settings = Settings()
