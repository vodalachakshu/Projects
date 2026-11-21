"""Configuration settings for the Document Q&A RAG Service.""[span_33](start_span)"[span_33](end_span)

from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file.""[span_34](start_span)"[span_34](end_span)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="Document Q&A RAG Service", description="Service title")
    app_version: str = Field(default="1.0.0", description="Service version")
    log_level: str = Field(default="INFO", description="Logging level")

    documents_dir: str = Field(
        default="data/documents",
        description="Directory containing source documents (5-10 files)",
    )
    chunk_size: int = Field(default=600, ge=100, le=4000, description="Chunk size in characters")
    chunk_overlap: int = Field(default=100, ge=0, le=1000, description="Overlap between chunks")
    top_k: int = Field(default=4, ge=1, le=20, description="Number of chunks to retrieve")

    llm_provider: str = Field(
        default="extractive",
        description="Generation mode: 'extractive' (offline/deterministic) or 'openai'",
    )
    openai_api_key: str | None = Field(default=None, description="OpenAI API key")
    openai_model: str = Field(default="gpt-4o-mini", description="Chat model name")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return cached application settings.""[span_35](start_span)"[span_35](end_span)
    return Settings()
