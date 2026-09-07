"""Application settings (Pydantic v2 ``pydantic-settings``).

Reads environment variables supplied by docker-compose / local .env.
Zero-login, zero-API-key: every value has a safe local default.
"""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the AskDocs AI backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Network / services
    ollama_host: str = "http://localhost:11434"
    redis_url: str = "redis://localhost:6379/0"

    # Persistence
    vector_store_path: str = "./data/chroma"
    data_root: str = "./data"
    exports_dir: str = "./data/exports"

    # CORS (localhost-only by default, zero external dependency)
    cors_allow_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:80",
        "http://127.0.0.1:80",
    ]

    # Huey broker selection: "redis" or "sqlite"
    job_broker: str = "redis"

    # Optional model overrides
    default_embedding_model: str = "nomic-embed-text"
    default_llm_model: str = "llama3.1:8b"

    # App metadata
    app_name: str = "AskDocs AI"
    app_version: str = "2.0"
    debug: bool = False


@lru_cache
def get_settings() -> Settings:
    """Cached accessor for application settings."""
    return Settings()
