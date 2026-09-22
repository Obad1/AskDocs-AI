"""LLM service package (provider failover router)."""

from .llm_router import configured_providers, generate

__all__ = ["generate", "configured_providers"]