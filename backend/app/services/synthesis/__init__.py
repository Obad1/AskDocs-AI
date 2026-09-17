# AskDocs AI — local LLM synthesis services (spec §4.4 / §4.8 / §6.3).
# All generation is performed against a local Ollama (or llama.cpp) server.
# Zero API keys, zero external network calls.

from app.services.synthesis import (  # noqa: F401  (re-export submodules)
    summary_engine,
    quiz_generator,
    flashcard_sm2,
    matrix_builder,
)

__all__ = ["summary_engine", "quiz_generator", "flashcard_sm2", "matrix_builder"]
