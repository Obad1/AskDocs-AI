# AskDocs AI — local LLM synthesis services (spec §4.4 / §4.8 / §6.3).
# All generation is performed against a local Ollama (or llama.cpp) server.
# Zero API keys, zero external network calls.

from app.services.synthesis.summary_engine import summary_engine
from app.services.synthesis.quiz_generator import quiz_generator
from app.services.synthesis.flashcard_sm2 import flashcard_sm2
from app.services.synthesis.matrix_builder import matrix_builder

__all__ = ["summary_engine", "quiz_generator", "flashcard_sm2", "matrix_builder"]
