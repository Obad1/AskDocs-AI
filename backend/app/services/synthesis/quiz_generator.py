"""JSON-constrained quiz generation (spec §4.4 / §4.8).

Uses a local Ollama server with ``format="json"`` (grammar-constrained output) to
emit structured quiz questions. No API keys.
"""
from __future__ import annotations

import json
from enum import Enum
from typing import Optional

try:
    import ollama
except Exception:  # pragma: no cover
    ollama = None

from app.core.config import get_settings

_SETTINGS = get_settings()


class QuizType(str, Enum):
    MCQ = "MCQ"
    TRUE_FALSE = "TrueFalse"
    FILL_IN = "FillIn"
    MATCHING = "Matching"


class Difficulty(str, Enum):
    EASY = "Easy"
    MEDIUM = "Medium"
    HARD = "Hard"


_TYPE_INSTRUCTION = {
    QuizType.MCQ: (
        'Multiple-choice. Each item: {"type":"MCQ","prompt":str,"options":[4 str],'
        '"answer":str (one of options),"explanation":str,"topic":str}.'
    ),
    QuizType.TRUE_FALSE: (
        'True/False. Each item: {"type":"TrueFalse","prompt":str,'
        '"answer":"True"|"False","explanation":str,"topic":str}.'
    ),
    QuizType.FILL_IN: (
        'Fill-in-the-blank. Each item: {"type":"FillIn","prompt":str (with "___"),'
        '"answer":str,"explanation":str,"topic":str}.'
    ),
    QuizType.MATCHING: (
        'Matching. Return ONE item of type "Matching" with "pairs":'
        '[{"left":term,"right":def}] and "answer":[left1,right1,...] in matched order.'
    ),
}


def _client():
    if ollama is None:
        raise RuntimeError("ollama python client is not installed")
    if _SETTINGS.ollama_host:
        ollama.host = _SETTINGS.ollama_host
    return ollama


def _clean_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except Exception:
        import re

        m = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
        if m:
            return json.loads(m.group(1))
        raise ValueError("model did not return valid JSON")


def generate_quiz(
    context: str,
    quiz_type: str = "MCQ",
    difficulty: str = "Easy",
    model: Optional[str] = None,
    temperature: float = 0.5,
) -> dict:
    """Return ``{"questions": [...]}`` from a local LLM with JSON grammar."""
    qtype = QuizType(quiz_type) if quiz_type in QuizType.__members__ else QuizType.MCQ
    diff = Difficulty(difficulty) if difficulty in Difficulty.__members__ else Difficulty.EASY
    system = " ".join(
        [
            "You are AskDocs AI's quiz generator.",
            f"Create {diff.value} difficulty questions based ONLY on the provided chunks.",
            'Return JSON: {"questions":[ ... ]}.',
            _TYPE_INSTRUCTION[qtype],
            "Return ONLY JSON.",
        ]
    )
    client = _client()
    resp = client.chat(
        model=model or _SETTINGS.default_llm_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"CONTEXT:\n{context}"},
        ],
        format="json",
        options={"temperature": temperature},
    )
    data = _clean_json(resp["message"]["content"])
    return data
