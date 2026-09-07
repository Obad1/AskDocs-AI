"""Local-LLM summarization (spec §3.4 / §6.3).

Wraps a local Ollama server (``ollama`` python client) and produces summaries at
three granularity levels with selectable role profiles. No API keys required.

If Ollama is unavailable, the engine degrades gracefully to an extractive
fallback so callers always receive a usable summary.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

try:  # ollama is an optional dependency at import time
    import ollama
except Exception:  # pragma: no cover - import guard
    ollama = None

from app.core.config import get_settings

_SETTINGS = get_settings()


class Granularity(int, Enum):
    LEVEL1 = 1  # one-line
    LEVEL2 = 2  # three-paragraph
    LEVEL3 = 3  # section outline


class RoleProfile(str, Enum):
    GENERAL = "General"
    STUDENT = "Student"
    RESEARCHER = "Researcher"
    EXECUTIVE = "Executive"
    LEGAL = "Legal"


_GRANULARITY_INSTRUCTION = {
    Granularity.LEVEL1: "Produce a SINGLE one-line summary (Level 1).",
    Granularity.LEVEL2: "Produce a 3-paragraph summary (Level 2).",
    Granularity.LEVEL3: "Produce a structured section outline with headings and bullets (Level 3).",
}

_PROFILE_INSTRUCTION = {
    RoleProfile.GENERAL: "Write for a general audience.",
    RoleProfile.STUDENT: "Write for a student studying the material; emphasize key concepts.",
    RoleProfile.RESEARCHER: "Write for a researcher; preserve nuance and methodology.",
    RoleProfile.EXECUTIVE: "Write for an executive; lead with conclusions and actionable takeaways.",
    RoleProfile.LEGAL: "Write for a legal reviewer; be precise about obligations and caveats.",
}


def _client():
    if ollama is None:
        raise RuntimeError("ollama python client is not installed")
    if _SETTINGS.ollama_host:
        ollama.host = _SETTINGS.ollama_host
    return ollama


def _extractive_fallback(text: str, granularity: Granularity) -> str:
    sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    if not sentences:
        return text[:200]
    if granularity == Granularity.LEVEL1:
        return sentences[0][:200]
    if granularity == Granularity.LEVEL2:
        return ". ".join(sentences[:6]) + "."
    return "\n".join(f"- {s}" for s in sentences[:10])


def summarize(
    text: str,
    granularity: int = 2,
    profile: str = "General",
    model: Optional[str] = None,
    temperature: float = 0.3,
) -> str:
    """Generate a local summary. Falls back to extractive summarization on error."""
    g = Granularity(max(1, min(3, int(granularity))))
    p = RoleProfile(profile) if profile in RoleProfile.__members__ else RoleProfile.GENERAL
    system = " ".join(
        [
            "You are AskDocs AI's summarization engine.",
            _GRANULARITY_INSTRUCTION[g],
            _PROFILE_INSTRUCTION[p],
            "Also emit 2-3 'Actionable Extract' callouts as lines prefixed with 'ACTION:'.",
        ]
    )
    try:
        client = _client()
        resp = client.chat(
            model=model or _SETTINGS.default_llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"TEXT TO SUMMARIZE:\n{text}"},
            ],
            options={"temperature": temperature},
        )
        return resp["message"]["content"].strip()
    except Exception as exc:  # degrade gracefully
        return f"[extractive fallback] {_extractive_fallback(text, g)}\n\n({exc})"
