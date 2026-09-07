"""Local media transcription via faster-whisper (spec Section 4.1).

faster-whisper runs the CTranslate2 model fully on-device. The heavy import is
deferred to call time so the module stays cheap to import in the integrated app.
"""
from __future__ import annotations

from typing import Optional


def transcribe_audio(
    path: str,
    model_size: str = "base",
    language: Optional[str] = None,
) -> str:
    """Transcribe an audio/video file to plain text."""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_size, device="auto", compute_type="int8")
    segments, _ = model.transcribe(path, language=language)
    return "\n".join(seg.text for seg in segments if seg.text)
