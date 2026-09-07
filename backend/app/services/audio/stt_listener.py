"""Local STT listener (spec §4.5, §8.3). faster-whisper runs fully on-device.
No OpenAI API call. Falls back gracefully if the model is not yet downloaded.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

_DEFAULT_MODEL = "base.en"


async def transcribe(
    audio_path: str | Path,
    model: str = _DEFAULT_MODEL,
) -> str:
    """Transcribe an audio file to text using faster-whisper."""

    def _run() -> str:
        from faster_whisper import WhisperModel

        m = WhisperModel(model, device="cpu", compute_type="int8")
        segments, _ = m.transcribe(str(audio_path), beam_size=5)
        return " ".join(seg.text for seg in segments)

    return await asyncio.to_thread(_run)


async def transcribe_bytes(
    data: bytes, model: str = _DEFAULT_MODEL, suffix: str = ".wav"
) -> str:
    path = Path(f"/tmp/askdocs_stt_{abs(hash(data))}{suffix}")
    path.write_bytes(data)
    try:
        return await transcribe(path, model)
    finally:
        path.unlink(missing_ok=True)


def available(model: str = _DEFAULT_MODEL) -> bool:
    try:
        from faster_whisper import WhisperModel  # noqa: F401

        return True
    except Exception:
        return False
