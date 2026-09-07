"""Local Voice Activity Detection (spec §3.7, §4.5). Silero VAD, run on-device.
Used by the Voice Interrupter overlay to detect speech onset without constant
STT polling. No network / API keys.
"""

from __future__ import annotations

import asyncio
from typing import Optional


async def detect_speech(audio_path: str) -> list[tuple[float, float]]:
    """Return list of (start_sec, end_sec) speech segments via Silero VAD."""

    def _run() -> list[tuple[float, float]]:
        import torch
        import torchaudio

        model, _ = torch.hub.load(
            "snakers4/silero-vad",
            "silero_vad",
            trust_repo=True,
        )
        wav, sr = torchaudio.load(audio_path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
        speech_timestamps = model.get_speech_timestamps(wav[0], threshold=0.5)
        return [(float(t["start"]), float(t["end"])) for t in speech_timestamps]

    return await asyncio.to_thread(_run)


def available() -> bool:
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False
