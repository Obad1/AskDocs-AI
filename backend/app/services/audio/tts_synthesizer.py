"""Local TTS synthesizer (spec §4.5, §8.3). Piper via subprocess CLI; the
public piper-voices models are downloaded once and cached. No API keys.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

PIPER_VOICE_PATHS = {
    "en_US-lessac-low": "en/en_US-lessac-low/en_US-lessac-low.onnx",
    "en_US-lessac-medium": "en/en_US-lessac-medium/en_US-lessac-medium.onnx",
    "en_US-libritts-high": "en/en_US-libritts-high/en_US-libritts-high.onnx",
    "en_US-multi-high": "en/en_US-multi-high/en_US-multi-high.onnx",
}


def _voice_onnx(voice: str) -> str:
    rel = PIPER_VOICE_PATHS.get(voice, PIPER_VOICE_PATHS["en_US-lessac-medium"])
    return str(Path.home() / ".local" / "share" / "piper" / rel)


async def synthesize(text: str, voice: str = "en_US-lessac-medium") -> bytes:
    """Return WAV audio bytes for `text` using the local Piper binary."""
    onnx = _voice_onnx(voice)
    json_path = onnx.replace(".onnx", ".onnx.json")
    proc = await asyncio.create_subprocess_exec(
        "piper",
        "--model", onnx,
        "--config", json_path,
        "--output_raw",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await proc.communicate(text.encode("utf-8"))
    if proc.returncode != 0:
        raise RuntimeError(f"Piper TTS failed: {err.decode('utf-8', 'ignore')}")
    return out
