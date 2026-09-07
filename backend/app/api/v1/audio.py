"""Audio API router (spec §4.5). All endpoints use local models only.

Endpoints:
  POST /tts   -> { text, voice } -> audio/wav
  POST /stt   -> upload audio    -> { text }
  POST /vad   -> upload audio    -> { segments: [[start, end], ...] }
"""

from __future__ import annotations

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import Response

from app.services.audio import tts_synthesizer, stt_listener, vad_detector

router = APIRouter(prefix="/audio", tags=["audio"])


@router.post("/tts")
async def tts(text: str, voice: str = "en_US-lessac-medium"):
    audio = await tts_synthesizer.synthesize(text, voice)
    return Response(content=audio, media_type="audio/wav")


@router.post("/stt")
async def stt(file: UploadFile = File(...), model: str = "base.en"):
    data = await file.read()
    text = await stt_listener.transcribe_bytes(data, model, suffix=".wav")
    return {"text": text}


@router.post("/vad")
async def vad(file: UploadFile = File(...)):
    data = await file.read()
    path = f"/tmp/askdocs_vad_{abs(hash(data))}.wav"
    with open(path, "wb") as f:
        f.write(data)
    try:
        segments = await vad_detector.detect_speech(path)
    finally:
        import os

        os.remove(path)
    return {"segments": segments}
