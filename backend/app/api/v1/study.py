"""Study & synthesis REST API (spec §4.4 / §4.8 / §3.6).

Endpoints:
  POST /api/v1/study/summarize  -> local LLM summary
  POST /api/v1/study/quiz      -> JSON-constrained quiz
  POST /api/v1/study/flashcard -> flashcard generation (JSON grammar)
  POST /api/v1/study/flashcard/schedule -> SM-2 scheduling step
  POST /api/v1/study/analytics -> score + weak-topic analytics
  POST /api/v1/study/matrix    -> cross-document comparison matrix

All generation runs against a local Ollama/llama.cpp server. Zero API keys.
"""
from __future__ import annotations

import json
import re
import time
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.synthesis import (
    summary_engine,
    quiz_generator,
    flashcard_sm2,
    matrix_builder,
)

router = APIRouter(prefix="/study", tags=["study"])


class SummarizeRequest(BaseModel):
    text: str
    granularity: int = 2
    profile: str = "General"
    model: Optional[str] = None


class QuizRequest(BaseModel):
    context: str
    type: str = "MCQ"
    difficulty: str = "Easy"
    model: Optional[str] = None


class FlashcardRequest(BaseModel):
    context: str
    model: Optional[str] = None


class SM2Request(BaseModel):
    repetitions: int = 0
    interval: int = 0
    easiness: float = 2.5
    due_date: int = 0
    rating: int
    now: Optional[int] = None


class AnalyticsRequest(BaseModel):
    results: List[Dict[str, object]] = []


class MatrixRequest(BaseModel):
    documents: Dict[str, str]


@router.post("/summarize")
def post_summarize(req: SummarizeRequest):
    try:
        out = summary_engine.summarize(
            req.text, req.granularity, req.profile, req.model
        )
        return {"summary": out}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"summarize failed: {exc}")


@router.post("/quiz")
def post_quiz(req: QuizRequest):
    try:
        return quiz_generator.generate_quiz(
            req.context, req.type, req.difficulty, req.model
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"quiz failed: {exc}")


@router.post("/flashcard")
def post_flashcard(req: FlashcardRequest):
    """Generate flashcards from context via local LLM (JSON grammar)."""
    system = (
        "You are AskDocs AI's flashcard generator. Create study flashcards from the chunks. "
        'Return JSON: {"flashcards":[{"front":str,"back":str}]}. Return ONLY JSON.'
    )
    try:
        import ollama

        if summary_engine._SETTINGS.ollama_host:
            ollama.host = summary_engine._SETTINGS.ollama_host
        resp = ollama.chat(
            model=req.model or summary_engine._SETTINGS.default_llm_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f"CONTEXT:\n{req.context}"},
            ],
            format="json",
        )
        data = json.loads(resp["message"]["content"])
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"flashcard failed: {exc}")
    return data


@router.post("/flashcard/schedule")
def post_flashcard_schedule(req: SM2Request):
    """Apply one SM-2 scheduling step server-side (mirror of Z §7.3)."""
    card = flashcard_sm2.SM2State(
        repetitions=req.repetitions,
        interval=req.interval,
        easiness=req.easiness,
        due_date=req.due_date,
    )
    now = req.now if req.now is not None else int(time.time() * 1000)
    updated = flashcard_sm2.review(card, req.rating, now)
    return {
        "repetitions": updated.repetitions,
        "interval": updated.interval,
        "easiness": updated.easiness,
        "due_date": updated.due_date,
    }


@router.post("/analytics")
def post_analytics(req: AnalyticsRequest):
    running = 0
    cum: List[int] = []
    miss: Dict[str, int] = {}
    for i, r in enumerate(req.results):
        correct = bool(r.get("correct", False))
        if correct:
            running += 1
        if req.results:
            cum.append(round(running / (i + 1) * 100))
        else:
            cum.append(0)
        topic = r.get("topic")
        if not correct and topic:
            miss[str(topic)] = miss.get(str(topic), 0) + 1
    weak = sorted(
        ({"topic": t, "misses": c} for t, c in miss.items()),
        key=lambda x: x["misses"],
        reverse=True,
    )
    return {"cumulative_score": cum, "weak_topics": weak}


@router.post("/matrix")
def post_matrix(req: MatrixRequest):
    return matrix_builder.build_matrix(req.documents)
