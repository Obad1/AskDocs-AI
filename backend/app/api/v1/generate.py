"""LLM generation router (blueprint §4 Phase 2 — multi-provider failover).

POST /api/v1/generate {messages: [{role, content}], temperature?, max_tokens?}
-> {text, provider, model}

Backed by :mod:`app.services.llm.llm_router` (Groq -> OpenRouter -> Cerebras).
Returns 502 when no provider key is configured or every provider fails.
"""
from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.llm import configured_providers, generate as llm_generate

router = APIRouter(prefix="/generate", tags=["generate"])


class ChatMessage(BaseModel):
    role: str = "user"
    content: str


class GenerateRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: float = 0.7
    max_tokens: int = 2048
    model: Optional[str] = None


class GenerateResponse(BaseModel):
    text: str
    provider: str
    model: str


@router.get("/health", tags=["generate"])
async def generate_health() -> dict:
    """Which LLM providers are currently keyed up."""
    return {"configured": configured_providers()}


@router.post("", response_model=GenerateResponse)
async def generate(req: GenerateRequest) -> GenerateResponse:
    try:
        result = llm_generate(
            [m.model_dump() for m in req.messages],
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            model=req.model,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc))
    return GenerateResponse(**result)