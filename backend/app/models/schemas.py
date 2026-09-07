"""API request/response schemas (Pydantic v2).

These wrap the internal domain models (``models.domain``) for the HTTP API
surface. Sibling routers (ingest, retrieval, study, audio, export) depend on
these schemas.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from app.models.domain import (
    AskDocsModel,
    AskDocsWorkspace,
    CONFIDENCE_LEVEL,
    ChunkID,
    DocID,
    DocumentState,
    FORMAT_TYPE,
    FlashcardState,
    HARDWARE_TIER,
    MODE,
    RATING,
    RetrievedChunk,
    WorkspaceID,
)


# --------------------------------------------------------------------------- #
# Ingest (parsing agent)
# --------------------------------------------------------------------------- #
class IngestRequest(AskDocsModel):
    filename: str
    workspace_id: Optional[WorkspaceID] = None
    chunk_size: int = 1000
    chunk_overlap: int = 150
    ocr: bool = False


class IngestResponse(AskDocsModel):
    doc_id: DocID
    status: str
    num_chunks: int = 0
    message: str = ""


# --------------------------------------------------------------------------- #
# Retrieval (rag agent)
# --------------------------------------------------------------------------- #
class RetrieveRequest(AskDocsModel):
    query: str
    top_k: int = 5
    doc_ids: Optional[list[DocID]] = None
    mode: MODE = MODE.CHAT
    workspace_id: Optional[WorkspaceID] = None


class RetrieveResponse(AskDocsModel):
    query: str
    chunks: list[RetrievedChunk]
    latency_ms: float = 0.0


class AskRequest(AskDocsModel):
    query: str
    top_k: int = 5
    mode: MODE = MODE.CHAT
    doc_ids: Optional[list[DocID]] = None
    workspace_id: Optional[WorkspaceID] = None


class AskResponse(AskDocsModel):
    query: str
    answer: str
    chunks: list[RetrievedChunk]
    confidence: CONFIDENCE_LEVEL = CONFIDENCE_LEVEL.UNKNOWN
    model_used: Optional[str] = None
    latency_ms: float = 0.0


class RateRequest(AskDocsModel):
    chunk_id: ChunkID
    rating: RATING


# --------------------------------------------------------------------------- #
# Study / LLM (llm/study agent)
# --------------------------------------------------------------------------- #
class QuizRequest(AskDocsModel):
    doc_id: Optional[DocID] = None
    num_questions: int = 5
    mode: MODE = MODE.QUIZ
    workspace_id: Optional[WorkspaceID] = None


class QuizQuestion(AskDocsModel):
    question: str
    options: list[str] = Field(default_factory=list)
    answer: str = ""
    explanation: str = ""


class QuizResponse(AskDocsModel):
    questions: list[QuizQuestion]
    model_used: Optional[str] = None


class SummaryRequest(AskDocsModel):
    doc_id: DocID
    max_sentences: int = 8
    workspace_id: Optional[WorkspaceID] = None


class SummaryResponse(AskDocsModel):
    summary: str
    model_used: Optional[str] = None


class FlashcardRequest(AskDocsModel):
    doc_id: Optional[DocID] = None
    num_cards: int = 10
    workspace_id: Optional[WorkspaceID] = None


class FlashcardResponse(AskDocsModel):
    cards: list[FlashcardState]


class RateFlashcardRequest(AskDocsModel):
    flashcard_id: str
    rating: RATING


# --------------------------------------------------------------------------- #
# Audio (audio agent)
# --------------------------------------------------------------------------- #
class TTSRequest(AskDocsModel):
    text: str
    voice: str = "default"
    format: str = "wav"


class TTSResponse(AskDocsModel):
    audio_path: str
    duration_sec: float = 0.0


class TranscribeRequest(AskDocsModel):
    audio_path: str
    language: Optional[str] = None


class TranscribeResponse(AskDocsModel):
    text: str
    language: Optional[str] = None
    segments: list[dict] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Export (export agent)
# --------------------------------------------------------------------------- #
class ExportRequest(AskDocsModel):
    workspace_id: Optional[WorkspaceID] = None
    format: FORMAT_TYPE = FORMAT_TYPE.ANKI
    deck_name: str = "AskDocs Export"


class ExportResponse(AskDocsModel):
    file_path: str
    format: FORMAT_TYPE
    size_bytes: int = 0


# --------------------------------------------------------------------------- #
# Workspace
# --------------------------------------------------------------------------- #
class WorkspaceResponse(AskDocsModel):
    workspace: AskDocsWorkspace


class WorkspaceListResponse(AskDocsModel):
    documents: list[DocumentState]
    workspace_id: WorkspaceID


# --------------------------------------------------------------------------- #
# Benchmark / hardware (model_registry integration)
# --------------------------------------------------------------------------- #
class BenchmarkRequest(AskDocsModel):
    force: bool = False


class BenchmarkResponse(AskDocsModel):
    tier: HARDWARE_TIER
    cpu_cores: int
    total_ram_gb: float
    available_ram_gb: float
    gpu_available: bool
    webgpu_proxy: bool
    disk_free_gb: float
    recommended_models: dict = Field(default_factory=dict)
    benchmarked_at: datetime = Field(default_factory=datetime.utcnow)


class ModelStatusResponse(AskDocsModel):
    tier: HARDWARE_TIER
    backend: str
    embedding_model: str
    llm_model: str
    tts_model: Optional[str] = None
    stt_model: Optional[str] = None
    models_available: dict = Field(default_factory=dict)


class HealthResponse(AskDocsModel):
    status: str = "ok"
    ollama_reachable: bool = False
    redis_reachable: bool = False
    version: str = "2.0"
