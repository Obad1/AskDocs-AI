"""Pydantic domain models mirroring the Z schemas in §7.

These are the authoritative internal types used across the backend.
API-facing schemas live in ``models.schemas`` and wrap these where needed.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class AskDocsModel(BaseModel):
    """Base model that tolerates the str-subclass identifier types (DocID, etc.)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)


# --------------------------------------------------------------------------- #
# Identifier newtypes (Z §7.1)
# --------------------------------------------------------------------------- #
class DocID(str):
    """Opaque document identifier (UUID4)."""

    @classmethod
    def make(cls) -> "DocID":
        return cls(str(uuid.uuid4()))


class ChunkID(str):
    """Opaque chunk identifier (UUID4)."""

    @classmethod
    def make(cls) -> "ChunkID":
        return cls(str(uuid.uuid4()))


class FlashcardID(str):
    """Opaque flashcard identifier (UUID4)."""

    @classmethod
    def make(cls) -> "FlashcardID":
        return cls(str(uuid.uuid4()))


class WorkspaceID(str):
    """Opaque workspace identifier (UUID4)."""

    @classmethod
    def make(cls) -> "WorkspaceID":
        return cls(str(uuid.uuid4()))


# --------------------------------------------------------------------------- #
# Enumerations (Z §7.3)
# --------------------------------------------------------------------------- #
class RATING(str, Enum):
    """User feedback rating for a retrieval / answer."""

    HELPFUL = "helpful"
    NOT_HELPFUL = "not_helpful"
    PARTIAL = "partial"


class CONFIDENCE_LEVEL(str, Enum):
    """Model-reported confidence for an answer / chunk."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class MODE(str, Enum):
    """Interaction mode selected by the user (Z §7)."""

    STUDY = "study"
    QUIZ = "quiz"
    CHAT = "chat"
    EXPLORE = "explore"


class FORMAT_TYPE(str, Enum):
    """Export / output format."""

    ANKI = "anki"
    PDF = "pdf"
    MARKDOWN = "markdown"
    CSV = "csv"


class HARDWARE_TIER(str, Enum):
    """Hardware capability tier (Z §5 / §7.4)."""

    TIER1_LOW = "tier1_low"
    TIER2_MID = "tier2_mid"
    TIER3_HIGH = "tier3_high"


class ENGINE_BACKEND(str, Enum):
    """Local inference backend (Z §7.4)."""

    OLLAMA = "ollama"
    LLAMA_CPP = "llama_cpp"
    WEBGPU_WEBLLM = "webgpu_webllm"
    NATIVE = "native"


# --------------------------------------------------------------------------- #
# Document & chunk state (Z §7.2)
# --------------------------------------------------------------------------- #
class DocumentState(AskDocsModel):
    """Z DocumentState: lifecycle of an ingested document."""

    doc_id: DocID
    filename: str
    title: Optional[str] = None
    mime_type: Optional[str] = None
    size_bytes: int = 0
    num_pages: Optional[int] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    status: str = "pending"  # pending|parsing|chunking|embedding|ready|failed
    error: Optional[str] = None
    num_chunks: int = 0
    checksum_sha256: Optional[str] = None
    workspace_id: Optional[WorkspaceID] = None


class RetrievedChunk(AskDocsModel):
    """Z RetrievedChunk: a chunk surfaced by retrieval."""

    chunk_id: ChunkID
    doc_id: DocID
    text: str
    score: float
    rank: int
    confidence: CONFIDENCE_LEVEL = CONFIDENCE_LEVEL.UNKNOWN
    metadata: dict = Field(default_factory=dict)


class QueryResult(AskDocsModel):
    """Z QueryResult: answer + supporting evidence."""

    query: str
    answer: Optional[str] = None
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    mode: MODE = MODE.CHAT
    confidence: CONFIDENCE_LEVEL = CONFIDENCE_LEVEL.UNKNOWN
    model_used: Optional[str] = None
    latency_ms: float = 0.0
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# --------------------------------------------------------------------------- #
# Flashcard / study state (Z §7.2)
# --------------------------------------------------------------------------- #
class FlashcardState(AskDocsModel):
    """Z FlashcardState: a generated study card."""

    flashcard_id: FlashcardID
    doc_id: Optional[DocID] = None
    front: str
    back: str
    rating: Optional[RATING] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    tags: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Workspace state (Z §7.2)
# --------------------------------------------------------------------------- #
class AskDocsWorkspace(AskDocsModel):
    """Z AskDocsWorkspace: the full persisted application state."""

    workspace_id: WorkspaceID = Field(default_factory=WorkspaceID.make)
    name: str = "default"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    documents: list[DocumentState] = Field(default_factory=list)
    flashcards: list[FlashcardState] = Field(default_factory=list)
    engine: "ModelEngineState" = Field(default_factory=lambda: ModelEngineState())
    settings: dict = Field(default_factory=dict)
    version: str = "2.0"

    model_config = {"arbitrary_types_allowed": True}


# --------------------------------------------------------------------------- #
# Model engine state (Z §7.4)
# --------------------------------------------------------------------------- #
class ModelEngineState(AskDocsModel):
    """Z ModelEngineState: selected models + backend + tier."""

    tier: HARDWARE_TIER = HARDWARE_TIER.TIER2_MID
    backend: ENGINE_BACKEND = ENGINE_BACKEND.OLLAMA
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "llama3.1:8b"
    tts_model: Optional[str] = None
    stt_model: Optional[str] = None
    webgpu_enabled: bool = False
    ollama_host: Optional[str] = None
    last_benchmark_at: Optional[datetime] = None

    model_config = {"arbitrary_types_allowed": True}


# Rebuild forward-refs
AskDocsWorkspace.model_rebuild()
