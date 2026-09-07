"""Ingestion pipeline orchestration.

Coordinates the sibling parsing and RAG services into a single flow:

    ingest -> parse -> chunk -> embed -> store

The parsing/rag *services* are owned by sibling agents and imported lazily
(so this module imports cleanly even if they are not yet present at test
time). The pipeline degrades gracefully: if a service module is missing it
raises a clear ``RuntimeError`` at call time rather than at import time.

Background execution uses Huey (Redis or SQLite broker per settings).
"""
from __future__ import annotations

import importlib
from typing import Any, Callable, Optional

from app.core.config import get_settings
from app.core.security import ensure_within_root, sanitize_filename
from app.models.domain import DocID, DocumentState
from app.models.schemas import IngestRequest, IngestResponse

# --------------------------------------------------------------------------- #
# Huey broker setup (no network unless Redis configured)
# --------------------------------------------------------------------------- #
settings = get_settings()

if settings.job_broker == "sqlite":
    from huey import SqliteHuey

    huey = SqliteHuey(filename=str(importlib.resources.files("app") / "huey.db"))
else:
    from huey import RedisHuey

    huey = RedisHuey(name="askdocs", url=settings.redis_url)


# --------------------------------------------------------------------------- #
# Lazy service resolution
# --------------------------------------------------------------------------- #
def _import(attr_path: str) -> Optional[Callable[..., Any]]:
    """Import ``module.submodule:callable`` and return it, or ``None``."""
    try:
        module_name, _, attr = attr_path.rpartition(":")
        module = importlib.import_module(module_name)
        return getattr(module, attr)
    except Exception:
        return None


# Plausible entry points owned by sibling agents. The pipeline tries each
# candidate until one resolves, so it is resilient to their exact layout.
_PARSE_CANDIDATES = [
    "app.services.parsing:parse_document",
    "app.services.parsing.pdf:parse_document",
    "app.services.parsing:extract_text",
]
_CHUNK_CANDIDATES = [
    "app.services.rag.chunk:chunk_text",
    "app.services.rag:chunk_text",
]
_EMBED_CANDIDATES = [
    "app.services.rag.embed:embed_chunks",
    "app.services.rag:embed_chunks",
]
_STORE_CANDIDATES = [
    "app.services.rag.store:store_chunks",
    "app.services.rag:store_chunks",
]


def _resolve(candidates: list[str]) -> Callable[..., Any]:
    for c in candidates:
        fn = _import(c)
        if fn is not None:
            return fn
    raise RuntimeError(
        f"No ingestion service resolved from candidates: {candidates}. "
        "Ensure the parsing/rag sibling services are mounted."
    )


# --------------------------------------------------------------------------- #
# Synchronous orchestration
# --------------------------------------------------------------------------- #
def build_document_state(req: IngestRequest, doc_id: DocID) -> DocumentState:
    return DocumentState(
        doc_id=doc_id,
        filename=sanitize_filename(req.filename),
        status="pending",
    )


def run_ingest(req: IngestRequest, source_path: str) -> IngestResponse:
    """Full ingest pipeline for one file at ``source_path``.

    ``source_path`` must already be inside the data root (validated).
    """
    resolved_path = ensure_within_root(source_path)
    doc_id = DocID.make()
    state = build_document_state(req, doc_id)
    state.status = "parsing"

    parse = _resolve(_PARSE_CANDIDATES)
    chunk = _resolve(_CHUNK_CANDIDATES)
    embed = _resolve(_EMBED_CANDIDATES)
    store = _resolve(_STORE_CANDIDATES)

    try:
        text = parse(str(resolved_path), ocr=req.ocr)
        state.status = "chunking"
        chunks = chunk(text, chunk_size=req.chunk_size, overlap=req.chunk_overlap)
        state.status = "embedding"
        vectors = embed(chunks)
        state.status = "storing"
        store(doc_id, chunks, vectors, workspace_id=req.workspace_id)
        state.num_chunks = len(chunks)
        state.status = "ready"
        return IngestResponse(
            doc_id=doc_id,
            status="ready",
            num_chunks=state.num_chunks,
            message="Ingested successfully.",
        )
    except Exception as exc:  # noqa: BLE001 - surface as failed state
        state.status = "failed"
        state.error = str(exc)
        return IngestResponse(
            doc_id=doc_id, status="failed", message=f"Ingest failed: {exc}"
        )


# --------------------------------------------------------------------------- #
# Background task (Huey)
# --------------------------------------------------------------------------- #
@huey.task(retries=2, retry_delay=30)
def ingest_async(req: "IngestRequest", source_path: str) -> dict:
    """Queue an ingest job to run in the background worker."""
    result = run_ingest(req, source_path)
    return result.model_dump()
