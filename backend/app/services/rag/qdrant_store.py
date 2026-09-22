"""Qdrant Cloud vector store (REST, zero deps) with local Chroma fallback.

Implements the *same contract* as :class:`app.services.rag.vector_store.VectorStore`
(add / query / get_texts) so the retrieval router can swap the ephemeral local
ChromaDB on Render free tier for a persistent Qdrant Cloud cluster without
changing call sites (blueprint §4 "External Persistent Vector Store").

Enabled when ``QDRANT_URL`` and ``QDRANT_API_KEY`` are set; otherwise
:func:`build_vector_store` returns the existing local store. The Qdrant REST
API is called directly via ``urllib`` because the container runs on
0.1 CPU / 512MB RAM and every dependency counts (no ``qdrant-client``).
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Any, Dict, List, Optional
from urllib import error, request

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class QdrantError(RuntimeError):
    """Raised when the Qdrant Cloud API rejects or cannot be reached."""


class QdrantStore:
    """Persistent vector store backed by a Qdrant Cloud cluster (REST)."""

    def __init__(
        self,
        url: str,
        api_key: str,
        collection: str = "askdocs",
        timeout: int = 30,
    ) -> None:
        self.base = url.rstrip("/")
        self.api_key = api_key
        self.collection = collection
        self.timeout = timeout

    # ------------------------------------------------------------------ HTTP
    def _headers(self) -> Dict[str, str]:
        return {
            "api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _req(
        self, method: str, path: str, payload: Optional[dict] = None
    ) -> Optional[Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        req = request.Request(
            f"{self.base}{path}",
            data=body,
            headers=self._headers(),
            method=method,
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                return json.loads(raw) if raw else None
        except error.HTTPError as exc:  # noqa: BLE001
            detail = ""
            try:
                detail = exc.read().decode("utf-8")[:300]
            except Exception:  # noqa: BLE001
                pass
            raise QdrantError(f"Qdrant {method} {path} -> HTTP {exc.code}: {detail}")
        except error.URLError as exc:  # noqa: BLE001
            raise QdrantError(f"Qdrant unreachable at {self.base}: {exc.reason}")

    # --------------------------------------------------------------- identity
    @staticmethod
    def _point_id(chunk_id: str) -> str:
        """Chunk ids look like ``<uuid>::chunk-0``; Qdrant needs UUIDs."""
        return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))

    # ------------------------------------------------------------- collection
    def _ensure_collection(self, dim: int) -> None:
        try:
            self._req("GET", f"/collections/{self.collection}")
        except QdrantError:
            # 404 -> create with Cosine (best-first; higher score = more similar).
            self._req(
                "PUT",
                f"/collections/{self.collection}",
                {"vectors": {"size": dim, "distance": "Cosine"}},
            )

    # ---------------------------------------------------------- store contract
    def add(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            return
        dim = int(len(list(chunks[0]["vector"])))
        self._ensure_collection(dim)
        points: List[Dict[str, Any]] = []
        for c in chunks:
            cid = str(c["id"])
            payload: Dict[str, Any] = {
                "chunk_id": cid,
                "docId": str(c.get("docId", "")),
                "text": str(c.get("text", "")),
            }
            if c.get("page") is not None:
                payload["page"] = int(c["page"])
            points.append(
                {
                    "id": self._point_id(cid),
                    "vector": [float(x) for x in c["vector"]],
                    "payload": payload,
                }
            )
        self._req(
            "PUT",
            f"/collections/{self.collection}/points?wait=true",
            {"points": points},
        )

    def query(self, vector: List[float], k: int = 10) -> List[Dict[str, Any]]:
        try:
            res = self._req(
                "POST",
                f"/collections/{self.collection}/points/search",
                {"vector": [float(x) for x in vector], "limit": k, "with_payload": True},
            )
        except QdrantError as exc:
            if "doesn't exist" in str(exc) or "not found" in str(exc).lower():
                return []
            raise
        out: List[Dict[str, Any]] = []
        for hit in (res or {}).get("result", []) or []:
            payload = hit.get("payload") or {}
            page = payload.get("page", -1)
            out.append(
                {
                    "id": payload.get("chunk_id", ""),
                    "score": float(hit.get("score", 0.0)),
                    "docId": payload.get("docId"),
                    "text": payload.get("text"),
                    "page": page if page != -1 else None,
                }
            )
        # Qdrant Cosine score: higher = closer. Best-first for RRF.
        out.sort(key=lambda x: x["score"], reverse=True)
        return out

    def get_texts(self, ids: List[str]) -> Dict[str, str]:
        if not ids:
            return {}
        try:
            res = self._req(
                "POST",
                f"/collections/{self.collection}/points",
                {
                    "ids": [self._point_id(i) for i in ids],
                    "with_payload": True,
                    "with_vector": False,
                },
            )
        except QdrantError:
            return {}
        out: Dict[str, str] = {}
        for item in (res or {}).get("result", []) or []:
            payload = item.get("payload") or {}
            cid = payload.get("chunk_id")
            if cid:
                out[cid] = payload.get("text", "")
        return out


def build_vector_store():
    """Return the configured vector store (Qdrant Cloud when configured).

    Falls back to the local ChromaDB :class:`VectorStore`, which is imported
    lazily so the Qdrant path works even when Chroma deps are absent.
    """
    settings = get_settings()
    if settings.qdrant_url and settings.qdrant_api_key:
        store = QdrantStore(
            settings.qdrant_url,
            settings.qdrant_api_key,
            collection=settings.qdrant_collection,
        )
        logger.info("Vector store: Qdrant Cloud (%s)", store.base)
        return store
    from app.services.rag.vector_store import VectorStore

    logger.info("Vector store: local ChromaDB (%s)", settings.vector_store_path)
    return VectorStore(persist_dir=settings.vector_store_path)