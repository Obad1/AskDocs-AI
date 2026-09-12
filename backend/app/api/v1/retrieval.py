"""FastAPI retrieval router.

POST /api/v1/retrieve {query, topK} -> RRF-fused (+ optional reranked) chunks.
Orchestrates the local services: VectorStore (Chroma), BM25Search (rank_bm25),
and HybridReranker (sentence-transformers cross-encoder). The incoming query is
embedded lazily with a local SentenceTransformer model (no API keys / cloud).

Mount elsewhere, e.g.:
    from app.api.v1.retrieval import router
    app.include_router(router, prefix="/api/v1")
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.rag.bm25_search import BM25Search
from app.services.rag.hybrid_reranker import HybridReranker, reciprocal_rank_fusion
from app.services.rag.vector_store import VectorStore

router = APIRouter(prefix="/retrieval", tags=["retrieval"])

_vs = VectorStore()
_bm25 = BM25Search()
_reranker = HybridReranker()
_embedder = None

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class RetrieveRequest(BaseModel):
    query: str
    topK: int = 10


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def _embed(text: str) -> List[float]:
    vec = _get_embedder().encode(text, normalize_embeddings=True)
    return [float(x) for x in vec]


@router.post("")
async def retrieve(req: RetrieveRequest) -> Dict[str, Any]:
    try:
        qvec = _embed(req.query)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=(
                "Retrieval not provisioned on this instance (local embedding "
                f"model unavailable): {exc}"
            ),
        )
    try:
        vec_res = _vs.query(qvec, k=req.topK * 2)
        bm25_res = _bm25.search(req.query, k=req.topK * 2)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=f"Retrieval index unavailable on this instance: {exc}",
        )

    rankings = [vec_res, bm25_res] if bm25_res else [vec_res]
    fused = reciprocal_rank_fusion(rankings, 60)

    # Fetch stored texts for the rerank pass.
    chunk_texts: Dict[str, str] = {}
    try:
        rows = _vs.collection.get(
            ids=[f["id"] for f in fused], include=["metadatas"]
        )
        for cid, meta in zip(rows["ids"], rows["metadatas"]):
            chunk_texts[cid] = meta.get("text", "")
    except Exception:
        pass

    candidates = [
        {"id": f["id"], "text": chunk_texts.get(f["id"], "")} for f in fused
    ]
    try:
        reranked = _reranker.rerank(req.query, candidates, top_n=req.topK)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=f"Reranker unavailable on this instance: {exc}",
        )

    chunks = [
        {
            "chunkId": r["id"],
            "text": chunk_texts.get(r["id"], ""),
            "score": r.get("score", 0.0),
        }
        for r in reranked
    ]
    return {"query": req.query, "chunks": chunks}
