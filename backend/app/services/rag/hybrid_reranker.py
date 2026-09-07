"""Hybrid fusion + optional cross-encoder reranking (backend).

- reciprocal_rank_fusion: hand-rolled RRF combining vector + BM25 rankings.
- HybridReranker: optional ms-marco cross-encoder (sentence-transformers)
  rerank of top candidates. Falls back to the input order on any failure.

All models are pulled from public HuggingFace repos; no API keys / cloud.
"""

from __future__ import annotations

from typing import Dict, List, Optional

DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def reciprocal_rank_fusion(
    rankings: List[List[Dict[str, float]]], k: int = 60
) -> List[Dict[str, float]]:
    """Combine ranked lists via RRF. Position-based; input scores unused."""
    fused: Dict[str, float] = {}
    for ranking in rankings:
        if not ranking:
            continue
        for rank, item in enumerate(ranking):
            cid = item["id"]
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (k + rank + 1)

    out = [{"id": cid, "score": s} for cid, s in fused.items()]
    out.sort(key=lambda x: x["score"], reverse=True)
    return out


class HybridReranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER) -> None:
        self.model_name = model_name
        self.model = None

    def _load(self) -> None:
        if self.model is None:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(self.model_name)

    def rerank(
        self, query: str, candidates: List[Dict[str, Any]], top_n: int = 10
    ) -> List[Dict[str, Any]]:
        if not candidates:
            return []
        try:
            self._load()
            pairs = [(query, str(c.get("text", ""))) for c in candidates]
            scores = self.model.predict(pairs)  # type: ignore[union-attr]
            for c, s in zip(candidates, scores):
                c["score"] = float(s)
            candidates.sort(key=lambda x: x["score"], reverse=True)
            return candidates[:top_n]
        except Exception:
            # Robust fallback: keep original order, neutral scores.
            for c in candidates:
                c.setdefault("score", 0.0)
            return candidates[:top_n]
