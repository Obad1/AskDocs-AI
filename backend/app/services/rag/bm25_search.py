"""BM25 keyword search wrapper around rank_bm25.

Contract: chunks are dicts with keys id, text. search() returns
[{id, score}, ...] ordered best-first by BM25 score. Pure local, no network.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from rank_bm25 import BM25Okapi


class BM25Search:
    def __init__(self) -> None:
        self.ids: List[str] = []
        self.texts: List[str] = []
        self.bm25: Optional[BM25Okapi] = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", (text or "").lower())

    def add(self, chunks: List[Dict[str, Any]]) -> None:
        self.ids = [str(c["id"]) for c in chunks]
        self.texts = [str(c.get("text", "")) for c in chunks]
        corpus = [self._tokenize(t) for t in self.texts]
        if corpus:
            self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, k: int = 10) -> List[Dict[str, Any]]:
        if self.bm25 is None or not query:
            return []
        scores = self.bm25.get_scores(self._tokenize(query))
        ranked = sorted(
            zip(self.ids, scores), key=lambda x: x[1], reverse=True
        )[:k]
        return [{"id": i, "score": float(s)} for i, s in ranked]
