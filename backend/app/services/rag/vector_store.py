"""ChromaDB embedded-mode vector store wrapper.

No managed/cloud Chroma — uses a local persistent client. Stores chunk
embeddings and returns nearest neighbors (L2 distance) for a query vector.

Contract: chunks are dicts with keys id, docId, text, vector (list[float]),
page (optional). Query returns [{id, score, docId, text, page}, ...].
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import chromadb
from chromadb.config import Settings


class VectorStore:
    def __init__(self, persist_dir: str = "./.chroma") -> None:
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=persist_dir, settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name="askdocs", metadata={"hnsw:space": "l2"}
        )

    def add(self, chunks: List[Dict[str, Any]]) -> None:
        if not chunks:
            return
        ids = [str(c["id"]) for c in chunks]
        embeddings = [list(c["vector"]) for c in chunks]
        metadatas = [
            {
                "docId": str(c.get("docId", "")),
                "text": str(c.get("text", "")),
                "page": int(c["page"]) if c.get("page") is not None else -1,
            }
            for c in chunks
        ]
        self.collection.upsert(ids=ids, embeddings=embeddings, metadatas=metadatas)

    def query(self, vector: List[float], k: int = 10) -> List[Dict[str, Any]]:
        if self.collection.count() == 0:
            return []
        res = self.collection.query(query_embeddings=[list(vector)], n_results=k)
        out: List[Dict[str, Any]] = []
        ids = res["ids"][0]
        distances = res["distances"][0]
        metadatas = res["metadatas"][0]
        for cid, dist, meta in zip(ids, distances, metadatas):
            out.append(
                {
                    "id": cid,
                    "score": float(dist),
                    "docId": meta.get("docId"),
                    "text": meta.get("text"),
                    "page": meta.get("page") if meta.get("page", -1) != -1 else None,
                }
            )
        # Sort descending by similarity (smaller L2 distance = better).
        out.sort(key=lambda x: x["score"])
        return out
