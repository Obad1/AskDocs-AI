"""HuggingFace Serverless Inference clients (embeddings + cross-encoder rerank).

Offloads the heavy torch/onnxruntime stacks off the Render free tier when
``HF_TOKEN`` is configured (blueprint §4 "Remote Inference API"):
- embeddings  -> ``/pipeline/feature-extraction/<model>``
- reranking   -> ``/models/<cross-encoder>`` on "query [SEP] passage" pairs

Every function raises a clear ``RuntimeError`` on failure so callers can fall
back to the local pipeline. Uses stdlib ``urllib`` only.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List
from urllib import error, request

logger = logging.getLogger(__name__)

API_BASE = "https://api-inference.huggingface.co"
SEP = " [SEP] "


def _post(url: str, body: Dict[str, Any], token: str, timeout: int = 180) -> Any:
    req = request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw else None
    except error.HTTPError as exc:  # noqa: BLE001
        detail = ""
        try:
            detail = exc.read().decode("utf-8")[:300]
        except Exception:  # noqa: BLE001
            pass
        raise RuntimeError(f"HF Inference HTTP {exc.code}: {detail}")
    except error.URLError as exc:  # noqa: BLE001
        raise RuntimeError(f"HF Inference unreachable: {exc.reason}")


def hf_embed(
    texts: List[str],
    model: str,
    token: str,
    normalize: bool = True,
) -> List[List[float]]:
    """Embed ``texts`` via the hosted feature-extraction pipeline."""
    url = f"{API_BASE}/pipeline/feature-extraction/{model}"
    data = _post(
        url,
        {
            "inputs": texts,
            "options": {"use_cache": True, "wait_for_model": True},
        },
        token,
    )
    if not isinstance(data, list):
        raise RuntimeError(f"HF embed returned unexpected payload: {data!r}")
    vectors: List[List[float]] = []
    for row in data:
        if not isinstance(row, list):
            # Single-vector reply wrapped oddly; attempt to coerce.
            row = data
        vec = [float(x) for x in row]
        if normalize:
            norm = (sum(v * v for v in vec) ** 0.5) or 1.0
            vec = [v / norm for v in vec]
        vectors.append(vec)
    return vectors


def hf_rerank(
    query: str,
    candidates: List[Dict[str, str]],
    model: str,
    token: str,
    top_n: int = 10,
) -> List[Dict[str, Any]]:
    """Cross-encoder rerank of ``candidates`` (``{id, text}``), best-first."""
    if not candidates:
        return []
    url = f"{API_BASE}/models/{model}"
    inputs = [f"{query}{SEP}{c['text']}" for c in candidates]
    data = _post(
        url,
        {"inputs": inputs, "options": {"wait_for_model": True}},
        token,
    )
    if not isinstance(data, list) or len(data) != len(candidates):
        raise RuntimeError(f"HF rerank returned unexpected payload: {data!r}")
    scored: List[Dict[str, Any]] = []
    for c, res in zip(candidates, data):
        score = 0.0
        if isinstance(res, list) and res:
            score = float(res[0].get("score", 0.0)) if isinstance(res[0], dict) else float(res[0])
        elif isinstance(res, dict):
            score = float(res.get("score", 0.0))
        scored.append({"id": c["id"], "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_n]