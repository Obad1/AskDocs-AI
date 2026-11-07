"""Deterministic, offline AskDocs retrieval/citation evaluation gates.

These checks verify reference identity and exact excerpts, not semantic
faithfulness of an LLM answer. Use labeled, consented test documents only.
No network, embeddings, customer data, external provider, or UI required.
"""
from __future__ import annotations
import json
import re
from pathlib import Path


def _ids(records):
    output = []
    for row in records:
        if not isinstance(row, dict):
            raise ValueError("Retrieved items must be objects")
        ref = str(row.get("chunkId") or row.get("id") or "").strip()
        if not ref:
            raise ValueError("Every retrieved item requires a nonempty chunk ID")
        if ref in output:
            raise ValueError("Duplicate retrieved chunk ID")
        output.append(ref)
    return output


def retrieval_metrics(retrieved, relevant_ids, k=5):
    """Precision/recall at k relative to explicit human-labeled chunk IDs.

    Empty ground truth is not silently scored as perfect recall.
    """
    if not isinstance(k, int) or isinstance(k, bool) or not (1 <= k <= 100):
        raise ValueError("k must be an integer from 1 through 100")
    found = _ids(retrieved)[:k]
    relevant = {str(item).strip() for item in relevant_ids}
    if not relevant or "" in relevant:
        raise ValueError("Human-labeled relevant chunk IDs are required")
    hits = [ref for ref in found if ref in relevant]
    return {
        "precision_at_k": round(len(hits) / k, 4),
        "recall_at_k": round(len(hits) / len(relevant), 4),
        "relevant_found": len(hits),
        "k": k,
    }


def _normalized(text):
    return re.sub(r"\s+", " ", str(text or "")).strip().casefold()


def citation_checks(retrieved, citations):
    """Refuse unsupported source IDs and quotations.

    This is literal passage evidence checking, NOT proof the full answer is
    semantically supported by the citations.
    """
    _ids(retrieved)
    by_id = {str(x.get("chunkId") or x.get("id")).strip(): x for x in retrieved}
    results = []
    for entry in citations:
        if not isinstance(entry, dict):
            raise ValueError("Citations must be objects")
        ref = str(entry.get("chunkId") or entry.get("id") or "").strip()
        quote = _normalized(entry.get("quote"))
        source = by_id.get(ref)
        reason = ("missing_source" if source is None else
                  "missing_quote" if not quote else
                  "quote_not_in_source" if quote not in _normalized(source.get("text")) else
                  "supported_excerpt")
        results.append({"chunkId": ref, "valid": reason == "supported_excerpt", "reason": reason})
    return {
        "valid": bool(results) and all(x["valid"] for x in results),
        "checked": len(results),
        "results": results,
    }


def evaluate_fixture(fixture):
    """Review a known synthetic/labeled retrieval example."""
    if not isinstance(fixture, dict):
        raise ValueError("fixture must be a JSON object")
    retrieved = fixture.get("retrieved", [])
    cited = fixture.get("citations", [])
    metrics = retrieval_metrics(retrieved, fixture.get("relevant_chunk_ids", []), fixture.get("k", 5))
    cited_result = citation_checks(retrieved, cited)
    return {
        "name": str(fixture.get("name", "fixture"))[:80],
        "retrieval": metrics,
        "citations": cited_result,
        "pass": cited_result["valid"] and metrics["recall_at_k"] >= float(fixture.get("min_recall", 0)),
    }


def evaluate_file(path):
    """Optional local CLI-friendly JSON fixture evaluation."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [evaluate_fixture(f) for f in (raw if isinstance(raw, list) else [raw])]
