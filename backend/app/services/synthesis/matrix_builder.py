"""Cross-document comparison matrix (spec §4.4 synthesis / comparison view).

Builds a term-frequency representation of each document, then computes a
document-vs-document cosine-similarity matrix and a concept-vs-document matrix.
Pure stdlib — no external model dependencies.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List

STOPWORDS = set(
    """
a an and or but if then of to in on for with as at by from is are was were be been
being this that these those it its their our your his her he she they we you i not
no can will would should may might must do does did has have had the
""".split()
)


def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z][a-z'-]{2,}", text.lower()) if t not in STOPWORDS]


def _tf(tokens: List[str]) -> Counter:
    return Counter(tokens)


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    num = sum(a[t] * b[t] for t in common)
    denom = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(
        sum(v * v for v in b.values())
    )
    return num / denom if denom else 0.0


def build_matrix(documents: Dict[str, str], top_k: int = 20) -> dict:
    """Return a comparison structure for the given ``{doc_id: text}`` map."""
    ids = list(documents.keys())
    tfs = {d: _tf(_tokenize(documents[d])) for d in ids}

    # Document-vs-document similarity matrix.
    sim_matrix = [[0.0] * len(ids) for _ in range(len(ids))]
    for i in range(len(ids)):
        for j in range(len(ids)):
            sim_matrix[i][j] = round(_cosine(tfs[ids[i]], tfs[ids[j]]), 3)

    # Global top concepts (by total frequency) for the concept-doc matrix.
    global_tf: Counter = Counter()
    for tf in tfs.values():
        global_tf.update(tf)
    concepts = [w for w, _ in global_tf.most_common(top_k)]

    concept_doc = {c: {d: tfs[d].get(c, 0) for d in ids} for c in concepts}

    return {
        "documents": ids,
        "similarity_matrix": sim_matrix,
        "concepts": concepts,
        "concept_document_matrix": concept_doc,
    }
