"""Page-aware fixed-budget chunking for document search.

Small standard-library-only primitive: never merges separate PDF/slide pages
and guarantees each chunk contains at most target whitespace tokens.
"""
from __future__ import annotations

import re
from typing import Any


_PAGE = re.compile(r"^<<<PAGE\s+(\d+)>>>$")


def chunk_marked_text(
    text: str, target: int = 500, overlap: int = 50
) -> list[dict[str, Any]]:
    """Return chunks with text and source page.

    Overlap is an exact number of words, not whole paragraphs. This preserves
    page attribution and bounds LLM context even for a single huge paragraph.
    """
    if target < 1 or overlap < 0 or overlap >= target:
        raise ValueError("target must be positive and 0 <= overlap < target")

    blocks: list[tuple[str, int | None]] = []
    page: int | None = None
    lines: list[str] = []

    def flush() -> None:
        body = "\n".join(lines).strip()
        if body:
            blocks.append((body, page))
        lines.clear()

    for line in text.splitlines():
        match = _PAGE.fullmatch(line.strip())
        if match:
            flush()
            page = int(match.group(1))
        else:
            lines.append(line)
    flush()

    chunks: list[dict[str, Any]] = []
    for body, source_page in blocks:
        words = body.split()
        stride = target - overlap
        for start in range(0, len(words), stride):
            end = min(len(words), start + target)
            chunks.append({"text": " ".join(words[start:end]), "page": source_page})
            if end == len(words):
                break
    return chunks
