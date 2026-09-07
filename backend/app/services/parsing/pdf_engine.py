"""PDF text extraction via pypdfium2 (spec Section 4.1, 8.1).

Per Section 8.1 partial-success recovery: individual corrupt pages are skipped
and recorded rather than failing the whole document. A page can also be
rasterized to PNG for the multimodal OCR cascade.
"""
from __future__ import annotations

import io
from typing import Optional

import pypdfium2 as pdfium


class PdfParseError(Exception):
    pass


def extract_pdf_text(data: bytes) -> dict:
    """Return parsed pages with per-page partial recovery.

    Returns a dict: ``{pages: [{page, text}], num_pages, broken_pages,
    truncated}``.
    """
    try:
        pdf = pdfium.PdfDocument(data)
    except Exception as exc:  # corrupt file
        raise PdfParseError(f"Cannot open PDF: {exc}") from exc

    num_pages = len(pdf)
    pages: list[dict] = []
    broken_pages: list[int] = []
    try:
        for i in range(num_pages):
            try:
                page = pdf[i]
                textpage = page.get_textpage()
                text = textpage.get_text_range().strip()
                pages.append({"page": i + 1, "text": text})
            except Exception:
                broken_pages.append(i + 1)
    finally:
        pdf.close()
    return {
        "pages": pages,
        "num_pages": num_pages,
        "broken_pages": broken_pages,
        "truncated": len(broken_pages) > 0,
    }


def render_page_image(data: bytes, page_num: int, scale: float = 2.0) -> bytes:
    """Rasterize a 1-based page number to PNG bytes (for OCR fallback)."""
    pdf = pdfium.PdfDocument(data)
    try:
        if page_num < 1 or page_num > len(pdf):
            raise PdfParseError(f"page {page_num} out of range")
        page = pdf[page_num - 1]
        bitmap = page.render(scale=scale)
        pil = bitmap.to_pil()
    finally:
        pdf.close()
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def pages_to_marked_text(pages: list[dict]) -> str:
    return "\n\n".join(
        f"<<<PAGE {p['page']}>>>\n{p['text']}" for p in pages
    )
