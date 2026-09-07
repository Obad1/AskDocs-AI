"""Ingestion API router (spec Section 4.1, 6.1, 8.1).

POST /ingest accepts an uploaded file (or a ``url`` for YouTube) and runs the
server parsing pipeline: format detection -> parse (with OCR fallback cascade)
-> NFKC clean -> chunk -> SHA-256 dedup hash. It returns the generated
``doc_id`` and chunks. Embeddings are produced client-side (per the
architecture), so this endpoint stops at chunked text.

Mounted automatically by ``app.api.router`` (see ``_SIBLING_ROUTERS``).
"""
from __future__ import annotations

import hashlib
import io
import os
import re
import tempfile
import unicodedata
import uuid
import zipfile
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.core.security import LocalOnly
from app.services.parsing import (
    docx_pptx_engine,
    media_transcriber,
    ocr_engine,
    pdf_engine,
    youtube_fetcher,
)

router = APIRouter(prefix="/ingest", tags=["ingest"])

_EXT_MAP = {
    ".pdf": "PDF",
    ".docx": "DOCX",
    ".pptx": "PPTX",
    ".epub": "EPUB",
    ".mp3": "AUDIO",
    ".wav": "AUDIO",
    ".m4a": "AUDIO",
    ".ogg": "AUDIO",
    ".flac": "AUDIO",
    ".mp4": "VIDEO",
    ".mov": "VIDEO",
    ".webm": "VIDEO",
    ".mkv": "VIDEO",
    ".avi": "VIDEO",
}
_MIME_MAP = {
    "application/pdf": "PDF",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "DOCX",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "PPTX",
    "application/epub+zip": "EPUB",
}
_TOKEN_FLOOR = 20  # Section 8.1 cascade threshold (tokens/page)


class IngestionChunk(BaseModel):
    chunk_id: str
    text: str
    page: Optional[int] = None


class IngestResponse(BaseModel):
    doc_id: str
    format: str
    hash_sha256: str
    num_chunks: int
    chunks: list[IngestionChunk]
    broken_pages: list[int] = []
    truncated: bool = False
    warnings: list[str] = []


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _detect_format(filename: Optional[str], mime: Optional[str]) -> Optional[str]:
    ext = os.path.splitext(filename or "")[1].lower()
    if mime and mime in _MIME_MAP:
        return _MIME_MAP[mime]
    return _EXT_MAP.get(ext)


def _clean_text(raw: str) -> str:
    text = unicodedata.normalize("NFKC", raw)
    text = "".join(ch for ch in text if ch == "\n" or ch == "\t" or ord(ch) >= 32)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _epub_text(data: bytes) -> str:
    z = zipfile.ZipFile(io.BytesIO(data))
    texts: list[str] = []
    for name in z.namelist():
        if (
            name.endswith((".xhtml", ".html", ".htm", ".xml"))
            and "META-INF" not in name
            and "container.xml" not in name
        ):
            raw = z.read(name).decode("utf-8", "ignore")
            raw = re.sub(r"<[^>]+>", " ", raw)
            raw = re.sub(r"\s+", " ", raw).strip()
            if raw:
                texts.append(raw)
    return "\n\n".join(texts)


def _chunk_text(text: str, target: int = 500, overlap: int = 50) -> list[dict]:
    blocks: list[tuple[str, Optional[int]]] = []
    cur_page: Optional[int] = None
    buf: list[str] = []
    for line in text.split("\n"):
        m = re.match(r"^<<<PAGE (\d+)>>>$", line)
        if m:
            if buf:
                blocks.append(("\n".join(buf).strip(), cur_page))
                buf = []
            cur_page = int(m.group(1))
        else:
            buf.append(line)
    if buf:
        blocks.append(("\n".join(buf).strip(), cur_page))

    chunks: list[dict] = []
    for body, page in blocks:
        if not body:
            continue
        paras = [p for p in re.split(r"\n{2,}", body) if p.strip()]
        acc: list[str] = []
        toks = 0
        for p in paras:
            t = len(p.split())
            if toks + t > target and acc:
                chunks.append({"text": " ".join(acc), "page": page})
                acc = acc[-overlap:] if overlap else []
                toks = sum(len(x.split()) for x in acc)
            acc.append(p)
            toks += t
        if acc:
            chunks.append({"text": " ".join(acc), "page": page})
    return chunks


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def _run_pipeline(data: bytes, filename: str, fmt: str) -> dict:
    warnings: list[str] = []
    broken_pages: list[int] = []
    truncated = False
    marked = ""

    if fmt == "PDF":
        result = pdf_engine.extract_pdf_text(data)
        pages = result["pages"]
        broken_pages = result["broken_pages"]
        truncated = result["truncated"]
        filled = []
        for pg in pages:
            toks = len(pg["text"].split())
            if toks < _TOKEN_FLOOR:
                try:
                    ocr = ocr_engine.ocr_pdf_page(data, pg["page"])
                    pg = {**pg, "text": ocr or pg["text"]}
                except Exception as exc:
                    warnings.append(f"OCR failed page {pg['page']}: {exc}")
            filled.append(pg)
        marked = pdf_engine.pages_to_marked_text(filled)

    elif fmt == "DOCX":
        marked = docx_pptx_engine.extract_docx_text(io.BytesIO(data))

    elif fmt == "PPTX":
        marked = docx_pptx_engine.extract_pptx_text(io.BytesIO(data))

    elif fmt == "EPUB":
        marked = _epub_text(data)

    elif fmt in ("AUDIO", "VIDEO"):
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=os.path.splitext(filename)[1] or ".bin"
        )
        tmp.write(data)
        tmp.close()
        try:
            marked = media_transcriber.transcribe_audio(tmp.name)
        finally:
            os.unlink(tmp.name)

    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {fmt}")

    return {
        "marked": marked,
        "broken_pages": broken_pages,
        "truncated": truncated,
        "warnings": warnings,
    }


def _run_youtube(url: str) -> dict:
    fetched = youtube_fetcher.fetch_youtube(url)
    text = fetched.get("captions") or ""
    warnings: list[str] = []
    if not text and fetched.get("audio_path"):
        try:
            text = media_transcriber.transcribe_audio(fetched["audio_path"])
        except Exception as exc:
            warnings.append(f"Whisper fallback failed: {exc}")
        finally:
            if fetched.get("audio_path") and os.path.exists(fetched["audio_path"]):
                os.unlink(fetched["audio_path"])
    return {"marked": text, "warnings": warnings}


# --------------------------------------------------------------------------- #
# Route
# --------------------------------------------------------------------------- #
@router.post("", response_model=IngestResponse)
async def ingest(
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None),
    format_hint: Optional[str] = Form(None),
    _: str = Depends(LocalOnly),
) -> IngestResponse:
    doc_id = str(uuid.uuid4())
    warnings: list[str] = []

    if url:
        fmt = "YOUTUBE"
        pipe = _run_youtube(url)
        marked = pipe["marked"]
        warnings.extend(pipe["warnings"])
        broken_pages: list[int] = []
        truncated = False
    elif file is not None:
        data = await file.read()
        filename = file.filename or "upload"
        fmt = format_hint or _detect_format(filename, file.content_type) or "PDF"
        if fmt not in _EXT_MAP.values() and fmt not in ("EPUB",):
            # Map unknown by extension fallback to PDF attempt.
            fmt = "PDF"
        pipe = _run_pipeline(data, filename, fmt)
        marked = pipe["marked"]
        broken_pages = pipe["broken_pages"]
        truncated = pipe["truncated"]
        warnings.extend(pipe["warnings"])
    else:
        raise HTTPException(status_code=400, detail="Provide either 'file' or 'url'.")

    cleaned = _clean_text(marked)
    raw_chunks = _chunk_text(cleaned)

    hash_sha256 = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()

    chunks = [
        IngestionChunk(chunk_id=f"{doc_id}::chunk-{i}", text=c["text"], page=c["page"])
        for i, c in enumerate(raw_chunks)
    ]

    return IngestResponse(
        doc_id=doc_id,
        format=fmt,
        hash_sha256=hash_sha256,
        num_chunks=len(chunks),
        chunks=chunks,
        broken_pages=broken_pages,
        truncated=truncated,
        warnings=warnings,
    )
