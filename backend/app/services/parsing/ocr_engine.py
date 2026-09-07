"""OCR engine wrapper around pytesseract (spec Section 4.1, 8.1 multimodal).

pytesseract shells out to a locally-installed Tesseract binary; no network or
API key is involved. Used as the structural-parse fallback when text density is
below the cascade threshold.
"""
from __future__ import annotations

import io
from typing import Optional

import pytesseract
from PIL import Image

from app.services.parsing.pdf_engine import render_page_image


def ocr_image_bytes(data: bytes, lang: str = "eng") -> str:
    img = Image.open(io.BytesIO(data))
    return pytesseract.image_to_string(img, lang=lang).strip()


def ocr_pdf_page(pdf_bytes: bytes, page_num: int, lang: str = "eng") -> str:
    """Render a PDF page to PNG and OCR it."""
    png = render_page_image(pdf_bytes, page_num)
    return ocr_image_bytes(png, lang=lang)
