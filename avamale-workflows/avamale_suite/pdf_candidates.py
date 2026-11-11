"""AA03 synthetic-only machine-readable PDF candidate parser.

Never passes data directly into a decision engine; every extracted value is
unapproved and requires explicit human confirmation. No OCR or customs claims.
"""
from __future__ import annotations
from io import BytesIO
import re

MAX_PDF_BYTES = 2_000_000
MAX_PAGES = 15
FIELDS = {
    "document_ref": r"(?:document|invoice|packing list|bill of lading|b/l)\s*(?:no\.?|number|ref(?:erence)?|#)\s*[:#-]\s*([^\n\r]{1,75})",
    "consignee": r"consignee\s*[:#-]\s*([^\n\r]{2,100})",
    "origin_country": r"(?:country of origin|origin country)\s*[:#-]\s*([^\n\r]{2,70})",
    "quantity": r"(?:total\s+)?quantity\s*[:#-]\s*(\d+(?:\.\d+)?)\b",
    "packages": r"(?:total\s+)?packages\s*[:#-]\s*(\d+)\b",
    "gross_weight_kg": r"gross\s+weight(?:\s*\(kg\))?\s*[:#-]\s*(\d+(?:\.\d+)?)\s*(?:kg|kilograms)?\b",
}

def extract_text_candidates(pages: list[str]) -> dict:
    if not 1 <= len(pages) <= MAX_PAGES:
        raise ValueError("PDF must have between 1 and 15 pages")
    candidates = []
    for page, raw in enumerate(pages, 1):
        if not isinstance(raw, str) or len(raw) > 150_000:
            raise ValueError("Invalid or oversized page text")
        for field, pattern in FIELDS.items():
            for match in re.finditer(pattern, raw, flags=re.IGNORECASE):
                value = match.group(1).strip().strip(" .\t")
                if not value or len(value) > 100:
                    continue
                left = max(0, raw.rfind("\n", 0, match.start())+1)
                right = raw.find("\n", match.end())
                excerpt = raw[left:right if right >= 0 else len(raw)].strip()[:160]
                candidates.append({"field": field, "value": value, "page": page,
                                   "excerpt": excerpt, "status": "needs_human_verification"})
    distinct = {}
    for entry in candidates:
        distinct.setdefault(entry["field"], set()).add(entry["value"].casefold())
    ambiguous = sorted(k for k, v in distinct.items() if len(v) > 1)
    for entry in candidates:
        if entry["field"] in ambiguous:
            entry["status"] = "ambiguous_requires_human"
    return {"status": "human_verification_required", "candidates": candidates,
            "missing_fields": sorted(set(FIELDS) - distinct.keys()),
            "ambiguous_fields": ambiguous,
            "notices": ["Use only fictional or explicitly authorized material",
                        "No OCR, inferred fields, or autonomous document decisions",
                        "All candidates must be verified before AA03 engine input"]}

def extract_pdf_candidates(content: bytes) -> dict:
    if not isinstance(content, bytes) or len(content) > MAX_PDF_BYTES or not content.startswith(b"%PDF-"):
        raise ValueError("Invalid PDF or exceeds 2 MB limit")
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("Install optional pypdf for PDF intake") from exc
    try:
        reader = PdfReader(BytesIO(content), strict=True)
        if reader.is_encrypted or not 1 <= len(reader.pages) <= MAX_PAGES:
            raise ValueError("Encrypted or unsupported PDF page count")
        pages = [p.extract_text() or "" for p in reader.pages]
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("Malformed or unreadable PDF") from exc
    if not any(p.strip() for p in pages):
        raise ValueError("Scanned PDF is not machine-readable")
    return extract_text_candidates(pages)
