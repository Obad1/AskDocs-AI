# AskDocs document structure extraction gate — 9 November 2026

This separate backend patch is synthetic-only and does not change the hosted frontend.

## What changed
- Word document body paragraphs and table rows are extracted in the actual XML order, preserving cross-row comparisons and cell values.
- PowerPoint extraction includes text frames, grouped shapes and table cells; slide numbers become page markers used for source citations.
- Backend chunking honors token budgets, exact word overlap and page boundaries, including long single paragraphs.
- EPUB zip decompression checks are bounded by file count and cumulative uncompressed size, in addition to the existing upload size cap.
- No OCR, embedding APIs or proprietary document processing services are added.

## Remaining release gates
- Browser-side JSZip parsing now preserves DOCX body order and table cells, and PPTX table rows and slide citations. Synthetic tests cover both paths. Complex nested tables, shapes, unsupported OOXML variants and real-world parity still require manual review.
- The backend still needs authenticated workspace ownership before connecting ingest to a remotely reachable vector store. Do not enable customer documents on this route yet.
- Tables inside nested Word cells, image tables, PDF layout analysis and complex scanned documents need real-world evaluation. No claim of complete extraction is made.
- Use test fixtures with fabricated invoice/shipment content; do not upload customer data for development.

## Verification
Run Python synthetic tests in `backend/tests/test_document_structure.py` and browser JSZip tests in `frontend/tests/office-structure.test.mjs`. CI must pass before merge; deployment references and backend API security need independent review.
