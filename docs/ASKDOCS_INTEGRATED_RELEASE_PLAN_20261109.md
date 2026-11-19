# AskDocs integrated release validation — 9 November 2026

This integration branch **combines** the isolated improvements prepared in PR #24 (hybrid local search and local guard), PR #25 (responsive workspace experience and source UX), and PR #26 (Office document parsing and page-aware chunking).

## Reconciled shared files
- `frontend/src/lib/storage/indexeddb.ts`: combines source filenames for the new library UX with cross-worker search-cache invalidation on add/delete, preserving all Dexie stores and existing data.
- `backend/app/api/v1/ingest.py`: retains 15 MiB upload cap from retrieval hardening, plus EPUB decompression bounds and the tested structured chunker. No backend remote vector indexing is enabled.

## Validation
- Dedicated integration GitHub Actions checks both Node browser relevance/Office parsing and Python Office/local-access tests.
- Existing repository CI and each independent PR remain additional safety checks.
- Run manual UI checks at 375/768/1280/1440 px, slow connection, 200% zoom, all four themes, source navigation, touch controls and actual local document ingestion.
- Validate on a nonproduction deployment with synthetic PDFs, Office files and text samples before merging to main. Do not merge an unreviewed integration branch solely because compilation passes.

## Known limitations
- This does **not** provide authenticated multi-tenant server ingestion, cloud storage or production customer privacy guarantees. The backend endpoint returns chunks but does not persist embeddings to a tenant-aware vector index.
- RRF and cross-encoder ranking scores are uncalibrated, and the UI deliberately no longer presents them as probabilities.
- Real world OCR, nested tables, layout fidelity, indexing performance, and visual screenshots require additional human-led QA before release.
- The legacy `avamale-workflows/` directory is still in AskDocs main. Its removal is PR #22, gated on separate Render deployment wiring, which has not been verified here.
- Do not alter the live user-facing UI or deployment automatically.
