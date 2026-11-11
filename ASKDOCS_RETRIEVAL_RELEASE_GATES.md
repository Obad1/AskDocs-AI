# AskDocs AI — Local-first RAG reliability milestone

**Branch:** \`feat/askdocs-free-hybrid-rag-foundation-20261109\`  
**Status:** proposed release, not deployed to production.

## Implemented in this branch

- Replace the misleading FlexSearch \`BM25Index\` with real in-memory Okapi BM25 scoring, including IDF, term frequency, document-length normalization, overwrite and deletion.
- Broadcast index invalidation when IndexedDB adds or deletes documents so searches rebuild from up-to-date embeddings and chunks across the indexing worker and browser main thread.
- Prevent concurrent redundant index builds; keep a bounded retry when data changes during rebuilding.
- Replace cosine-on-independent-embeddings "reranking" with actual MS MARCO sentence-pair classification scores. The model is loaded only when reranking is explicitly enabled and retrieval falls back safely if unavailable. Official model usage: https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2
- Fix token-overlap selection in the frontend chunker (previously it retained the last N **paragraphs**, not tokens).
- Correct local-only auth: forged HTTP \`X-Forwarded-For\` or \`Origin\` cannot cause a remote request to be treated as loopback.
- Bound server-side ingestion reads to 15 MiB and return HTTP 413 for larger uploads.
- Add CI for frontend TypeScript/build, BM25 behaviors and backend local-only security checks.

## Explicitly not done yet

The browser stores embeddings in IndexedDB and already does semantic search locally. The **backend** ingestion router currently returns chunks without writing embeddings to Chroma/Qdrant, so hosted backend RAG still needs a secure persistence plan before public enablement. That is not solved by this change.

The public backend has no user accounts or tenant-level access enforcement. Do **not** remove local-only access checks or expose document uploads / workspace / generation endpoints publicly. Design authenticated ownership, deletion, rate limits, and durable storage first. A service behind an untrusted reverse proxy must be assessed for whether its actual ASGI peer is loopback; do not infer user identity from forwarded IP headers.

Docling/Xberg/MarkItDown should be benchmarked on consented synthetic PDF, DOCX tables, scanned PDFs, and PPTX before replacing the current parsers. No new hosted API or paid provider is required for this milestone.

## Mandatory merge/deploy gates

1. GitHub Actions workflow \`AskDocs local retrieval and security checks\` must pass on this PR head.
2. Exercise upload -> search -> delete -> search, twice, in an actual browser (normal and private browsing) to verify worker-side BroadcastChannel invalidation and IndexedDB cleanup.
3. Test scanned PDF and tables; confirm citation source page and chunk IDs survive.
4. Test the opt-in ONNX reranker availability on a low-memory device; keep fallback to RRF.
5. Inspect existing Render runtime before deployment: backend-only API callers may be affected by stricter LocalOnlyAuth, while the browser-local worker path should continue unaffected.
6. Do not merge the Operations Studio removal PR until its new repo and hosted dependencies are verified.

No real customer documents, credentials or new cloud accounts should be used for these tests.
