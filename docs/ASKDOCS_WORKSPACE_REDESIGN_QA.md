# AskDocs workspace usability and structure — November 2026

## Scope
This PR retains the existing AskDocs identity and four theme palettes while repairing document navigation and responsive interaction. It does **not** enable customer-data hosting or change the production deployment.

### Changes
- Desktop resizing: pointer drag and keyboard arrows, Home/End, double-click reset; locally remembers the split.
- Mobile: Document / Assistant view switcher to avoid unusable side-by-side columns.
- Top bar: removes the fake non-functional workspace selector, shows meaningful library/mode information.
- Document library: search, original file names for new uploads, accessible delete controls and mobile slide-out.
- IndexedDB workspace: delay persistence until saved state has loaded; removes a data-loss race.
- Ingestion: detect duplicate hashes and remove attempted duplicate records rather than accumulating orphan entries.
- Reading surface: text preview styled as a page; citation scrolling waits for the page, stable effect callback (prevents endless rerender).
- Chat: meaningful empty state and suggested prompts, responsive composer, clearer model status, and visible error feedback.
- Zen: transparent pointer-pass-through overlay so focus mode does not hide the reading content.
- Closed dialogs: lazily mount only when opened rather than loading heavy modules during first render.
- Contextual drawers: Escape key, focus return and Tab loop.

### Manual acceptance checklist (not automatically proven by the CI build)
- At 375, 768, 1280 and 1440 px: no horizontal overflow or tiny chat/document columns.
- Mobile Browse files opens and closes over the workspace; the Document/Assistant switch retains in-progress content.
- Sidebar shows filenames for freshly ingested documents, search matches by name, deletion works on touch.
- Refresh a library with stored documents: no temporary empty-workspace save overwrites the content.
- Upload the same content twice and confirm duplicate is rejected without a second document row.
- Read both PDF-canvas preview (when bytes supplied) and stored text fallback; citation jumps work after async render.
- Ask a question with no documents and validate the response and evidence states.
- Test dark/sepia/graphite themes, 200% browser zoom, keyboard-only sidebar/drawer/resize behavior.
- Test Zen mode: original document must remain visible under the controls.
- Check slow network and offline reload: closed optional modules do not flash loading overlays.

### Outstanding separate engineering work
- PR #24: local retrieval/embedding and backend guard hardening (review independently).
- PR #22: migration cleanup; blocked until dedicated Operations Studio deployment references are reviewed.
- Hosted ingestion currently requires authenticated workspace ownership, quotas, durable storage, content security checks and index consistency before customer use.
- Accurate calibrated retrieval confidence remains unproven. Do not interpret RRF score as a probability or production guarantee.
- End-to-end screenshot and user acceptance testing have not been done by this PR.

### Safety
No real customer documents are required in QA. Use synthetic samples only. Keep live branch unchanged until all CI gates, deployment checks and review are complete.
