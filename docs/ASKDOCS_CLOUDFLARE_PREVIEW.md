# AskDocs Cloudflare Pages migration — review-only (18 November 2026)

**Status:** staging preparation, not a deployment, DNS change, or working-product certificate.

## Cloudflare Pages project settings

- GitHub repo: `Obad1/AskDocs-AI`
- Deploy **branch**: `migration/askdocs-cloudflare-preview-20261118` only
- Root directory: `frontend`
- Framework preset: `Vite`
- Build command: `npm run build`
- Build output directory: `dist`
- Node.js runtime: `22` (do not expose backend credentials at build time)
- Use Cloudflare's default free `*.pages.dev` preview URL; no custom domain or traffic cutover.

Do not configure automatic production deployments from `main` before review. The app is a browser-side React/Vite application using workers, WebGPU/web inference and IndexedDB. Those capabilities require genuine browser tests; a build alone does not prove them.

## Acceptance before switching traffic

1. Build succeeds; static assets have no unresolved paths.
2. Desktop and mobile visual checks: approved AVAMALE identity, brand colour, readable chat and evidence, buttons and no horizontal overflow.
3. In a clean browser profile, load the three fictional sample documents **through the real indexing worker**, ask a question, and inspect an actual supporting excerpt. Record real screenshots and an MP4; never fabricate an answer.
4. Check caching, WebAssembly/model downloads, relevant CORS/CSP headers, model size and browser requirements. Server-only integrations must remain available or be explicitly disabled.
5. Validate IndexedDB persistence and deletion; documents stored under the Render origin do **not** automatically transfer to `pages.dev`. Before cutover give users an export/backup method and a clear transfer notice.
6. Only after review, associate a verified domain and move public links; leave the original Render service in place for rollback until traffic is confirmed.

**Important:** No real user files in CI, videos or screenshots. No live domain, billing or Render change is authorized by this branch.
