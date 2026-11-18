# Talk with a Friend — Cloudflare migration gate (18 Nov 2026)

**Status: design + source-protocol audit; NOT DEPLOYED.** This branch is based on the unmerged real-stranger-chat source `feat/talk-with-a-friend-free-20261109`. The existing Render endpoint and DNS must remain unchanged. Do not publish a static-only copy of the landing page; that would make the primary service unusable.

## Smallest real zero-upfront hosting plan

1. Serve the existing HTML/CSS/JS from Cloudflare static assets, **on the same origin** as its APIs. Existing relative paths (`/api/sessions`, `/api/match/status`, `/api/feedback`, `/ws`) then continue to resolve to the new host.
2. Use **one named Cloudflare Durable Object namespace** as the sole matching coordinator. Route API requests and WebSocket upgrades to the same coordinator instance for the free chat pilot; no fake match or bot fallback.
3. Port the existing state machine and message protocol instead of replacing the UI: `join`, `ready`, `searching`, `matched`, `message`, `sent`, `typing`, `next`, `reported`, `partner_reconnecting`, `partner_left`, `leave`, `left`, `error`. Match actual distinct 18+ sessions, never a placeholder or an invite link.
4. Store **minimal session/matching metadata**, if needed for reconnects, using a bounded TTL and authenticated session tokens. Do **not** store chat messages. Persist report receipts where human moderation can actually retrieve them, implement retention, and explicitly disclose moderation limitations.
5. The Worker must enforce Origin and session verification, rate limits, request-size caps, no cross-user message leakage, valid pairing, disconnect cleanup, next/no-immediate-rematch, report/block, idle timeout, and capacity. Confirm Cloudflare limits and account eligibility before deploying.
6. **Paid advice remains disabled.** Existing Paystack/host WebSocket/DB routines need a separate, security-reviewed persistence and payment architecture. The final price, availability, provider verification and refund policy require explicit approval. Never put Paystack keys in client JavaScript.
7. Add two-real-browser smoke tests under the same **new Cloudflare preview URL** (not a fake in-process match): waiting queue, matching, bidirectional messages, typing, next, report and receipt, blocking, leave, reconnect, mobile, origin rejection, and concurrent users.
8. Only after all gates and owner review: verify privacy and consent terms, data transfer and safe domain switch. Keep Render for rollback until the new real-time chat is stable.

## Current source evidence

- Free chat: `talk-with-a-friend/app.py` (FastAPI in-memory matching + WebSockets).
- Frontend: `talk-with-a-friend/static/{index.html,app.js,style.css}`.
- Safety and functional tests: `talk-with-a-friend/tests/test_chat.py` and `test_browser.py`.
- Payments: `talk-with-a-friend/advice.py`; **must remain disabled**.
- Legal/privacy improvements: `feat/talk-legal-consent-20261109` is a *separate open PR*, not merged into this branch.

## Blocking items (concrete)

- No Cloudflare account/team credential or connected Cloudflare deployment tool is available here yet.
- Durable Object implementation has **not been ported** and Cloudflare WebSocket parity has **not been tested**. The existing Python/FastAPI process cannot simply run as a Cloudflare Pages static site or Vercel serverless function.
- No verified report moderation or age assurance; broad public chat is not approved.
- No real paid advice readiness; no live payment or host availability should be enabled.

This file is a staging checklist, not a release. Do not disable old hosting to save hours until the actual new service passes complete two-client tests.
