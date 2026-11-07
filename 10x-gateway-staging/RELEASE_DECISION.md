# 10X gateway release decision — 7 November 2026

## Completed (verified)
- PostgreSQL 16 in isolated GitHub Actions with psycopg3: real database claims and concurrent worker tests.
- A **real operating system process** committed a SENDING row and was forcibly terminated. A fresh independently spawned process observed the same durable row and could not claim it again. See CI run [42376686044](https://github.com/Obad1/AskDocs-AI/actions/runs/42376686044) at commit `780d1574`.
- Transaction rollback and append-only audit tests.
- A synthetic, Gmail-provider-shaped adapter handles `from_`, `to`, `labels`, `next_page_token` and refusal for partial pagination, replies or conflicting history.
- A policy module refuses dispatch if supplied proof flags (approval, account, aliases, capacity, opt-outs, replies, weekday window, exclusive claim) fail.
- A loopback-only HTTP service exposes health/readiness and five fixed synthetic cases. It cannot send mail.

## Important limitations: NO-GO
1. **Gmail real read-only connector**: the live tool has been inspected interactively, but this running Python service has no OAuth connection or independent proof of complete account-wide, historical, alias-level evidence. Do not inject a synthetic result and call that live verification.
2. **Approval and suppression authority**: the service does not yet read user approvals, opt-outs, bouncing, manual ownership or alias groups from a trusted authenticated datastore. Caller-supplied booleans are *not* trusted authorizations for live dispatch.
3. **Live provider uncertainty**: no sanctioned true Gmail send transport exists in this service; there is no real send/timeout acceptance test. A timeout stays UNCERTAIN/SENDING, never replayable.
4. **External deployment**: Render can create a public web service, but this fixture server is loopback-only and deliberately has no authentication/tenant isolation. Do not create a public service or attach Google credentials until security review/approved deployment plan.
5. **Migration**: old JSON version 945 remains an archive and the original hourly dispatcher is unchanged. A dual-read/shadow comparison, immutable history backfill, and measured release/rollback are required before switch.
6. **Limited test coverage**: one process-kill case is proven; not a full network-partition, worker restart and live Gmail provider matrix. Full company dedup must consider canonical entity aliases, campaign stages, all-mail threads and sender identities.

## Operational recommendation
Keep the existing Gmail-Sent + Google Sheets reconciliation as the source of dispatch evidence; never use a stale JSON pending state to resend. The new gateway remains **DRAFT / staging-only / send-disabled**. Work on trusted read-only service integration, permission/suppression provenance, private authenticated deployment and shadow comparisons before asking for production enablement. No silent approval.

## Production acceptance criteria
- Trusted authentication, RBAC/tenant separation, encrypted credentials and a security-reviewed non-public or authenticated service.
- An authenticated Gmail read-only adapter that proves full query pagination, account identity, aliases, reply/opt-out/bounce freshness and normalized evidence; test real provider errors.
- User-approved and event-specific send permissions retrieved from the source of truth. Strict one-send state machine, unique ledger identifiers, no auto-retry on uncertainty, shared rate limits.
- Audit evidence and read-only shadow comparison against actual Gmail Sent and Sheet statuses for an agreed observation period, then separate owner signoff for live dispatch.
- Staged deployment with smoke tests, rollback/disable switch, monitoring and incident response.

**This note documents why the remaining gates cannot be honestly certified solely from CI and currently connected tools.**
