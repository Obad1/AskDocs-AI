# Talk with a Friend: random 18+ chat and paid general advice

## Free random stranger chat
Users confirm they are 18+, accept the community rules, and select Meet a stranger. The in-memory WebSocket matchmaking queue pairs the next two eligible sessions. Users can skip, temporarily block/report, or leave without an account or invitation. No images, files, subscriptions, fees, AI or message history. Free Render restarts erase all active rooms and chat sessions.

**Safety caveat:** self-declared age is not independently verified. Reports are temporary in-memory metadata without active human review. This feature must not be marketed as fully moderated, end-to-end encrypted, safe for minors, or always available. Broad public release requires age-assurance, ongoing moderation, abuse review, escalation and user privacy/terms.

## Paid advice (NOT therapy)
Separate paid 30-minute general-advice conversations can be booked only when the owner explicitly opens availability and sets a published price. The customer provides an email and a short topic, receives a Paystack-hosted checkout, and the server independently verifies transaction reference, amount, currency and success. The host must accept the paid booking in a restricted dashboard; the customer then joins a separate private WebSocket conversation with an individual secret. No payment may unlock the conversation until server-side verification and host acceptance.

**Therapy is disabled:** do not list or advertise therapists until identity, licensing/jurisdiction, professional safeguards, clinical privacy, crisis escalation and provider agreements are established. General advice is not psychotherapy or professional clinical care.

## No-upfront subscription architecture
- FastAPI/WebSockets and vanilla frontend on an existing Render Free service, with auto-deploy disabled.
- PostgreSQL via a separately secured and provisioned free-tier database for any paid booking records. The service refuses paid checkout without DATABASE_URL.
- Hosted Paystack checkout, no subscription required for the API integration, but transaction processing fees apply.
- No paid SMS, AI, email or video services. Live support availability and connection stability are **not** guaranteed by the free tier.

## Paid release settings (all server secrets in Render, never repository/Drive)
- ADVICE_BOOKINGS_ENABLED=1: explicitly open bookings; default OFF.
- ADVICE_HOST_AVAILABLE=1: owner confirms real availability; default OFF.
- ADVICE_PRICE_NGN: owner's approved public integer price; no invented default.
- ADVICE_HOST_SECRET: strong random secret, at least 32 characters.
- PAYSTACK_SECRET_KEY: verified merchant secret; test key first; keep private.
- DATABASE_URL: persistent PostgreSQL DSN with isolated tables and backups.
- TALK_PUBLIC_URL: exact HTTPS public origin (already configured for Render).

## Release checklist
- [ ] Price, time slots, refunds/rescheduling and privacy policy explicitly approved and visible before any charges.
- [ ] Confirm merchant identity, test checkout, callback/verify and correct amount in kobo.
- [ ] Test database persistence/restart and secure customer access.
- [ ] Test host dashboard acceptance and customer/private host chat across two devices.
- [ ] Test malicious payment references, repeated confirmation, wrong amount, wrong currency and unverified payments.
- [ ] Add human report triage and adequate age/safety safeguards before public stranger matching at scale.
- [ ] Confirm mobile layout, keyboard navigation and actual production WebSocket behaviour.
- [ ] Only then set ADVICE_BOOKINGS_ENABLED=1 and ADVICE_HOST_AVAILABLE=1.

No real checkout is enabled until those settings and the release conditions are met. No real customer payments or appointments have been verified.

Paystack integration reference: https://paystack.com/docs/api/transaction/ and https://paystack.com/docs/payments/accept-payments/
