# AVAMALE Operations Studio · AA01–AA18

**A runnable web-based pilot workbench, not 18 finished production SaaS deployments.**

This is a substantial extension of the existing distinct AA01–AA18 review engines. It gives every product its own searchable workspace, input schemas, upload flow, recorded findings, a persistent case queue, human reviewer decisions with history, and CSV/Excel/JSON delivery. It is delivered as **one web platform with 18 specialized workspaces**, so the source is maintainable and a single deployment can demo the entire catalog. AA02 and AA03 retain their deeper transaction and freight matching engines.

## Quick start (local only)

Run from the `avamale-workflows` directory:

```bash
python -m pip install -r requirements.txt -r product_app/requirements.txt
export AVAMALE_ADMIN_PASSWORD='use-a-unique-long-password-here'
export AVAMALE_SESSION_SECRET='use-at-least-32-random-characters-here'
python -m product_app.server
```

Open **http://127.0.0.1:8000**, sign in, select any AA01–AA18 workspace, and use **Use included sample data** to run the synthetic workflow. Every run persists to a SQLite DB in `private_data/` (excluded from source control). No external network services or sending are needed for demos.

For reproducible acceptance testing:

```bash
python -m pytest tests/test_product_app.py -q
python -m unittest discover -s tests -p 'test_*.py' -q
node --check product_app/static/app.js
node tests/test_engine.cjs
```

## What each workspace actually does

1. Accepts one or two supported CSV/values-only XLSX structured sample files, validating required fields and file limits.
2. Executes its specific rule-bound engine (see `avamale_suite/workflows.py` or `avamale_review.py`).
3. Saves the **findings, input checksums, report timestamps, and file names** to SQLite; raw uploaded files are temporary and deleted after processing.
4. Creates per-finding reviewer statuses: **pending, accepted, rejected, needs_info**, with notes and append-only audit entries.
5. Exports a recorded result and reviewer decisions as **CSV, XLSX, JSON** without executing real-world actions or contacting customers.
6. Supports an optional consent-gated enquiry endpoint, **off by default** until privacy/security reviews and hosting are approved. Leads are not emailed automatically.

Each product is accessible at `/products/AA01` ... `/products/AA18` as a dedicated navigable workspace. The shared console is intentionally a reusable foundation, not 18 duplicated unmaintainable frontends.

## Production blockers (critical)

- **Do not upload real bank/ledger, security, customer, staff or patient data.** This release is synthetic-only, and the operator declaration cannot independently detect private records.
- Single-operator password authentication is **not multi-user/tenant authorization**. Must add SSO/roles, per-tenant isolation, production database, backup, encryption, retention/deletion, audit controls, security and penetration testing before customer production deployment.
- No QuickBooks/Xero/Zoho integrations, no OCR/PDF extraction, no live ticket system, no payments, email sending, bank feeds, customs, clinical or legal decisions, KYC verification or threat detection.
- No automated reconciliation approval, ledger posting, maintenance dispatch, message sending, account disabling or medical decision.
- The role of each vertical domain expert and the exact customer fields/acceptance criteria must be documented before live pilot promises.
- Processing and storing any actual client data requires appropriate authorization, signed NDA/DPA where applicable, approved security controls, and relevant data-protection assessment.

**This is a sellable demo/paid-diagnostic platform only once prospect scope and protections are agreed, not proof of customer-ready SaaS.** Invoice amounts and contract terms are not set or accepted by this application.

## Hosting notes

The included `Dockerfile` is a demonstration deployment baseline only. A public deployment must set `AVAMALE_PUBLIC=1`, `AVAMALE_COOKIE_SECURE=1`, a unique admin password and session secret, HTTPS termination and **a durable storage volume for SQLite**. Do not deploy to a public URL while sensitive workflows are enabled without independent controls. If using Render's ephemeral filesystem, saved runs will be lost on redeploy/restart unless a persistent disk is configured. No live customers are assumed.

`AVAMALE_ALLOW_LEADS=1` enables consent-based public enquiries only after a privacy notice, controlled admin access, retention policy and deployment protections are in place. The form records messages; it does **not** send confirmation emails or contact prospects on its own.

## AA01–AA04 durable synthetic PostgreSQL pilot

The Operations Studio can use an existing Neon PostgreSQL database instead of ephemeral SQLite. The application now selects `PostgresStore` whenever `DATABASE_URL` is defined. It preserves the existing HTTP contract and local SQLite tests. **No raw uploaded files are retained**. Each case, finding, human reviewer decision, and audit event is stored transactionally.

- Local/offline: leave `AVAMALE_PUBLIC=0` and `DATABASE_URL` empty to use the existing SQLite demonstration database.
- Persistent staging: configure `DATABASE_URL` as an environment secret containing the Neon PostgreSQL SSL connection string; use a dedicated **synthetic** branch/database and least-privilege role. Never commit connection strings.
- Public HTTPS demonstration: `AVAMALE_PUBLIC=1`, `AVAMALE_COOKIE_SECURE=1`, `AVAMALE_ADMIN_PASSWORD` (long unique secret), `AVAMALE_SESSION_SECRET` (32+ random characters), `AVAMALE_ALLOW_LEADS=0`, and `DATABASE_URL` must be set. Public mode fails closed without persistent PostgreSQL or secure cookies.
- Existing Render AskDocs-AI service is a separate product and must not be overwritten. Create a **distinct free web service** linked to a reviewed commit, root `avamale-workflows`, build `pip install -r requirements.txt -r product_app/requirements.txt`, start `python -m product_app.server`. Configure credentials and an isolated Neon branch in Render secrets. Do not use an ephemeral SQLite file on Render.
- The current app is **single-operator and synthetic-data-only**; it has neither multi-tenant access nor approved real customer intake. A declaration that uploaded files are synthetic cannot reliably detect confidential contents. Never invite customers to upload real invoices, bank files, security questionnaires or personal data until the security program is complete.
- Run `python -m pytest tests/test_postgres_store.py -v` with disposable `TEST_POSTGRES_URL`. GitHub workflow `avamale-postgres-pilot.yml` provides a temporary PostgreSQL 16 service, validates authenticated AA01–AA04 upload/review/export across app restarts and checks the existing SQLite regressions. Never run destructive cleanup tests against Neon.
- `/healthz` additionally verifies PostgreSQL connectivity, returning HTTP 503 on storage failure. It preserves the original JSON response shape so existing clients are unaffected.
- The existing Neon project `tiny-pond-50909066` has an isolated branch `aa01-aa04-persistent-pilot-20261108` (`br-misty-moon-bcedl3ga`). Existing records remain untouched; a production migration or real customer pilot is **not** implied by this branch.
- Deployment acceptance: (1) all CI jobs green; (2) verified isolated Render service URL; (3) synthetic AA01, AA02, AA03, AA04 runs appear and survive a restart/redeploy; (4) reviewer history and exports survive; (5) unauthenticated requests, CSRF-bypassed writes and real-file upload attempts are rejected; (6) no public customer-data intake or outreach enabled. Until all are observed, label the milestone **code tested, deployment unverified**.

### Boundaries after this milestone

A durable synthetic demonstration backend is not the same as a production service. Separate customer approval, privacy notice, NDA/DPA where relevant, tenant-level authorization, backup and restore tests, retention/deletion policy, malware/content scanning, upload provenance checks, security audit and authorized operator roles are required before real-document intake. This repository does **not** implement those controls.
