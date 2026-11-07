# AA01-AA18 Neon staging migration (not deployed)

This feature branch adds **opt-in** PostgreSQL storage while leaving local SQLite as the default. Only synthetic data is authorized. Do not expose the console publicly or use customer data. The existing root AskDocs Render service deploys from `main`; this feature branch must not be merged or deployed without separate approval.

## Files changed
- `product_app/postgres_store.py`: PostgreSQL implementation of the same case/enquiry contract.
- `product_app/api.py`: explicit storage selector with fail-closed conflicting settings.
- `product_app/requirements.txt`: psycopg 3 PostgreSQL driver.
- `product_app/.env.example`: configuration example.
- `tests/test_storage_selection.py`: offline configuration regression tests.
- `tests/test_neon_staging_http.py`: guarded full-catalog FastAPI/Neon acceptance test.

## Local SQLite regression (no cloud required)

From `avamale-workflows/`:

```bash
python -m pip install -r requirements.txt -r product_app/requirements.txt pytest httpx
python -m pytest -q tests/test_product_app.py tests/test_storage_selection.py
python -m pytest -q tests
node --check product_app/static/app.js
node tests/test_engine.cjs
```

## Isolated disposable Neon acceptance

Run **only** with credentials injected securely in an isolated runner. Never commit the URL to GitHub, logs, a report, or source control. The test creates synthetic records and does not delete existing records; use an empty dedicated database, not a customer or shared database.

```bash
export AVAMALE_TEST_DATABASE_URL='<SECRET - inject in environment>'
export AVAMALE_TEST_DATABASE_NAME='avamale_synthetic'
export AVAMALE_RUN_NEON_ACCEPTANCE=1
python -m pytest -q tests/test_neon_staging_http.py
```

There is no preconfigured Neon secret in GitHub Actions. The live HTTP test is skipped by default. No real Neon HTTP test has passed yet. The earlier direct Neon SQL smoke test verifies schema, 18 case IDs, decisions, events, and rollback only.

## Render staging proposal — separate approval required

Create a separate **private/restricted** Render service, not a replacement for the existing AskDocs-AI root service. Use `avamale-workflows` as root directory, `Dockerfile.product-app` as Dockerfile, and the selected feature branch; prevent automatic branch deployments until reviewed. Configure secrets through the Render dashboard only:

- `AVAMALE_STORAGE_BACKEND=postgres`
- `DATABASE_URL=<Neon secure URL>`
- `AVAMALE_ADMIN_PASSWORD=<strong private secret>`
- `AVAMALE_SESSION_SECRET=<32+ random characters>`
- `AVAMALE_COOKIE_SECURE=1`
- `AVAMALE_ALLOW_LEADS=0`
- `AVAMALE_PUBLIC=1` is required by current server launcher for Render bind, but it must not be enabled until access restrictions are reviewed and approved.
- **Do not set `AVAMALE_DB_PATH` for PostgreSQL**.

Important: `AVAMALE_PUBLIC=1` binds to all interfaces; it is *not* proof of authenticated or restricted public access. The single-operator shared password is not production tenancy/RBAC. Public access and real customer data require separate security approval.

## Outstanding gates

1. CI on a PR (branch-only commits do not trigger existing main/PR workflows).
2. Full SQL-backed FastAPI integration on an isolated checkout.
3. Cold start, connection errors, concurrent reviewer updates.
4. Restore from an independent backup into another disposable database.
5. Security review of session management, reviewer identity, access restrictions, and data retention.
6. Separate approval before any merge, Render changes, or deployment.

Do not claim that successful direct SQL tests validate the whole application.
