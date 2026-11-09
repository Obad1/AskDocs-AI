# AVAMALE Operations Studio — standalone repository handoff

**Prepared:** 9 November 2026
**Source:** Obad1/AskDocs-AI, branch `project/avamale-operations-standalone-20261109`
**Root of source branch:** originally `avamale-workflows/`, now standalone at `/`
**Status:** independently structured Git branch; **new GitHub repository not yet created**.

## Target repository

Suggested name: `Obad1/Avamale-Operations-Studio` (prefer private for future commercial development). This standalone content is already public through AskDocs-AI; moving it into a private repository does not erase that history.

The new repo should contain the following at root: `avamale_suite/`, `product_app/`, `tests/`, `samples/`, `docs/`, `.github/workflows/`, `requirements.txt`, `README.md`, `Dockerfile.product-app` and other existing suite files. Do not nest everything under another `avamale-workflows/` directory.

## Local import without carrying AskDocs commit history

After the new empty repository is created on GitHub, on a development machine:

```bash
git clone --depth 1 --single-branch --branch project/avamale-operations-standalone-20261109 https://github.com/Obad1/AskDocs-AI.git avamale-operations-stage
mkdir avamale-operations-studio
cp -R avamale-operations-stage/. avamale-operations-studio/
cd avamale-operations-studio
rm -rf .git
git init -b main
git add .
git commit -m "Initialize standalone AVAMALE operations studio"
git remote add origin https://github.com/Obad1/Avamale-Operations-Studio.git
git push -u origin main
```

If the destination was initialized with a README or license, do not force-push. Reconcile its history first. Confirm repo privacy, branch protections, GitHub Actions permissions, and collaborators.

## Verification before cutting over

1. Install Python 3.11+, Node.js 22+.
2. Run `python -m pip install -r requirements.txt -r product_app/requirements.txt pytest httpx`.
3. Run `python -m pytest -q tests` and `node tests/test_engine.cjs`.
4. Run `python -m avamale_suite list` and `python -m avamale_suite demo --out ./demo_outputs_new`.
5. Check `.github/workflows/test.yml` on the target repository.
6. Audit environments, secrets, databases, storage durability, and any Render service/CI refs before switching deployment.
7. Only then open an AskDocs removal PR, deleting `avamale-workflows/` and its root `.github/workflows/avamale-product-app.yml` and `.github/workflows/avamale-suite.yml`. Update AskDocs README to stop linking into the removed folder. Run AskDocs tests independently; confirm hosted service is unchanged.

## Guardrails

This is a synthetic-demo-only, human-reviewed software suite, not an independently production-certified SaaS platform. Do not publish customer data, deploy sensitive features to the public internet, or enable unreviewed automations.