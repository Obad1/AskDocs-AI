# AskDocs / AVAMALE Operations Studio — extraction and release gates

**Prepared:** 9 November 2026. **Status:** candidate cleanup ONLY. Do not merge until the independent repository is created, verified, and deployment consumers are re-pointed.

## Distinct ownership

- **AskDocs AI** remains in `Obad1/AskDocs-AI`, with its original `backend/`, `frontend/`, UI, document parsing, and deployment files.
- **AVAMALE Operations Studio** contains AA01–AA18 workflow engines and the synthetic pilot console. The validated source tree is staged at [project/avamale-operations-standalone-20261109](https://github.com/Obad1/AskDocs-AI/tree/project/avamale-operations-standalone-20261109).
- **AvamPlay** remains its own mixed frontend/backend product at `Obad1/Avam-Play`. Do not extract its tightly integrated components while Render uses that repo.
- Do not move any collaborator-owned repositories or private data.

## Tests and proof so far

- The standalone Operations Studio branch's GitHub Actions [test run 42559737944](https://github.com/Obad1/AskDocs-AI/actions/runs/42559737944) completed successfully. This is **synthetic acceptance and build verification, not customer-data readiness or a deployed production instance**.
- AskDocs `main` was unchanged when this cleanup branch was created.

## Mandatory before merging this proposal

1. Create `Obad1/Avamale-Operations-Studio` as an **empty, preferably private** GitHub repository. GitHub API access available to this assistant cannot create new repositories.
2. Import the isolated staging branch's *working tree* into the new repository's root, ensuring `avamale_suite/`, `product_app/`, `tests/`, `samples/`, `docs/` and `.github/workflows/test.yml` are present. Use the staging branch's `STANDALONE_MIGRATION.md` handoff.
3. Confirm GitHub Actions pass in the **new repository** and verify checkout, test imports, sample data, container commands, secrets, and collaborator access.
4. Check Render and all CI/deploy integrations for old paths `AskDocs-AI/avamale-workflows`. Migrate each only after reviewing dependency and data persistence, not automatically.
5. Confirm AskDocs `main` still deploys independently and its **own** tests pass.
6. Only then merge this AskDocs-only cleanup. Keep rollback pointers and old Git history intact.

## Proposed changes in this branch

- Remove embedded `avamale-workflows/` subtree from AskDocs source tree (only from this candidate branch).
- Remove two workflow-specific GitHub Actions files from AskDocs.
- Update AskDocs README to stop advertising the operational review suite as part of AskDocs.

This branch is intentionally a **blocked draft** until there is a real separate repository. Do not force-push or delete the historical source.
