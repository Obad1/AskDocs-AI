# Standalone GitHub project staging — 9 November 2026

This branch presents the **AVAMALE Operations Studio / AA01-AA18 review suite** at repository root, independent of AskDocs application files. It was extracted from `Obad1/AskDocs-AI/avamale-workflows` without changing the original default branch.

**Current status:** isolated GitHub branch, **not yet a separately created GitHub repository**, nor a production deployment. This branch is public because its parent repo is public. The original code has already been publicly available.

**To finish the move:** create a distinct repository under Obad1, import this standalone tree as its root, run the CI and synthetic acceptance tests, confirm deployment root/commands, then open a separately reviewed removal PR for the old embedded `avamale-workflows/` directory and its root CI references. Do not delete either implementation before the target is verified.

**Exclusions:** AG Shield risk scoring; customer files; credentials; private agreements; PII; unapproved payment/email integrations. Do not alter `DamilolaDami/Luno` or AskDocs default deployment.