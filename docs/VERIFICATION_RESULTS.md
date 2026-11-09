# AVAMALE workflow verification — 7 November 2026

## Public branch (as published, excluding AG Shield)

A clean reconstruction of the 72-file public Git tree was executed outside the original working directory:

```text
python -m unittest discover -s tests -p 'test_*.py' -v
Ran 54 tests in 2.287s
OK

node tests/test_engine.cjs
Browser engine assertions and Python/JS fixture parity passed.

python -m avamale_suite list
All AA01–AA18 workflows enumerated.
```

Earlier development runs with fewer tests (35 and 45) were superseded by this public-snapshot verification. The full *local-only* development tree, including a separate unpublished AG Shield demo, passed 60 Python tests and the JavaScript checks. Do **not** treat that 60-test total as the public branch's test count.

## Demo and readiness boundary

Synthetic AA01–AA18 demonstration reports were generated locally. Output files are human-review artifacts, not autonomous decisions. AA02 and AA03 compare structured data; they do **not** perform accounting-platform integration, OCR or customs validation.

**NOT verified:** production uptime, real customer data or case study, deployment-specific security, data protection compliance, true precision/recall on independently labeled datasets, commercial acceptance, payment collection, or clinical/legal/security/financial decision quality. Tests cover chosen synthetic fixtures only.

Product feedback is opt-in and de-identified with human roadmap triage. No buyer feedback is used for model training or customer-data sharing.

The branch is staged for review, not a completed production deployment. AG Shield scoring code is deliberately absent from this public repository and requires its own dedicated private repository.

## Hardening release — 7 November 2026 (merged PR #2)

**GitHub release:** [Fix PR #2](https://github.com/Obad1/AskDocs-AI/pull/2), merged to `main` as commit `bb630877c12f2c592a9c6bdaa2111c3e9a4c8175`.

### Executed against the patched code

- **66 Python unittest cases passed** locally, including fail-closed missing-link validation across all 16 general adapters, invalid record types, repeated/invalid evidence, opt-out uncertainty, malformed CSV, normalized duplicate IDs, file-size/row/column/cell limits and 499-row reconciliation acceptance.
- **JavaScript engine regression tests passed**, including parser refusal of malformed quoting, normalized IDs, Python fixture parity, and 500 matching pairs.
- **40 additional seeded Python–JavaScript comparison scenarios** passed locally with varied dates, amounts, currencies, blanks and ambiguity, up to 500 combined rows per scenario. These extra exploratory scenarios are independent verification, not part of the committed CI test count.
- **All AA01–AA18 synthetic demonstrations completed** locally with test-only, human-review-required reports.
- Verified exact uploaded Git blob hashes against the tested local code and test files.
- GitHub hosted **[AVAMALE regression](https://github.com/Obad1/AskDocs-AI/actions/runs/42321065635)** and **[AskDocs CI](https://github.com/Obad1/AskDocs-AI/actions/runs/42321065732)** both completed successfully on the exact PR head prior to merging; the AskDocs pipeline now fails for fatal syntax/undefined-name lint problems instead of suppressing them.

### Reproduction commands

From `avamale-workflows/`:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_*.py' -v
node tests/test_engine.cjs
python -m avamale_suite demo --out /tmp/avamale-synthetic-new
```

Choose an empty output directory. All included samples are synthetic. Passing deterministic fixtures does **not** establish production security, compliance with any data-protection law, real-world precision/recall, live integrations, uptime, or actual customer payment.
