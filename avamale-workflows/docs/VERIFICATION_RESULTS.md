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
