# AVAMALE Workflow Review Suite — AA01–AA18

**Status: engineering prototypes and synthetic demonstrations. NOT production-certified.**

A single, local-first framework for 18 narrow operational review workflows. The priority commercial prototypes are **AA02: bank-to-ledger exception review** and **AA03: freight-document discrepancy review**; the other 16 are **rule-bound structured-record review foundations**, not complete vertical SaaS products. No workflow autonomously makes financial, legal, medical, security or compliance decisions.

## What is implemented

- AA02: deterministic 1:1 candidate matching on exact amount, currency and normalized reference, within a configurable 0–7 day window. Duplicates, missing references, unmatched records and ambiguity remain flagged. Source rows traceable; no ledger postings.
- AA03: explicit comparisons of manually transcribed item-level invoice, packing list and Bill of Lading fields. Detect missing document types, duplicates and inconsistencies; does not verify source documents or perform customs checks.
- AA01 and AA04–AA18: a distinct adapter for each recorded workflow, documented CSV schema and synthetic fixture. Flags candidate evidence or inconsistencies for human validation. Some are deliberately *metadata-only*, not document extraction/AI or enterprise integrations.
- Offline CSV / values-only XLSX inputs and HTML / CSV / XLSX / JSON plus reproducibility manifest outputs; no networking within the review engines.
- Local opt-in product-feedback registry for **approved, de-identified summaries only**; consent required, deduplication and prioritization summaries; ability to delete. No automatic roadmap changes or training on private comments.
- A standalone browser review lab for AA02 and AA03 using synthetic input, plus a static catalog of all 18 report previews.

## Requirements

Python 3.10+, `openpyxl` (see `requirements.txt`). Node.js 18+ is optional for cross-engine regression tests. No cloud/API account is necessary for the local CLI.

```bash
python -m pip install -r requirements.txt
python -m avamale_suite list
python -m avamale_suite demo --out ./demo_outputs_new
python -m unittest discover -s tests -p 'test_*.py' -v
node tests/test_engine.cjs
```

Do not run `demo` into an existing output directory: reports intentionally refuse to overwrite previous runs. If demo outputs are included in a download bundle, treat them as frozen **synthetic** examples. Generate fresh reports into a new directory. The GitHub source upload may omit generated reports.

### Run AA02

```bash
python -m avamale_suite run AA02 \
  --bank samples/bank_synthetic.csv --ledger samples/ledger_synthetic.csv \
  --date-tolerance 1 --synthetic --out ./review_aa02_new
```

### Run AA03

```bash
python -m avamale_suite run AA03 \
  --data samples/freight_synthetic.csv --synthetic --out ./review_aa03_new
```

### Run any of the other 16 workflows

```bash
python -m avamale_suite run AA04 --data samples/AA04.csv --synthetic --out ./review_aa04_new
```

**Sensitive workflows AA12, AA13, AA14, AA16, AA17 and AA18 require `--synthetic`** until privacy, regulatory and operational safeguards are separately assessed. The command flag is an operator declaration; it cannot independently detect whether input is synthetic. Never use confidential or personal client data without signed agreements, verified authorization and a security-approved environment.

## Incorporating customer feedback safely

Only capture **consented, approved, de-identified summaries**, never raw emails/quotes or customer files:

```bash
python -m avamale_suite feedback add --file ./private_data/feedback.jsonl \
  --workflow AA02 --category feature_request \
  --summary 'Reviewers need an editable exception-resolution status column.' \
  --impact high --source-type consented_interview \
  --source-hint interview_a1 --consent-confirmed
python -m avamale_suite feedback summary --file ./private_data/feedback.jsonl
python -m avamale_suite feedback review --file ./private_data/feedback.jsonl --id RECORD_ID --status triaged --reviewed
python -m avamale_suite feedback delete --file ./private_data/feedback.jsonl --id RECORD_ID
```

The registry offers a minimal screening filter, **not** guaranteed de-identification, permission to use customer information, or a compliant data-processing environment. Protect the folder at the OS level. Do not commit any feedback registry or raw customer comments to GitHub.

## User interface and reports

Open `Review_Lab_AA02_AA03.html` in a browser for the offline AA02/AA03 demo. Open `catalog.html` for the collection of synthetic report previews. The reports can contain whatever the operator supplied and must be handled as sensitive files when real data is permitted.

## Commercial readiness

See `docs/READINESS_AND_GATES.md` and `docs/WORKFLOW_MATRIX.md` for the difference between **code built**, **synthetic test passed**, **customer-validated**, and **production-certified**. In particular, no 18 product deployments or paying customers are asserted.

## Security and limits

Read `SECURITY.md` before connecting to a customer system. This suite does not include authentication, encryption-at-rest, organization tenancy, audited retention/deletion, role-based permissions, live integrations, OCR, OCR review, SOC 2 certification, NDPA legal certification, financial statements, clinical authorizations or automated production decisions.

## Publishing to GitHub

Target owner: `Obad1`. The repository installation is now accessible. For the initial publication, keep this package separate under `avamale-workflows/` on the `Obad1/AskDocs-AI` **feature branch**; never overwrite existing AskDocs sources or alter `DamilolaDami/Luno`. See `docs/GITHUB_ACCESS.md` for the rollout approach. Later move it to a dedicated private repository for commercial development.

## Additional adjacent product experiments (public subset)

Three local, synthetic-only helpers in `avamale_suite/addons.py` are available: `BIZ_ASSIST` (manual buyer qualification), `ASKDOCS` (source-labeled lexical passage retrieval), and `INTAKE` (manual enquiry review). No customer records or automated customer communications are included.

```bash
python -m avamale_suite addon list
python -m avamale_suite addon run BIZ_ASSIST --data samples/addons/biz_assist.csv --synthetic --out ./my_biz_assist_review.json
```

**Confidential source withheld:** AG Shield's rule-scoring sandbox and synthetic risk scenario data are intentionally excluded from this public branch. Do not add them here. A dedicated **private** `Obad1` GitHub repository is required before publishing that module.

## Product readiness

AA02 and AA03 are the only comparatively mature engines; neither is independently validated on customer data. The remaining AA01-AA18 prototypes are review-assistance adapters, not full integrations or production SaaS deployments. No production or sale claims are made. Read `docs/REVENUE_FIRST_VALIDATION.md` for acceptance gates.
