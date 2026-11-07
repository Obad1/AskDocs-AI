# AA02 buyer demo and first paid-pilot acceptance pack

**Status:** proposed and ready for buyer qualification using wholly synthetic data. Nothing here is evidence of a signed contract, accepted price, production certification, or verified customer ROI.

## What buyer receives in the demonstration

- **Inputs:** two authorized, structured files representing a bank export and ledger export with columns `id,date,amount,currency,reference`. The sample is synthetic; values-only CSV/XLSX, not unprocessed bank PDFs.
- **Processing:** normalized reference, exact signed amount and currency, configurable date difference from 0–7 days. Only unique one-to-one candidates are suggested. Missing references, duplicate candidates, currency mismatches, amount mismatches and unmatched lines require human review.
- **Outputs:** source-row-linked CSV, XLSX, HTML, JSON review artifacts and reproducibility manifest. Nothing is posted into QuickBooks, Xero, Zoho, or any other accounting system.
- **Limitations:** no split-payment matching, fuzzy categorization, direct bank feeds, accounting integrations, verified storage controls, financial statement certification or independently validated accuracy.

## Run a no-client-data live walkthrough

From `avamale-workflows/`:

```sh
python -m pip install -r requirements.txt
python -m unittest discover -s tests -p 'test_*.py' -q
python -m avamale_suite run AA02 --bank samples/bank_synthetic.csv --ledger samples/ledger_synthetic.csv --date-tolerance 1 --synthetic --out ./new-buyer-demo-AA02
```

The generated output folder must not exist before the run. Open `new-buyer-demo-AA02/review.html`; compare sample rows to the output and walk the reviewer through statuses. The interactive offline HTML demonstration `Review_Lab_AA02_AA03.html` is also available but is a *demo* not a secured hosted customer portal.

### Synthetic reference outcome

Using the bundled examples at date tolerance 1 day, the observed result counts were:

| Output status | Count |
| --- | ---: |
| SUGGESTED_MATCH | 2 |
| UNMATCHED_BANK | 3 |
| DUPLICATE_OR_AMBIGUOUS | 3 |
| MISSING_REFERENCE | 1 |
| DUPLICATE_OR_AMBIGUOUS_LEDGER | 3 |
| MISSING_REFERENCE_LEDGER | 1 |
| UNMATCHED_LEDGER | 3 |

A suggested match is **never** a financial posting or completed reconciliation.

## Peacebrooks discovery questions

The first follow-up was **already sent** in Gmail thread `1a1a89a5f6cd14c6`, with a proposed ₦150,000 introductory pilot. **Do not send a duplicate.** No buyer acceptance is recorded.

When Peacebrooks next replies, determine:

1. Which specific preparation task remains manual *after* their existing accounting tools, and its frequency?
2. Can the relevant staff demonstrate the expected exception decision using **fully invented dummy data** without revealing actual bank/client records?
3. Which record fields, currencies, date conventions, duplicate semantics and source references are non-negotiable?
4. What outcome would justify paying (reviewer minutes, number of errors caught, or turnaround)—and who is authorized to agree?
5. Which confidentiality and data-processing agreement templates and approval route do they use?

## Indicative scope — not an accepted quotation

| Component | Included in proposed introductory pilot |
| --- | --- |
| Fee | ₦150,000, **proposal sent, not accepted** |
| Organization | One legal entity |
| Account | One bank account and one ledger export |
| Period | One statement period |
| Size | Up to 500 total bank and ledger rows combined |
| Rules | One mutually agreed deterministic rule set |
| Outputs | One exception-review workbook/CSV and demonstration |
| Iterations | One feedback revision |
| Not included | System integration, OCR/PDF extraction, ongoing hosting, ongoing support, actual account reconciliation sign-off, regulatory certification |

Before accepting any order, estimate *actual* engineer hours, delivery/review time, quality checks, security overhead and revisions; reserve a margin. If the estimate does not support ₦150,000, propose a revised **written** statement of work, do not absorb unpriced scope. For new buyers, quote based on problem complexity and validated value; do not automatically anchor everyone at ₦150,000.

## Acceptance gates before billable customer engagement

1. Buyer confirms a real residual task, intended user and budget authority.
2. Engineer and accountant agree a blinded **synthetic** test suite: correct unique matches; duplicate bank/ledger; missing reference; cross-currency; incorrect amount; date-boundary; unsupported split transactions. Define expected output for each.
3. Reviewer validates **all** test cases and signs a short acceptance note. Count false positives and negatives on the agreed dummy fixtures; do not extrapolate to production accuracy.
4. Written scope, price, payment terms, change-order and data-access boundaries agreed. No work on real customer data until signed NDA/DPA **and** independent assessment of access controls, deletion, encryption, hosting and incident response.
5. Capture only consented **de-identified** customer pain/feedback and obtain human approval before adding to the roadmap.

## Sales decision

- **Go:** buyer verifies a specific unmet need, chooses acceptance criteria and accepts a paid pilot in writing.
- **Revise:** their existing software already covers the task; sell a narrower exception-approval or reconciliation QA workflow only if separately evidenced.
- **Stop:** no identifiable task, buyer, budget, authorization or compliant delivery path.

No external buyer should be told this package is production-ready or generates proven savings.
