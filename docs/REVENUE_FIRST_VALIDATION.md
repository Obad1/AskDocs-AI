# Revenue-first engineering & commercial acceptance — November 2026

This is a synthetic demonstration package, not a production certification.

## 1. AA02 — preferred first paid pilot

- Buyer: finance controller or managing partner with recurring **residual exception-review work** after QuickBooks/Zoho/Xero/native reconciliation.
- Inputs: authorized values-only CSV/XLSX exports with exact fields `id,date,amount,currency,reference`. No client records required for initial evaluation.
- Engine: unique one-to-one exact amount, currency and normalized nonempty reference within configured 0–7-day window; duplicates, ambiguity, missing reference and unmatched bank/ledger lines are held for a reviewer. No auto-posting, split payments, fuzzy matching, or bank-feed connection.
- Customer decision: does a specific manual review task exist? Confirm transaction volume, reviewer-hour baseline and acceptance examples; compare time/accuracy on agreed synthetic test cases. Do not promise cash savings absent evidence.
- Commercial scope: the **NGN 150,000** Peacebrooks *introductory offer was sent but not accepted*. One entity, one account, one statement period and up to 500 total rows. Price must cover actual engineer/reviewer time, revisions and secure environment. Other buyers are not quoted at that number automatically.
- Go-live gate: explicit SOW, review roles, NDA/DPA before any permitted client data, access/security/retention plan, customer sample approval, signed acceptance criteria, refund/change order policy.

## 2. AA03 — secondary commercial lane

- Inputs: **pre-transcribed** item-level invoice/packing-list/bill-of-lading rows; not raw shipping PDFs. Compare fields where equivalent units and definitions are explicitly agreed.
- Engine: discrepancy list with shipment/item/document source-row evidence; missing or duplicated document types remain held.
- Commercial gate: buyer shows repetitive document discrepancies, examples with approved dummy data, review time, acceptable false-positive rates; no promise of customs compliance, OCR or reduced port delays.

## 3. Confidential risk-scoring module (not published)

Risk-scoring source, weighting details, examples and validation artifacts are withheld from this public repository. A separate private repo must be established before code publication. No claim of fraud-detection accuracy or live bank integration is made.

## 4. Evidence & test command

```bash
python -m unittest discover -s tests -p 'test_*.py' -v
node tests/test_engine.cjs
```

The tests cover deterministic synthetic scenarios, not independently verified accuracy on customer data. Record exact test versions/commit SHAs and human acceptance in a signed engagement record.

## 5. Feedback into product growth

Only use permitted, appropriately minimized and de-identified summaries from buyer conversations. Link each request to product, observed problem, evidence, affected workflow, potential buyer value, and a human go/no-go decision. Do not automatically train on customer comments or publish raw replies. Prioritize objections blocking a paid pilot ahead of feature counts.
