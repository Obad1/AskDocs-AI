# AVAMALE AA03 — Buyer-facing synthetic pilot pack (review draft)

**Status:** Synthetic demonstration only. Not a deployed customer SaaS or verified customer case study. Do not use client shipment documents or promise customs clearance, shipment services, document authentication, OCR or regulatory compliance.

## Job-to-be-done
A freight documentation manager must compare information received from multiple parties before handoff. Mismatches in item quantity, packages, weight, consignee or origin, along with duplicate or missing documents, may require manual rework. **The value proposition remains a hypothesis until operators confirm frequency, actual cost and their current tooling gap.**

AVAMALE is a **business technology company**, not a shipping line, freight forwarder, customs broker or cargo customer.

## First-touch buyer offer
"Would your documentation team benefit from a fictional one-page discrepancy example showing source-linked differences across structured Bills of Lading, commercial invoices and packing lists? If existing tools already catch these problems reliably, that's useful feedback too."

Don't attach a large MP4 or claim production readiness. Share the demo selectively after the recipient expresses interest.

## What exists now
- Existing local structured-record engine: `python -m avamale_suite run AA03 --data samples/freight_synthetic.csv --synthetic --out ./review_aa03_new`.
- Inputs: structured CSV or values-only spreadsheet, with manually transcribed document fields.
- Outputs: exception suggestions linked to source rows and human review flags, plus report exports.
- Frozen synthetic example: [AA03 report](../demo_outputs/AA03/review.html).
- Buyer-facing interactive illustrative page: [AA03 Buyer Review](aa03-buyer-review.html).
- Not implemented: original document authentication, OCR/raw PDF verification, Customs filings, automatic corrections, verified customer data processing, real-data privacy controls or multi-tenant deployment.

## Buyer discovery questions (record responses without confidential files)
1. What was the last case of a document mismatch causing rework?
2. Which documents and specific fields did it involve?
3. How many such cases per month and approximately how much staff time does each cost?
4. What tool or checklist is used today, and which work remains manual?
5. Who approves corrections, what proof must be shown, and what would make an exception report useful?
6. Who can authorize a small paid pilot? What budget and timing would be realistic?

## Example synthetic acceptance set
| Case | Expected reviewer finding |
|---|---|
| Quantity difference invoice=100, packing list=98, BOL=98 | FIELD_DISCREPANCY; source-row evidence; human review |
| Bill of Lading not present | MISSING_DOCUMENT; human review |
| Two invoice records for item | DUPLICATE_DOCUMENT_ITEM; human review |
| All compared fields equal | CONSISTENT_FIELDS_REVIEW_REQUIRED; not an authenticity or clearance pass |

## Pilot structure (proposed, unquoted)
A responsible operator and AVAMALE agree on one document family, a known synthetic test set, exact fields, acceptance checks, reviewer decisions and a bounded revision. Quote **only after** the buyer confirms expensive residual work; calculate engineering cost, security overhead and margin. Current historic ₦200k diagnostic / ₦750k pilot figures are hypotheses, not accepted rates. **No live client data** without signed documents, authorization and independently validated security.

## Decision rules
- **Proceed:** named buyer, recurring unmet problem beyond existing tools, accepted fictional-test criteria and authorized budget owner.
- **Narrow scope:** many exceptions already caught by existing ERP/forwarder system; only preserve a demonstrable residual use case.
- **Stop:** no recurring rework, no differentiated value, no owner, or production processing required before security readiness.

## Conversion and follow-up
- After a buyer requests the example, share the fictional review page (when hosted and checked) and offer a short conversation. Do not imply the GitHub raw HTML is a live hosted product.
- Use an email reply CTA until a verified live lead form is connected and tested.
- After a first email, wait 3–5 business days, check Gmail for replies/bounces/opt-outs and consider at most one thoughtful, relevant follow-up. No batch follow-up to everyone regardless of engagement.
- Track stages: SENT ≠ DELIVERED ≠ REPLIED ≠ QUALIFIED ≠ DEMO ≠ PROPOSED ≠ PAID.
- Post and ad content remain in **review only**.
