# AVAMALE flagship four: buyer problem, proof and safe release gates

Updated 2026-11-09. **Internal working hypotheses**, based on the [Product Portfolio sheet](https://docs.google.com/spreadsheets/d/1_aXvTaSHUwiZKJDFP_apsnzTgtr0lRHGENCCqyAQXTg/edit) and [AA03 buyer validation plan](https://docs.google.com/document/d/1xLIfojcJLF1B241VHIIQ3qp7zVuT4r4oe6arafCS4RU/edit). No independently verified customer ROI, signed pilot, accepted prices, or real-data production clearance.

## Which problem is worth selling?

| Offer | Buyer with budget | Real pain hypothesis | Present working solution (synthetic) | Competing alternative / rejection test | First measurable buyer proof |
|---|---|---|---|---|---|
| **AA03 — Freight discrepancy review** (first) | Freight documentation operations manager; importer logistics controller | Repeated mismatches across invoice, packing list and B/L force rekeying, amendment checks, disputes about units/weight, and delayed handoffs | Structured, **pre-transcribed** invoice/packing/BOL comparisons; report with shipment/row evidence, missing-document and mismatch flags | Existing TMS/forwarder tools may cover checks; if no recurrent *residual* rework, do **not** pitch | Count last 30-day rework cases and manual total minutes; test 20 fictional scenarios; track critical detection/false positives, human review time including extraction |
| **AA01 — Security evidence** | Security/GRC and sales engineering teams selling into enterprises | Every buyer questionnaire prompts repeated evidence collection and drafting; unsupported claims create reputational/contractual risk | Map supplied control IDs to *approved* evidence references; leave unsupported answers for human reviewer | Existing GRC/security questionnaire tool; decline if buyer already has frictionless approved responses | Baseline vs assisted reviewer hours for 20 sample questions; percent with explicit evidence citations; zero unsourced attestation released |
| **AA04 — Invoice 3-way exceptions** | AP manager/procurement controller | PO, invoice and receipt may mismatch; partial deliveries and duplicate invoices consume manual reconciliation and delay legitimate approvals | Structured PO/invoice/receipt checks; exception queue and approval-prep report | ERP's three-way match may already work; sell only residual exception workflows | Count exception queues and exception handling hours; measure precision/recall for duplicate/quantity/price flags; every payment stays human-authorized |
| **AA02 — Residual bank reconciliation** (conditional) | Accounting firm owner or finance controller with substantial unresolved items | Existing bank/ERP reconciliation leaves ambiguous or unmatched entries that require multi-source research | Exact structured ledger/bank matches; duplicates and ambiguity isolated; source-row exception workbook | QuickBooks, Xero and Zoho may already solve ordinary matching; **do not** position simple one-to-one matching as novel | Prove >0 genuinely unresolved recurring exceptions; quantify staff minutes per exception and false-match rate; stop campaign if none |

## Translate pain into truthful copy (Hormozi-style specificity, without invented proof)

- **AA03**: "Still fixing invoice, packing-list and shipping-document mismatches by hand? Show us the recurring field errors. We'll demonstrate a reviewer-led discrepancy report on fictional shipment records, with the exact row behind every flag."
- **AA01**: "Does each enterprise security questionnaire send your team hunting for the same control evidence? See a controlled response-evidence map with missing approvals clearly held for review."
- **AA04**: "Your ERP flags invoice exceptions. Who still spends hours working out the reason? We organize three-way source discrepancies into a review-ready report. Nothing is paid or approved automatically."
- **AA02**: "If routine reconciliation is automated but a costly tail of exceptions remains, test a source-row-linked exception report against your current workflow. If the current system already resolves it, we are not a fit."

These are **positioning hypotheses**, not validated testimonials or ROI promises. Never claim 'X% saved' until timing includes extraction/prep, checking, overrides, corrections and report export.

## Product-release and acquisition sequence

1. **Safety floor:** fictional-data-only, single authenticated reviewer, immutable source identifiers, no silent decisions or real-client uploads; validate retention, backup, security, access policy before any real data.
2. **Input honesty:** only structured CSV/values-only XLSX where implemented. In particular AA03 does **not** parse freight PDFs; AA01 does not retrieve approval documents from enterprise systems. Show input schema before upload.
3. **Proof sheet:** 20 fictional cases per workflow: clean, mismatched, missing, duplicate/ambiguous and hard false-positive traps. Capture baseline minutes, assisted minutes (including extraction), true/false positives and reviewer override. Log reviewer names only when consented.
4. **Buyer discovery:** ask about most recent exception, frequency, existing tools, cost of rework and purchase authority. At least two buyers must describe a repeatable residual pain before scaling outbound.
5. **Narrow paid pilot:** buyer-authorized written acceptance metrics, period and case limits, named outputs, privacy terms, quoted cost, change-order handling. A sent proposal is not revenue. Prices in existing sheets remain hypotheses.
6. **Only then** expand ingestion (AA03 source PDF -> human-corrected fields; AA04 partial receipts/taxes; AA01 evidence permissions; AA02 splits/FX) if supported by buyer demand and risk assessment.

## Do not promise yet

No autonomous customs clearance, financial posting, compliance sign-off, live accounting or bank integration, security attestation, extraction of arbitrary PDFs, multi-tenant privacy certification, guaranteed reduction in delays, or processing of real customers' private financial/operational documents on the synthetic pilot.

## Definition of done for the UI

- AA01–AA04 distinguished as flagship *demonstrations*, not four full-fledged production SaaS systems.
- For each, buyer pain, reviewer solution, delivery artifact, evidence-based trial metric, and current constraints are visible before upload.
- Other 14 synthetic workflows remain accessible in a separate "Research and extended workflows" section.
- No unsafe HTML injection; data is rendered as text, not innerHTML.
- Responsive keyboard navigable form, focus indicators, comprehensible error/success states, no unverified ROI statistics.
