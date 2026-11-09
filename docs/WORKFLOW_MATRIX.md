# 18-workflow engineering matrix

**All rows: deterministic local prototype with synthetic fixture and HTML/CSV/XLSX/JSON report. All require final human decisions.** No live buyer integration or performance/accuracy study has been performed.

| ID | Offered review workflow | Current engine capability | Buyer-specific gaps / required sign-off |
|---|---|---|---|
| AA01 | Security questionnaire evidence | Match control ID to uniquely tagged approved evidence metadata | Source verification, approvals, versioning, access controls, human security review |
| AA02 | Bank-to-ledger reconciliation | Unique exact-value/reference candidates; duplicates/ambiguity exceptions | Actual bank/ERP format mappings, split/partial transactions, totals and balance proof, accountant acceptance |
| AA03 | Freight document comparison | Pre-transcribed item-level comparison across invoice, packing list and B/L | PDF/OCR extraction, shipment-level semantics, customer-approved fields, customs specialist acceptance |
| AA04 | AP three-way review | Compare one PO, invoice and receipt line by PO/item | Partial receipts, taxes, payment controls, ERP integration, AP sign-off |
| AA05 | Helpdesk triage | Keyword-based suggested urgency and category | Routing rules, duplicate ticket handling, service desk SLA and safe escalation |
| AA06 | Chargeback packet review | Dispute/evidence metadata completeness | Card-network rules, time limits, authorizations, evidence validation |
| AA07 | Lease date review | Check supplied dates and clause/source metadata | OCR/extraction, amendments, legal/jurisdiction interpretation |
| AA08 | Maintenance dispatch review | Suggest verified-metadata contractor candidates | Live availability, emergency safety response, pricing, approvals |
| AA09 | Subscription recovery review | Flag failed-payment candidates only with consent/opt-out evidence | Actual payment state, consent/legal review, authorized messaging systems |
| AA10 | Cloud-cost anomaly review | Compare baseline and actual against illustrative thresholds | Real billing integration, currency/credits, seasonality, agreed thresholds |
| AA11 | Supplier ESG evidence collection | Detect missing/unverified/expired document metadata | Authenticity, current reporting standards, supplier authorization |
| AA12 | KYC/AML alert evidence review | Synthetic-only evidence-presence suggestions | Regulator-specific obligations, verified identity, privacy and compliance oversight |
| AA13 | Data-subject request preparation | Synthetic-only identity/date/authority metadata flags | Validated requestor authentication, legal deadlines, secure disclosure controls |
| AA14 | Phishing alert metadata | Synthetic-only basic indicators/metadata suggestions | Isolated analysis, trusted data feeds, SOC process and access controls |
| AA15 | Offboarding checklist review | Completion and approver metadata | Identity and provisioning integration, legal holds, approval workflow |
| AA16 | Healthcare credential review | Synthetic-only expiry/source metadata flags | Primary-source verification, sanctions, healthcare legal review |
| AA17 | EHR intake completeness | Synthetic-only field completeness by case | Consent, patient identity, system mappings, clinical safety and privacy controls |
| AA18 | Prior authorization packet review | Synthetic-only evidence-type presence | Payer-specific rules, clinician oversight, minimum necessary handling |

**Priority sequence**: AA02 (revenue validation) → AA03 (secondary freight buyers) → AA01/AA04 (adjacent pilot when real demand documented); AA05–AA18 are prototypes for discovery only until their respective controls, tests, integrations, and qualified buyers exist.
