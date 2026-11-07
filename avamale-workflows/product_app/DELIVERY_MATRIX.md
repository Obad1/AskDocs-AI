# AA01–AA18: what is implemented and what is not

The original `18 Solutions` sheet specifies 18 separate *commercial offerings*. This shared web platform provides 18 usable review workspaces with unique validation engines and a consistent case lifecycle. **It is not proof that all 18 fully integrated commercial products are complete.**

| Offer | Distinct runnable feature | Still required for full production product |
| --- | --- | --- |
| AA01 Security questionnaires | Approved evidence/control matching, source citations, reviewer decision queue | Actual document parser, permissioned evidence store, questionnaire authoring, approval workflow, enterprise integrations |
| AA02 Reconciliation | Deterministic bank/ledger matching, duplicates and exceptions, workbook export, reviewer decisions | Account balances, partial/split transactions, ERP connectors, secure client onboarding, accountant sign-off |
| AA03 Freight | Item-level invoice/packing/BOL comparison, discrepancies and missing docs | Real shipping PDFs, approved OCR, document version reconciliation, freight-system integration and buyer validation |
| AA04 AP matching | Three-way PO/invoice/receipt checks and exception holds | Partial receipts, taxes, supplier master, accounting integrations, payment approval workflow |
| AA05 Helpdesk | Keyword urgency suggestions, review queue | ITSM integration, verified history/ticket taxonomy, routing SLA and escalation approvals |
| AA06 Chargebacks | Dispute-to-evidence reference checks and packet review | Merchant/processor APIs, evolving rules, time limits, authorized submissions |
| AA07 Lease | Supplied lease metadata and notice-date checks | Secure PDF clause extraction, calendar alerts, legal review, document amendments |
| AA08 Maintenance | Metadata-based contractor suggestions and human dispatch queue | Contractor availability, location, costs, work orders, scheduling and emergency escalation |
| AA09 Recovery | Consent/opt-out check and candidate queue | Billing-provider verification, compliant messaging, deliverability/attribution, opt-out synchronization |
| AA10 FinOps | Illustrative baseline-to-actual anomaly checks | Real billing APIs, cost allocations, seasonality, thresholds, alerting and remediation approvals |
| AA11 Supplier ESG | Evidence presence, marked verification and expiry checks | Supplier submissions, primary validation, supplier portal, standard-specific assurance |
| AA12 KYC/AML | Synthetic alert-to-evidence metadata checklist | Licensed compliance professional, verified risk rules, identity systems, regulator-specific audit |
| AA13 DSAR | Synthetic identity/authority/date review queue | Actual identity proofing, source system search, secure disclosure and deadline tracking |
| AA14 Phishing | Synthetic metadata triage and analyst queue | Secure isolation, threat intel, malicious URL sandbox, SOC response and labeled accuracy evaluation |
| AA15 Offboarding | Staff/resource/approver checklist with review states | HRIS, IAM/MDM integrations, legal hold controls, actual account/license removal verification |
| AA16 Credentials | Synthetic credential expiry/source review | Primary-source verification, sanctions, role restrictions and healthcare regulatory review |
| AA17 EHR intake | Synthetic field presence/deduplication review | EHR connector, clinical field mapping, patient identity, HIPAA/NDPA review and data correction |
| AA18 Prior authorizations | Synthetic minimum packet type and evidence checks | Payer-specific rules, medical necessity validation by clinicians, secure submissions and tracking |

**Existing working platform behavior across all 18:** operator authentication (single user), synthetic CSV/values-only XLSX uploads, validation, saved run/job queue, reviewer statuses, decision audit, CSV/XLSX/JSON output, public product pages, optional consent-gated enquiry capture (disabled by default), API and UI tests.

**Not yet supported across all 18:** multi-tenant accounts, direct payments, billing, signed customer contracts, customer-data import, independently audited data protection, external integrations, automated correspondence, or uptime assurances.

## Commercial priority

AA02 and AA03 have the highest current buyer-facing relevance. AA01 and AA04 are next if buyers confirm use cases. Healthcare and regulated cases should remain isolated synthetic demonstrations until a qualified specialist validates them and proper safeguards exist. Fees in pricing sheets are **proposals, not customer acceptance**.
