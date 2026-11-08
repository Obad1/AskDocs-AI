# AA03 — Paid pilot proposal and acceptance template
**Status:** INTERNAL TEMPLATE / NOT AN ACCEPTED CLIENT OFFER OR SIGNED CONTRACT
**Version:** 8 November 2026

## The offer
**Name:** Source-linked freight-document exception review, human approved.

**AVAMALE is a business technology company — not a shipping line, freight forwarder, customs agent, haulier or exporter.** We are not offering freight quotes, carriage, cargo clearance, document authenticity verification or regulatory advice.

### Who this pilot is for
A documentation team that already checks Bills of Lading (BOL), invoices, packing lists and shipping instructions, yet repeatedly loses time investigating **specific residual mismatches** not addressed by its existing processes. If a competent staff member or an existing tool already catches every important discrepancy cheaply, we should not sell the pilot.

### Discovery must precede price
Before sending any binding proposal, document the last actual expensive mismatch, its field(s), number of cases per month, hours per exception, cost of corrections, currently used systems, accountable decision-maker, and staff reviewer. Do not ask for identifiable shipment records during discovery.

## Pilot scope — propose, then mutually finalize
**Pilot objective:** Determine if an exception queue helps one buyer's human reviewer locate inconsistent **already structured** shipping-document fields faster and more reliably than their existing process.

- One agreed document family, such as invoice + packing list + BOL for a single shipment type.
- One agreed schema (shipment reference, item identifier, document type, quantity, packages, gross weight and unit, consignee, origin, source record identifier).
- **Synthetic CSV/values-only spreadsheet input** only in the default evaluation. No customer files.
- A bounded set of **[N] fictional shipment cases**, agreed before work. Suggested initial laboratory size: 20–40 cases, not a delivery promise.
- Rules-based structured field differences, missing/duplicate input, reference to source record/row, and explicit ambiguity/needs-human-review flags.
- Human reviewer labels each exception; deliver a review summary plus a source-evidence report and export that can be inspected.
- One rounds-of-feedback session and up to **[number]** revisions, agreed before quoting.
- No autonomous changes to shipping documents, customs filings, shipment release, payments, declarations or ERP systems.

## Out of scope without a separate tested and signed change order
- OCR or extraction from raw PDFs/images, handwriting recognition, scans or email attachments.
- Authenticity or fraud adjudication, insurance/legal/customs advice or regulatory approval.
- Integrations to customs or carrier portals, live freight tracking, shipment bookings, payment processors or production customer systems.
- Multi-tenant hosting, file sharing, retention policy guarantees, encryption claims, automatic deletion or access-controlled customer data uploads.
- Any claim of savings/ROI/accuracy from a hypothetical case.

## Acceptance test worksheet — fictional only
| Fixture | Expected outcome | Human reviewer check |
| --- | --- | --- |
| Same fields across all three source rows | `CONSISTENT_FIELDS_REVIEW_REQUIRED` | Not a claim of authenticity/compliance |
| Invoice quantity 100; packing list + BOL quantity 98 | `FIELD_DISCREPANCY` | Resolve meaning and units manually |
| BOL absent | `MISSING_DOCUMENT` | Identify missing source, don't invent it |
| Two invoices for one shipment item | `DUPLICATE_DOCUMENT_ITEM` | Determine amendment/duplicate manually |
| Consignee differs by entity string | `FIELD_DISCREPANCY` | Check source authority |
| Two equal-looking fields with different units/currencies | `NEEDS_HUMAN_REVIEW` unless explicit normalization rule agreed | Never silently coerce incomparable units |

**Acceptance:** Buyer and AVAMALE sign an agreed fixture manifest; reviewer matches expected exceptions, identifies unresolved cases and grades report usefulness against the buyer's current workflow. Compare **minutes per case and missed-issue counts** only after a properly designed side-by-side test.

## Commercial terms to fill after qualification
- **Client and decision-maker:** [legal business name, authorized signatory]
- **Pilot fee:** ₦[scoped fee] — not agreed until discovery and written acceptance
- **Payment schedule:** [e.g. advance at signing + balance at mutually accepted milestone], to be negotiated; no guarantee or request for payment until agreement
- **Duration and milestones:** [start, review, delivery dates, buyer input dependencies]
- **Support and revision cap:** [business hours, response expectation, authorized communication channel]
- **Cancellation and refund boundaries:** [agreed written terms]
- **Confidentiality, permitted data, processor role and retention:** [NDA/DPA/data schedule if applicable]
- **Delivery jurisdiction, liability limits and governing law:** [legal review]
- **Final report ownership and IP license:** [legal review]

No real shipment files, third-party personal data or production access are accepted until security controls, purpose, retention, access, deletion testing and contractual permissions are separately approved. A synthetic proof phase is not evidence those release gates have passed.

## Buyer discovery call agenda (15–30 minutes, invitation only)
1. Walk through the last incident **without** sharing private client data.
2. Quantify monthly frequency, minutes lost, amendment cost and owner.
3. Review current tools and their exact residual gap.
4. Show the [fictional eight-case sample](https://headless-tnhgpdqwfaa-avamalemoses-140c.wix-site-host.com/aa03-review.html).
5. Agree what a successful small fictional evaluation would prove or disprove.
6. Ask who authorizes spending and whether they want a written scoped proposal.

## Reply decision matrix
| Signal | Action |
| --- | --- |
| Acknowledgment only, asks for freight quote | Clarify technology scope; do not send invoice or shipping quote |
| Says current tool handles it | Thank them; close as not-fit, no repeated follow-up |
| Gives a last problem and monthly recurrence | Invite a short discovery call; use only generalized examples |
| Requests real-data processing immediately | Decline intake until signed security and data review |
| Requests pricing but no measurable problem | Ask a few qualifying questions; quote a **bounded range only if costs can be estimated** |
| States explicit decline or opt-out | Suppress future contacts |

## Release and evidence checklist
- [x] Synthetic structured-data demonstration exists in AA03 repository.
- [x] Wix buyer page published, source and links read back.
- [x] Dedicated AA03 six-field hosted form created and API smoke-tested with fictional test; test deleted.
- [ ] External real-browser visual and visitor submission confirmation.
- [ ] Qualified buyer with an actual unmet recurring cost.
- [ ] Written acceptance criteria validated by intended human reviewer.
- [ ] Proposed fee and milestones accepted by authorized client.
- [ ] Signed terms, payment evidence, security/legal approvals where applicable.

**DO NOT claim we already have paying customers, verified conversion, customs integrations, or a production-safe document intake.**
