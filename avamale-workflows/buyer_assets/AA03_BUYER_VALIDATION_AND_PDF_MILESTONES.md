# AA03 — First Buyer, Measured Proof and Paid Pilot: Execution Plan

**Date:** 8 November 2026  
**Status:** Internal execution plan. Not a production launch, proven customer case, or signed commercial agreement.  
**Primary offer:** AA03 freight-document reviewer. AA01 and AA04 are smaller comparison lanes; AA02 is conditional and Peacebrooks declined.

## Decision

**Do not wait for automatic PDF extraction before speaking to buyers.** Run a *concierge-assisted document review* first to prove what causes costly rework, while engineering develops an offline PDF-to-structured-fields workflow with explicit human review. Sell the output/decision support, not the number of automation features. No real customer data may be uploaded to the Wix form or existing synthetic prototype.

## Workstream A — Find a genuine buyer

### Who
Documentation manager, customs-operations head, freight forwarding operations lead, or importer logistics controller who repeatedly cross-checks invoice, packing list and Bill of Lading/shipping instruction. Prefer teams with meaningful repeated exception volume over an organization that handles only occasional shipments.

### 5-question discovery interview
1. Walk me through the **last specific** document mismatch that forced additional checks or an amendment. No client identity or private shipment numbers.
2. What systems and people were involved? Which check was already automated, and what remained manual?
3. About how many such cases occur monthly, and how many minutes does the staff member spend per case? If a monetary delay or correction fee occurred, how was it calculated?
4. Which documents and fields cause the biggest problem—consignee, goods description, units, packages, gross weight, reference or origin?
5. Who owns the workflow and would authorize a small paid process improvement? What evidence would convince that person it helps?

### Outreach strategy
- The existing personalized AA03 campaign and Wix form are already live. Do not resend recent contacts until reply/bounce/opt-out eligibility is checked.
- Seek five **warm introductions** from freight, business and procurement networks; ask for *feedback on a workflow*, not a product purchase.
- Send a short question first, then share the synthetic example after interest. Never call a sent email a delivered, opened, qualified or paid lead.
- Qualify an opportunity only when the operator names a recurring problem, current alternative, expected business impact, authorized next-step owner and agreed interest in a test.

**Early discovery target (not a promised outcome):** 5 real operator conversations, at least 2 detailed repeatable pain cases, and one decision-maker willing to consider a scoped evaluation. If none can identify expensive residual work, stop promoting AA03 indiscriminately and revisit the offer.

## Workstream B — Demonstrate an actual before/after

### What to show
Use 20 *fictional cases based on the problem categories operators describe*, without copying their confidential details. Clearly label as buyer-informed synthetic scenarios, not verified customer outcomes. Include:
- 5 consistent cases
- 5 field mismatches (quantity, units, weight, packages, consignee)
- 3 missing document-type cases
- 3 duplicate/amended version cases
- 4 ambiguous or mixed-source cases requiring a reviewer, including a false-match trap.

### Run the measurement
Assign a document reviewer to process the same difficulty mix under two conditions (counterbalance case groups to reduce learning bias):
1. **Current process:** reviewer manually compares documents and records corrections and evidence.
2. **AVAMALE-assisted:** a human transcribes/validates source values and uses AA03's structured engine to review exceptions.

Record **all time**: document opening, transcription, checking, correction, review and report export. Log detection quality as true positives, false positives, missed critical differences and reviewer overrides.

Use this practical benchmark:
- **Manual total minutes** = document prep + comparison + confirmation
- **Assisted total minutes** = extraction/transcription + human verification + exception review
- **Time reduction** = (manual minus assisted) / manual
- **Net monthly staff hours saved** = max(0, (manual minutes per case minus assisted minutes per case) × cases per month ÷ 60)
- **Potential monthly value** = verified hours saved × buyer's actual loaded cost/hour + documented avoided rework costs; never insert guesses into an offer as fact.

A demo that makes the table look impressive but requires more staff time is **not a success**.

### Go/no-go (proposed internal gates; not external claims)
- No unreviewed high-risk mismatches silently presented as resolved.
- Operator confirms the exception report is actionable and source-verifiable.
- At least two independent buyer conversations suggest a meaningful unresolved need.
- One authorized sponsor agrees to acceptance criteria and a next step.
- Benchmarks show enough net workflow improvement to justify the economics of the proposed fee. A 30% total-time improvement is a provisional *ambition*, not a guaranteed outcome or a pass substituted for quality.

## Workstream C — Close the PDF intake gap (free/open-source, local-first)

**Current reality:** AA03 already accepts structured item-level records via CSV or values-only XLSX. It has rules and synthetic tests. It **does not** process raw PDFs or scans. The Wix form is not a secure file-upload system.

### Candidate pipeline
```
fictional invoice / packing-list / BOL PDFs
   ↓ digital text? (pdfplumber or pypdf)
   ↓ scanned page? optional local Tesseract OCR
   ↓ candidate fields + page/region provenance + low-confidence markers
   ↓ mandatory human verify/correct stage
   ↓ normalized synthetic AA03 rows (same schema already in repo)
   ↓ existing discrepancy / missing / duplicate engine
   ↓ reviewer report and measured acceptance test
```

Field schema from existing `freight_synthetic.csv`:
`id,shipment_id,document_type,document_ref,item_code,quantity,packages,gross_weight_kg,consignee,origin_country`.

### Technical delivery gates
- **M1:** parse machine-generated fictional PDF text and extract candidate header/item fields with page provenance, rejecting missing/ambiguous fields instead of filling them.
- **M2:** handle visually complex PDF tables, multi-page records and units; detect duplicate source records; human confirms each candidate.
- **M3:** handle scanned fictional PDFs with opt-in Tesseract OCR, mark uncertain text, preserve visual source snippets/page IDs, never auto-authorize field values.
- **M4:** integrate verified fields with existing AA03 engine, run 20-case fixtures and record extraction quality + total process time.
- **M5:** independent technical/security review before any real documents. Avoid production claims until auth, access, retention/deletion, isolation and contracts are tested.

Text extraction is not OCR; scanned images need a different path and mistakes require human review. Open-source dependencies are subject to license and security review before use in commercial deployment.

## Workstream D — Convert an interested operator to a paid pilot

### Customer journey
1. Personalized note or warm introduction.
2. Short discovery interview; identify one recurring measurable workflow.
3. Show 1–2 fictional examples shaped by **their actual described problem** (without their data).
4. Agree acceptance metrics and operator, verify budget owner.
5. Issue bounded scope-of-work and mutually agreed price/timeline. The current public **₦200,000 for up to 20 fictional cases / 10 business days** is an **introductory proposal**, not an accepted rate or guaranteed result.
6. If a buyer signs and follows agreed payment terms, perform the synthetic proof and log true measured outcomes; expand only with appropriate privacy/security controls.

Do not charge for a generic screenshot; the customer must understand what written review/report they receive. If the buyer wants automatic raw PDF extraction, explicitly separate that from the current structured comparison and provide a revised scope once M1–M4 pass.

### Draft discovery reply
> Thank you for describing the documentation issue. AVAMALE develops reviewer-led technology, not freight or customs services. To determine whether we're useful, could we review the last category of mismatch (no client details), your current checking process and how often it causes rework? I'll show a fictional example with the same pattern and we can see whether a bounded paid pilot makes business sense.

### Payment and scope checklist
- Name of legal company and authorized buyer; exact deliverable; supported inputs; no live documents by default.
- Cost estimate: staff transcription time + reviewer time + engineering + revisions + contingency + margin.
- Price, schedule and acceptance written into SOW, payment method and collection evidence, data/use terms and cancellation details.
- **No counted revenue until actual cleared funds are independently verified.**

## Workstream E — Website and conversion verification

Current Wix site and AA03 form are linked and the API smoke test on synthetic data passed; the test record was deleted. No end-to-end visitor-browser confirmation and no qualified real submission have been verified. Before soliciting sensitive files:
- Test mobile and desktop page rendering, sample filtering, accessibility and separate primary/secondary CTAs.
- In a real browser, submit a unique *noncustomer* test identity, confirm visible success, verify Wix entry and notifications, then delete and document cleanup.
- Ensure offer includes realistic price/time/deliverable, and differentiates fictional-data demonstration from any processing of live PDF attachments.
- Log traffic acquisition channel, inquiry, reply, meeting, scope issued, signed pilot and verified payment separately.

## Day-by-day proposed execution (not work scheduled or completed)

| Day | Sales / buyer evidence | Engineering / proof |
| --- | --- | --- |
| 1 | Review replies, declines, warm-intro sources; identify 10 high-fit operators | Confirm current fixture baseline and measurement sheet |
| 2 | Ask 5 warm introductions and arrange short calls | Prepare 3 machine-generated fictional source PDFs with known truth |
| 3 | First 1–2 operator interviews | Implement text-first PDF field candidate extraction (M1) |
| 4 | Map actual residual pain and current tools | Human review/correction UI or simple CSV review gate |
| 5 | More interviews; reject poor-fit buyers | M2 table/unit ambiguity tests |
| 6 | Show buyer-informed fictional before/after | Start optional scanned-PDF Tesseract fixture work (M3) |
| 7 | Score one buyer's budget/approval path | Measure manual versus assisted total time on separate cases |
| 8 | Present results and seek acceptance criteria | Integrate corrected fields to existing AA03 and test |
| 9 | Send a written scoped proposal only if qualified | Final report QA, known-limitations and security review |
| 10 | Obtain decision; record signed/sent/paid status separately | Package synthetic demonstration and audit test artifacts |

**Stop rule:** If there is no demonstrated buyer pain or the human-in-the-loop method is slower than current workflows, revise or shelve AA03 rather than claim success from outreach volume.

## Immediate operational scoreboard
- Unique qualified contacts researched
- Actual outbound actions with Gmail IDs (Rule of 100, not a promise)
- Meaningful replies and opt-outs
- Operator interviews / named last incidents / baseline hours
- Interested budget holders
- Synthetic benchmark pass/fail; measured manual and assisted time
- Pilot scopes requested, sent, accepted
- Signed pilots and **verified paid receipts**

### References within this repo
- `avamale-workflows/samples/freight_synthetic.csv`
- `avamale-workflows/avamale_suite/__main__.py`
- `avamale-workflows/tests/test_review.py`
- `avamale-workflows/buyer_assets/AA03_PAID_PILOT_SOW_TEMPLATE.md`
- `avamale-workflows/docs/READINESS_AND_GATES.md`

**No prospective customer is implied to have agreed to, received or paid for any pilot by this plan.**
