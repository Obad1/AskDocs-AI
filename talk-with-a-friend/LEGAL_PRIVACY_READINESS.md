# Talk with a Friend — legal/privacy implementation and launch gates

Drafted 9 November 2026. **Not legal advice, legal approval, audit certification or a guarantee against lawsuits.** These pages are intentionally marked *draft* pending lawyer, privacy and consumer-safeguards review. The word "pledge" is deliberately used rather than a purported sworn oath; this is not an affidavit.

## Relevant Nigerian legal framework (review sources)

- Nigeria Data Protection Act 2023 (NDPA): https://nccc.gov.ng/resources/nigeria-data-protection-act-2023
- NDPC General Application and Implementation Directive (GAID) 2025 — FINAL published text: https://ndpc.gov.ng/wp-content/uploads/2025/07/NDP-ACT-GAID-2025-MARCH-20TH.pdf
- Nigeria Federal Competition and Consumer Protection Act 2018, particularly sections 127–129 on unfair terms and conspicuous notice: https://fccpc.gov.ng/wp-content/uploads/2022/07/FCCPA-2018.pdf
- NDPC authority: https://ndpc.gov.ng/
- EU GDPR may have extraterritorial reach where relevant; scope needs fact-specific review, not an automatic assumption.

## Actual implemented controls on feature branch

| Safeguard | Code location | What it does | Limitation |
|---|---|---|---|
| Readable legal documents | legal.py, GET /legal/terms, /privacy, /pledge, /safety, /advice | Five explicit readable notices with a version and date, plus transparent draft banner | No counsel review, formal legal identity, address or privacy inbox established |
| Current policy version | legal.py, GET /api/legal/version | Client retrieves exact current version from server | Versions need formal approval/change process |
| Anonymous chat clickwrap | index.html/app.js and POST /api/sessions | Separate adult, terms, privacy acknowledgement, pledge and safety acknowledgements; backend refuses missing, false or stale version | Self-reported age, not identity-verified |
| Anonymous receipt | app.py Stranger.legal_version / terms_accepted_at | Evidence of version/acceptance time in volatile process memory | **Ephemeral**: lost on free-tier restart, cannot reliably prove historic consent |
| Paid advice clickwrap | index.html/app.js and POST /api/advice/bookings | Separate paid-advice acknowledgment; server validates current version and all flags | Paid advice off and payment sandbox not independently verified |
| Paid booking consent record | advice.py ensure_schema and INSERT | Version + timestamps in persistent PostgreSQL for terms, privacy, pledge, safety and advice service terms | No live configured database; migration untested against real Postgres |
| Paid launch blockers | advice.py enabled() | Requires approval marker, privacy contact email and published refund policy as well as real payment/db/host settings | Environment flags do not constitute legal compliance and must not be enabled before review |
| No broad rights waiver | legal.py | Preserves consumer/privacy rights, rejects blanket negligence/fraud/privacy exclusions | Enforceability is jurisdiction-specific; lawyer approval required |

## NDPA/GAID operational gaps — non-negotiable before broad release

1. **Real legal entity**: publish actual controller legal name, business address, country, designated privacy contact email and process for privacy requests. The current draft explicitly says these are pending.
2. **Data inventory and records of processing**: document sessions, IP usage/host logs, text relay, ephemeral report metadata, optional booking email/topic/access hash, PostgreSQL and processors. Map collection purposes, legal bases, lawful disclosure and any marketing consent separately from terms.
3. **Retention schedule and rights request workflow**: publish concrete retention periods for each system (including Render/platform access logs and paid records), rights verification/deletion procedures, realistic access responses, exceptions under law, and a dedicated request inbox with operational ownership.
4. **Security, processors and transfers**: document Render Germany hosting, customer-to-stranger disclosures, Paystack and selected DB processor roles, hosting agreements, international transfer assessments/safeguards, TLS key storage and access controls.
5. **DPIA/registration**: evaluate whether a data protection impact assessment, NDPC registration as data controller/processor of major importance, DPO appointment or audit returns are required by the project's scale and risk. Seek Nigerian specialist counsel.
6. **Breach response**: maintain an incident playbook, on-call owner, risk analysis, secure audit logs and notification decision tree. Under NDPA section 40(2), certain personal data breaches likely to risk individuals' rights require NDPC notification within 72 hours; high-risk subjects may need immediate notice. An indemnity cannot replace this duty.
7. **Age safety**: the 18+ tickbox is not age verification. Introduce proportionate age assurance, usable abuse reports, staff triage, safety escalation and moderation before marketing a public random-stranger product.
8. **Paid consumer protections**: publish actual delivery schedule and operator, fair cancellation, non-performance, refunds/rescheduling, disclosure of any payment fees, professional scope, complaints contact and consumer rights. The FCCPA limits one-sided waivers and certain liability exclusions.
9. **Advertising**: no "100% safe", "fully anonymous", "GDPR compliant", "certified", "licensed therapist", "waiver of all lawsuits", or fabricated claims. Paid sessions remain nonclinical.
10. **Counsel approval gate**: have qualified Nigerian legal counsel and privacy specialist review final copy, risks and operational practices before any assertion of formal compliance or enabling paid sessions.

## Known implementation limitations

- Text is briefly visible to the server to relay, and the recipient can copy or screenshot. **Not end-to-end encrypted**.
- Free application state and reports are volatile; service-provider system logs may retain technical metadata separately.
- No established staff abuse-review queue, age verification, dedicated privacy inbox, audit log, privacy deletion endpoint or durable consent receipt for free strangers.
- Anonymous checkbox acceptance does not itself establish freely given GDPR/NDPA consent for unrelated processing. The intended basis for necessary matching is contractual/precontractual delivery; legitimate security interests may cover limited abuse controls subject to assessment. Marketing would need separate lawful basis.
- The legal document version contains "-draft-" and must be replaced by counsel-approved text/version before paid launch.
- Even a legally enforceable agreement cannot eliminate the risk of someone filing a lawsuit, or replace statutory duties, reasonable care, consumer protections and actual security measures.
