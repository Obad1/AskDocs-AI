# Readiness evidence and release gates — 7 November 2026

## What has been certified *by tests*

- A local Python test suite plus a Node.js AA02/AA03 engine parity script passed on bundled **synthetic** data.
- The CLI has eighteen registered workflows, including sixteen per-workflow adapters and two deeper AA02/AA03 engines.
- Each supplied synthetic fixture generates review outputs with source-row identifiers and required human review flags.
- Regulated and high-risk workflows have a synthetic-only CLI gate. A user can falsely assert `--synthetic`; this is not a full data-loss-prevention control.
- The opt-in feedback workflow rejects obvious emails, URLs, phone-like and long account-number-like strings, deduplicates comments, scores aggregate categories and supports deletion. This is not a foolproof PII detector.

## What is **not** certified

No independent security audit, production deployment, SLA, multi-tenant isolation, encrypted secrets, data privacy compliance opinion, NDA/DPA approval, integration with client systems, production monitoring, penetration test, real-data precision/recall study, external clinical/financial/legal approval, or verified customer revenue. No user should infer these from passing synthetic tests.

## Go/no-go requirements for AA02 first pilot

1. **Problem and buyer** — real finance reviewer identifies residual exception-review work and agrees to a bounded test.
2. **Scope** — one entity, statement window, file format and agreed exact matching logic; specify limits including split payments/duplicates/rounding and the review burden.
3. **Samples** — start with fully synthetic dummy files, not vaguely "anonymized" live bank data. Any later personal/financial information requires written authorization, signed NDA/DPA and appropriate security controls.
4. **Technical acceptance** — agree positive, negative, ambiguous, duplicate, missing-reference, cross-currency and date-boundary test cases; demonstrate outputs against blinded expected results.
5. **Reliability and privacy** — isolate execution, use least privilege, log processing without customer contents, restrict access, define deletion and incident response.
6. **Commercial acceptance** — written statement of work, deliverables, fee, payment terms and actual responsible delivery owner.
7. **Follow-up** — collect only approved de-identified buyer feedback and create evidence-linked issues with human approval.

**Decision**: internal prototype readiness for synthetic demonstrations; **NO-GO for unsupported production claims or live client-data processing.**
