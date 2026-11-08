# AVAMALE free intake and pilot release runbook (November 2026)

**Status: REVIEW ONLY — no real client document intake, unapproved marketing sends or automated approvals.**

## System of record

- Website: https://headless-tnhgpdqwfaa-avamalemoses-140c.wix-site-host.com/service-enquiry.html
- Official Wix form: https://avamalemoses.wixforms.com/f/7524962517638972612
- Form ID: `34030b14-4829-4a50-ac05-fb067740e86c`
- Google Sheet: https://docs.google.com/spreadsheets/d/16UlIQmwyrvvIArY-9PCMp-XJ5DWzNNoCm6X67T_7Tmk/edit
- Restricted owner-only folder: https://drive.google.com/drive/folders/1bLgIGvKGHG115olwXLM6SyTMP4jQbeKz

## 1. Wix → Sheets, the free path

The native Wix form has a confirmed *owner notification* automation. Wix does **not** currently have a configured Google Sheets action. The Gmail notification includes the five field values but not the underlying Wix submission ID. Until the Wix ID is independently reconciled, the importer must label rows `INBOUND_UNVERIFIED` and must never start prospecting sequences.

- In the existing Lead Engine Google Sheet, open **Extensions → Apps Script**.
- Paste `avamale-workflows/integrations/wix_email_to_sheets.gs` into a new script file. Do not overwrite another working script. Copy the parser and config constants unchanged unless the actual sender domain requires an audited update.
- Check the Wix notification **From** domain from the owner's legitimate Gmail header. The source code permits `wix.com`, `wixautomations.com`, `wixemails.com`; adjust only for a verified first-party Wix domain.
- Run `wixImportPreview()` and read execution logs. It sends no messages and writes no rows. If notifications are absent, don't switch on a trigger.
- Run `wixInstallIntakeTrigger()` only after owner authorization. It installs exactly one **hourly intake-only** trigger; does not modify other triggers.
- Submit a non-sensitive owner-controlled form entry through the **public Wix-hosted form**, then confirm the native Wix status and owner notification. Check that exactly one Ledger row appears with `gmail_message_id`, `INBOUND_UNVERIFIED` status and no `step` or `last_sent`.
- Run the importer a second time. It should append **zero** additional rows. Review the Wix dashboard to associate the correct native submission ID if needed; never guess it.
- Delete the synthetic test lead after recording a QA log. Do not count as a real opportunity.

**Release criterion:** independent visitor submission, native confirmation, owner notification, exactly one Sheets row, no duplicate on retry, opt-out/suppression preserved. A mere created Wix test using a privileged API does not prove the visitor flow.

**Important:** the Apps Script is a *proposed integration*, not deployed by this repository. A GitHub pull request cannot authorize Google OAuth, install Apps Script triggers or verify mailbox sender identity.

## 2. Reply-stop and sending

Config must remain `PAUSE=TRUE`, `ARMED=FALSE`, `DRY_RUN=TRUE` until a provider-verified live test passes. `release_gates.py` is a pure preflight and **does not send email**.

The final sender must:

1. Re-read Config and full suppression ledger immediately before each attempt.
2. Check complete provider message/thread identity, not just a partial inbox search; include opt-outs, bookings and bounces.
3. Treat uncertain sender headers, truncated queries, API failures or incomplete checks as **STOP**.
4. Deduplicate via an immutable provider message/thread ID and atomic send ledger. Never retry a message when the previous send outcome is uncertain.
5. Honor reply, booking, opt-out or manual handoff immediately. Keep Peacebrooks on `REPLIED_HOLD`, never auto-sequence it.
6. Prove controls using the owner's test addresses only: pause, dry-run, invalid address, bounce, opt-out, reply, duplicate and restart tests. Review each case before any real dispatch.

**Release criterion:** installed provider adapter, reproducible tests and human authorization. Existing draft CI does not establish these.

## 3. Real document intake — HOLD

The existing local engine works with **pre-transcribed** CSV/values-only XLSX structured records. It does not read PDFs, authenticate shipments, or make customs/payment decisions. A private Drive folder with a single owner permission is necessary but **not sufficient** for sensitive customer records.

Before any real file:

1. Buyer identity and written scoped purpose confirmed; NDA/DPA reviewed where applicable.
2. Choose approved private transfer access; no public links or emailing unredacted statements by default.
3. Verify file upload limits, MIME/type validation, formula/CSV injection controls, malware policy, access logs and incident response.
4. Record permitted roles, processing location, subprocessor disclosures and retention/deletion schedule, and test revocation.
5. Run and document an encrypted backup/recovery drill where applicable; verify Neon/Render durability separately.
6. Agree reviewer, deliverable, limitation disclosure, turnaround window and deletion evidence.
7. Validate the real workflow against synthetic, consented and de-identified tests before allowing live documents.

**Release criterion:** signed-off access and data-processing plan plus observed end-to-end test. Until then use fictional records only. No confidential uploads via the public Wix form.

## 4. Synthetic product pilot

Use GitHub CI tests `tests/test_published_offer_pilots.py`:

- AA03: compare invoice 500 cartons, packing list 480, and bill of lading 480; show reviewer-required quantity and consignee differences. Do **not** imply that the present engine detected a document revision-date mismatch; the schema has no date field.
- AA03 clean case: consistent fields still require a human.
- AA04: PO 100, invoice 100, receipt 70 should generate `HOLD_THREE_WAY_EXCEPTION`, **not** block a payment automatically.
- AA04 clean case: agreement still requires human confirmation.
- Confirm HTML/CSV/XLSX reports remain non-executable, traceable, and labeled synthetic.

### Proof and release language

"AVAMALE is testing a human-reviewed document comparison workflow on synthetic structured records. The example illustrates potential exceptions; it is not proof of actual missing goods, money recovered, authentication, or production readiness."

## 5. Approval matrix

| Gate | Currently proven | Open requirement |
|---|---|---|
| Official Wix form | Public URL and owner email notifications | Visitor-level end-to-end confirmation |
| Wix → Google Sheet | One past manual synthetic transfer | Apps Script OAuth, sender verification, hourly trigger and duplicate replay |
| Reply-stop | Offline Python safety tests | Provider adapter, full inbox audit and live stop-on-reply test |
| Drive folder | Owner-only permission, not shared | Data protection review and actual storage workflow |
| AA03/AA04 | Synthetic engine tests in draft PR | Independent reviewer sign-off and real pilot acceptance |
| Revenue | Public offers and fictional report | Qualified lead, accepted quote, payment and delivery |

Do not claim these gates passed until an observable receipt/test report is available.
