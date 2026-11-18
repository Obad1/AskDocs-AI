# Talk with a Friend — public claim ledger and operating decision

Updated: 2026-11-18. Status: HOLD — protected 18+ experimental beta only. Marketing cannot be published automatically on code presence alone.

## Evidence rule

1. Public feature: actual https://talk-with-a-friend.onrender.com HTML + API + browser test on the deployed revision.
2. Source support: app.py and static/app.js, plus the exact test and commit.
3. If proof is from isolated local browsers rather than the deployed service, mark LOCAL_ONLY / not independently hosted.
4. Any claim whose evidence, age safeguards, staff coverage, user rights, music/license or exact final asset fails is HOLD.
5. Content approvals must identify the exact MP4 SHA-256; copy approval is not approval of a later cut.

| Candidate claim | Implementation | Evidence threshold | Current allowed wording |
| --- | --- | --- | --- |
| "Meet a randomly matched stranger" | WebSocket waiting queue, app.py | Two independent real browsers through public Render host; safety readiness | HOLD public claim until hosted interactive proof and age/moderation gates |
| "No account or invitation link" | Session token API, static homepage | Host route and ordinary browser path checked | Say "No account needed in our experimental demo"; does not imply identity or age assurance |
| "Next / Report / Leave" | app.py and app.js | Real hosted browser buttons and server receipts | Local demo shows controls with fictional participants; public claim HOLD |
| "No saved chat messages" | app.py has no app transcript persistence | Review the provider logs, backups, browser side and exceptions | Say "The application is designed not to retain chat transcripts; hosting providers process connection logs" |
| "Report received" | safety.py + separate Neon reports | Transaction committed, moderator queue retrievable, no secret leakage, failure test | Claim only after production DB test; currently HOLD |
| "18+ only" | Self-attestation checkbox | Suitable independent age assurance verified | Say "Intended for adults; age is self-declared and not independently verified" |
| "Safe, anonymous, moderated" | No comprehensive implementation | Real moderation staffing, safeguarding, identity/privacy assessment | NEVER use these blanket claims |
| "Therapy and emotional support from licensed clinicians" | No verified provider pipeline | Licensure, clinical protocol, regulatory/privacy review | DO NOT advertise |
| "Paid one-to-one general advice" | Payment integration is disabled | Live merchant tests, host availability, refund/consumer notices, persistent booking DB | "Planned private general-advice sessions; bookings currently closed" |

## Product demo truth

The reproducible capture script scripts/record_review_demo.py records the genuine local app interactions across four Playwright sessions. It saves a 9:16 H.264 MP4, raw recording and manifest. The footage is authentic to the *local beta app*, but not evidence that the live platform is available to every visitor; invented test participants are not real customers. The video is REVIEW ONLY. It has no owner-approved master A logo or independent rights/music review and is not an approved advertisement.

## Operating decision

- Public matchmaking: HOLD; TALK_PUBLIC_MATCHING_ENABLED=0.
- Private beta: Source tests permitted on a controlled local/staging environment. No claims of staffed moderation or age verification.
- Moderator queue: Access controlled by TALK_MODERATOR_SECRET, service database by REPORT_DATABASE_URL. Never expose secrets in media, logs, reports or GitHub.
- Reports: No chat transcript is deliberately retained; reason/receipt/status retained in database. A reviewer needs to be assigned before users can reasonably rely on follow-up. Until staff availability is established, do not send high-risk users into unattended chat.
- Revenue: ADVICE_BOOKINGS_ENABLED=0; ADVICE_HOST_AVAILABLE=0. Never collect payment while no actual service capacity and consumer terms are approved.
- Public marketing: HOLD pending hosted two-browser proof, actual product screenshot/brand verification, independent creative assessment, user rights/privacy and the exact-media release gate.
- Testing: Local CI regression and one-shot hosted HTTP proof are valuable but are not complete production safety clearance.
- Content production: Lead with the "one hello" concept and actual human interactions, not a generic AI/slideshow montage or a faux customer review.
- No cron, automatic keep-awake pings or posting jobs have been added.

## Evidence links

- Public URL: https://talk-with-a-friend.onrender.com
- Render service: https://dashboard.render.com/web/srv-dboj6k5g1s2s73fc6clg
- Safe hosted HTTP verification: https://github.com/Obad1/AskDocs-AI/actions/runs/44271578925
- Honest demo capture workflow: https://github.com/Obad1/AskDocs-AI/actions/workflows/talk-friend-demo.yml
- Baseline local tests: talk-with-a-friend/tests/test_chat.py and tests/test_browser.py

## Pending outside code

- A real staffed escalation plan, not only a queue interface.
- Jurisdiction-appropriate age assurance and legal/privacy review, including the Nigerian NDPA 2023 and GAID 2025.
- A trustworthy privacy-request support channel, defined retention/deletion schedule, terms of service, documented responsible operator.
- Real Paystack test account credentials, actual host, booking calendar, refund/cancellation process, and payment reconciliation.
- Independent perceptual video/audio review, exact AVAMALE approved A-master asset, and explicit product-claim authorization.
