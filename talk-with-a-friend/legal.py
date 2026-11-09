"""Public legal notices and version endpoint for Talk with a Friend.

This is lawyer-review DRAFT text, not certification of compliance or immunity.
Never put user-submitted markup in this HTML.  No legal notice excuses unlawful
conduct, negligent operation, a data breach, or infringement of statutory rights.
"""
from __future__ import annotations

from html import escape
from fastapi import HTTPException
from fastapi.responses import HTMLResponse

LEGAL_VERSION = "2026-11-09-draft-1"
UPDATED = "9 November 2026"

POLICIES = {
  "terms": (
    "Terms of Use",
    [
      ("Service and operator", "Talk with a Friend is a pilot text-chat service under the Avamale name. The legal entity, registered business address and a dedicated support/legal contact are not yet finalized; commercial launch is pending those details. Free random stranger chat is separate from optional paid general-advice conversations."),
      ("Eligibility", "You must be at least 18 and able to enter an agreement. Age is self-declared rather than independently verified. Do not use the service if you are not eligible."),
      ("What the service provides", "The free mode pairs adult strangers for temporary text conversations. The paid feature, if enabled later, provides a scheduled 30-minute text conversation for general life decisions and perspective. No one is guaranteed a match, response, outcome, continuous availability, or permanent record."),
      ("Community rules", "No harassment, threats, hate speech, exploitation, sexual coercion, grooming of minors, stalking, financial solicitation, fraud, impersonation, doxxing, spam, malicious links, unlawful content, or attempts to bypass restrictions. Respect boundaries; a participant may leave at any time. You must not record or republish another person's private messages without any consent or lawful basis required."),
      ("Personal safety and limits", "Strangers may misrepresent themselves or say offensive or inaccurate things. The beta has no continuously staffed content moderation, and in-app reports currently store only temporary session metadata; they may not be reviewed. Do not disclose passwords, precise addresses, phone numbers, payment details or sensitive information. The host server relays text; chats are not end-to-end encrypted."),
      ("Not professional or emergency care", "General advice is not psychotherapy, licensed counselling, a medical diagnosis, an emergency line, legal representation or individualized regulated financial advice. Therapists are not currently bookable. Contact appropriate emergency responders or qualified professionals for urgent or specialized help."),
      ("Service interruptions and security", "Connections and messages may be lost when the free-tier server sleeps, restarts or experiences errors. We do not promise absolute security or anonymity. We remain responsible for safeguards and conduct required by law; these terms do not replace those responsibilities."),
      ("Fair liability and protected rights", "To the fullest extent lawful, each user accepts ordinary risks inherent in meeting unknown people and is responsible for their own unlawful conduct. No term excludes liability for our fraud, negligence, privacy breaches, unlawful conduct or any liability that cannot legally be excluded. Nothing limits a consumer's rights or ability to complain to a competent court or regulator."),
      ("Changes and disputes", "We may update the terms prospectively. Material changes require a new version and a fresh affirmative acceptance before starting a new session. Applicable mandatory rights remain protected. Nigerian law is the proposed governing law, subject to lawfully applicable protections elsewhere. Our legal and privacy contact is pending; for now @obadiah248 on Instagram is a provisional inquiry route.")
    ]
  ),
  "privacy": (
    "Privacy Notice",
    [
      ("Controller and law", "The service is operated under the Avamale name. The full legal data-controller identity, business address and dedicated privacy contact have not yet been confirmed, so this notice is a draft. Our intended compliance framework includes the Nigeria Data Protection Act 2023 and the NDPC General Application and Implementation Directive 2025. Additional laws, such as the EU GDPR, may apply when a service targets people in those jurisdictions; the geographic applicability needs review."),
      ("What we handle in free chat", "To operate random matching we temporarily process a randomly generated session identifier, a hashed session secret, connection status, time of session, and the text sent while the connection is active. A network address is used transiently to rate-limit session creation. Content is relayed to the connected stranger, who may copy or screenshot it. The application does not intentionally save chat transcripts to a database or collect profile names or phone numbers."),
      ("Abuse reports", "When someone reports another user, the current beta temporarily holds a report reason, anonymous session IDs, and a timestamp in the application server's memory. These reports are not a reliable staffed complaint system and disappear when that process restarts. The feature is not suitable for emergency reporting."),
      ("Paid booking information", "If paid advice is enabled, the system would store your booking email, optional short discussion topic, a transaction reference, transaction amount and currency, paid/accepted/completed status, creation and acceptance timestamps, an access-token hash and the version and time of your accepted booking terms in PostgreSQL. The customer access token is temporarily stored in the browser's sessionStorage. Paystack handles card payment on its own site; we do not request full card numbers."),
      ("Why and on what basis", "Live session identifiers, relayed text and booking details are processed to deliver features you request or to take pre-contractual steps. Rate limits and lawful incident response may rely on legitimate security interests or legal obligations, subject to applicable law. Optional promotional messaging would require an appropriate separate lawful basis; accepting these terms does not consent to marketing. This page describes processing, it is not a blanket consent to unrelated data use."),
      ("Recipients, hosting and transfers", "Text you send is disclosed to the connected chat participant or paid advice host. Server infrastructure and technical access information are processed by Render and its suppliers; the connected Render instance is currently hosted in Frankfurt, Germany. If paid checkout opens, Paystack and any contracted PostgreSQL host will process information necessary for payment or booking. Cross-border transfers and processor safeguards must be assessed and documented before large-scale or paid processing."),
      ("Retention and security", "Free random-chat state and messages are held in volatile memory and cleared on restart or session expiry; server and provider infrastructure logs may have separate retention periods. No end-to-end encryption is provided; transport uses HTTPS/WSS when hosted. Paid records need a defined retention/deletion schedule, encryption, access controls, verified backup policy, incident response and processor agreements before payments are opened."),
      ("Data-subject rights", "Subject to applicable law and verification where necessary, users may request access, rectification, erasure, restriction, objection, portability where available, or withdrawal of consent where processing relies on consent. Requests about ephemeral data may be impossible after it has already disappeared. You can complain to the Nigeria Data Protection Commission via ndpc.gov.ng. A direct dedicated privacy email is a prerequisite to full commercial release."),
      ("Cookies and third parties", "The current app does not intentionally use analytics cookies or an ad-tracking SDK. It stores a paid customer's temporary booking token in sessionStorage when checkout is initiated. External links to Instagram, Paystack, and hosting providers have their own privacy notices."),
      ("Children and sensitive information", "This service is intended only for adults 18+. Do not share health, biometric, identity, location, banking or other sensitive personal information in free chat or optional booking topics. Age verification and safeguards against minors are incomplete and require review before broad public advertising."),
      ("Incidents and changes", "Suspected personal data breaches should be investigated and escalated under the NDPA/GAID and other applicable laws; statutory notifications depend on risk and timing. A privacy notice cannot eliminate those duties. Material policy updates will be versioned and shown before a new session.")
    ]
  ),
  "pledge": (
    "Community Pledge",
    [
      ("My voluntary user commitment", "I affirm I am at least 18. I will speak respectfully, will not threaten or harass another person, and will not exploit or try to contact minors through the service."),
      ("Privacy and boundaries", "I will not ask for passwords, precise locations, bank details, identity documents or sensitive private information. I will not publish someone's private chat without any legally required permission."),
      ("What I understand", "I understand that participants are strangers and their identities are not verified; I can leave or skip. In-app reports have no active staff review in this beta. The platform is not an emergency or professional mental-health service."),
      ("Legal rights preserved", "This is an ordinary user pledge, not a sworn oath or affidavit. Agreeing does not surrender my privacy, consumer, regulatory, criminal-reporting or other non-waivable legal rights; it also does not exempt the operator from legal responsibility.")
    ]
  ),
  "safety": (
    "Safety and Crisis Notice",
    [
      ("Not safe for confidential disclosures", "Do not share bank credentials, full identity, home address, contact details, location, intimate media or private health records. Random strangers can misrepresent themselves and can preserve what they read. Chats are not end-to-end encrypted."),
      ("Reporting limitations", "Use Next, Report or Leave to exit unsafe chat, but do not rely on the beta's temporary reports for emergency intervention or later investigation. There is no continuous human moderation or guarantee of report review."),
      ("No professional or emergency service", "Do not use this website for suicide crisis response, medical emergencies, threats of immediate harm, psychotherapy, diagnosis, or legal advice. Seek local emergency services or appropriate qualified professional support."),
      ("Developer safety obligations", "Public promotion to unknown users should not scale until operational moderation, reliable abuse-report triage, reasonable age assurance, response procedures, data governance and security controls have been reviewed.")
    ]
  ),
  "advice": (
    "Paid Advice Terms and Cancellation Notice",
    [
      ("Service description", "Private Clarity Session is a proposed 30-minute, one-to-one LIVE TEXT conversation with a general advice host, presently priced at NGN 10,000 if offered. It is not a voice/video call, psychotherapy, mental-health treatment, professional counselling, or a promise of a particular result."),
      ("Payment and booking status", "Bookings are closed until the host publishes availability and the payment processor and persistent booking database are verified. A checkout or browser return does not itself confirm a successful payment. The server must independently verify the amount and reference, and the host must accept the booking before the private session is available."),
      ("Cancellation, rescheduling and refund rights", "The final cancellation notice, rescheduling window, refunds and support method must be published before any payment is taken and must comply with applicable consumer law. No 'no refunds under any circumstances' clause is imposed. Customers retain their mandatory cancellation/refund rights."),
      ("Scope and confidentiality", "Only send what is needed to discuss an everyday non-clinical question. Do not disclose highly sensitive medical, financial or identity information. The live text is relayed by the service and can be copied by the recipient. Chat transcripts are not intentionally stored by the application; no absolute confidentiality or end-to-end encryption is promised."),
      ("Fair liability", "No contractual term excludes negligence, fraud, privacy obligations, non-waivable statutory rights or valid consumer remedies. The host is responsible for delivering the advertised service; a disclaimer does not excuse non-performance.")
    ]
  ),
}


def install(app):
    @app.get("/api/legal/version")
    async def legal_version():
        return {"version": LEGAL_VERSION, "updated": UPDATED,
                "documents": ["/legal/" + key for key in POLICIES]}

    @app.get("/legal/{document}", response_class=HTMLResponse)
    async def legal_document(document: str):
        if document not in POLICIES:
            raise HTTPException(404, "Unknown legal notice")
        title, sections = POLICIES[document]
        body = "".join(
            "<section><h2>" + escape(heading) + "</h2><p>" +
            escape(paragraph) + "</p></section>"
            for heading, paragraph in sections
        )
        links = " · ".join(
            '<a href="/legal/' + escape(name) + '">' + escape(item[0]) + "</a>"
            for name, item in POLICIES.items()
        )
        result = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                  '<meta name="viewport" content="width=device-width, initial-scale=1">'
                  "<title>" + escape(title) + " — Talk with a Friend</title>"
                  '<link rel="stylesheet" href="/assets/style.css"></head>'
                  '<body class="legal-body"><main class="legal-sheet">'
                  '<a href="/" class="legal-home">← Return to Talk with a Friend</a>'
                  '<p class="legal-kicker">DRAFT · FOR LOCAL LEGAL REVIEW</p>'
                  "<h1>" + escape(title) + "</h1>"
                  '<p class="legal-date">Updated ' + escape(UPDATED) + ' · Version '
                  + escape(LEGAL_VERSION) + "</p>"
                  '<p class="legal-alert">This draft does not replace a qualified legal'
                  ' review, professional safeguarding or appropriate privacy compliance.'
                  ' No clause guarantees protection from lawsuits.</p>'
                  + body + '<nav aria-label="Other legal notices">' + links + '</nav>'
                  '</main></body></html>')
        return HTMLResponse(result)
