"""Non-sending release preflight for AVAMALE. No network, file or mail side effects."""
from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Mapping

EMAIL = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
FORM_ID = "34030b14-4829-4a50-ac05-fb067740e86c"
STOP = frozenset({"REPLIED","REPLIED_HOLD","OPTOUT","BOUNCED","LOST","PAID",
                  "CALL","QUOTE","SAMPLE","DONE","MANUAL_ONLY","HOLD",
                  "QA_TEST_NO_SEND","DO_NOT_CONTACT","MANUAL_MERGE"})

@dataclass(frozen=True)
class Decision:
    allowed: bool
    code: str
    reason: str

def truthy(value):
    return str(value).strip().lower() in {"true","yes","y","1"}

def preflight_send(*, config: Mapping[str,object], contact: Mapping[str,object],
                   reply_audit_complete: bool, suppression_complete: bool,
                   sender_identity_verified: bool, delivery_provider_verified: bool) -> Decision:
    """Recheck before each send; authorization does not itself cause dispatch."""
    if truthy(config.get("PAUSE",True)):
        return Decision(False,"PAUSED","Global PAUSE is on")
    if not truthy(config.get("ARMED",False)):
        return Decision(False,"UNARMED","Explicit authorization absent")
    if truthy(config.get("DRY_RUN",True)):
        return Decision(False,"DRY_RUN","Dry-run only")
    if not all((reply_audit_complete,suppression_complete,sender_identity_verified,delivery_provider_verified)):
        return Decision(False,"AUDIT_INCOMPLETE","Inbox, suppression, identity or provider unverified")
    email=str(contact.get("email") or "").strip().lower()
    if not EMAIL.fullmatch(email):
        return Decision(False,"INVALID_EMAIL","Verified address required")
    if str(contact.get("status") or "").strip().upper() in STOP:
        return Decision(False,"TERMINAL","Human-owned or stopped contact")
    if truthy(contact.get("replied")) or truthy(contact.get("opted_out")):
        return Decision(False,"SUPPRESSED","Reply or opt-out recorded")
    if not truthy(contact.get("verified")):
        return Decision(False,"UNVERIFIED","Verification required")
    return Decision(True,"PRECHECK_ONLY","Provider-side deduplication and fresh recheck still required")

def normalize_wix_submission(submission: Mapping[str,object], *,
                             known_ids: set[str], suppressed_emails: set[str]):
    """Normalize confirmed form entries; caller must atomically dedupe in Sheets."""
    sid=str(submission.get("id") or "").strip()
    if not sid or sid in known_ids:
        return Decision(False,"DUPLICATE_OR_MISSING_ID","Existing/absent Wix ID"),None
    if submission.get("formId")!=FORM_ID or submission.get("status")!="CONFIRMED":
        return Decision(False,"NOT_CONFIRMED","Wrong form or unconfirmed"),None
    fields=submission.get("submissions")
    if not isinstance(fields,Mapping):
        return Decision(False,"INVALID_FIELDS","Fields absent"),None
    email=str(fields.get("email_avsample") or "").strip().lower()
    if not EMAIL.fullmatch(email):
        return Decision(False,"INVALID_EMAIL","Work email required"),None
    if email in {x.strip().lower() for x in suppressed_emails}:
        return Decision(False,"SUPPRESSED","Existing suppression is authoritative"),None
    name=str(fields.get("first_name_avsample") or "").strip()[:100]
    company=str(fields.get("company_avsample") or "").strip()[:150]
    service=str(fields.get("service_avsample") or "").strip()[:120]
    problem=str(fields.get("bottleneck_avsample") or "").strip()[:900]
    if not name or not company or not service:
        return Decision(False,"INVALID_FIELDS","Qualification incomplete"),None
    if any(x in (name+" "+company+" "+problem).upper() for x in
           ("INTERNAL QA","SYNTHETIC NOTICE CHECK","WIX-GSHEET-SYNC-QA","CONTROLLED TEST")):
        return Decision(False,"QA_EXCLUDED","Synthetic test is not a buyer"),None
    return Decision(True,"READY_TO_SYNC","Ready for locked insertion and review"),{
        "wix_submission_id":sid,"wix_submission_status":"CONFIRMED",
        "contact_name":name,"email":email,"company":company,"service":service,
        "notes":problem,"status":"INBOUND_NEW",
        "next_action":"Human qualification; no automatic cold send"}

def classify_pilot_upload(*, filename: str, documented_synthetic: bool,
                          safe_intake_approved: bool) -> Decision:
    """Initial filename gate only; never substitute for malware/privacy review."""
    if not documented_synthetic and not safe_intake_approved:
        return Decision(False,"REAL_DATA_HOLD","NDA, access, deletion and security review required")
    if not filename.lower().endswith((".csv",".xlsx")):
        return Decision(False,"UNSUPPORTED_FORMAT","Structured CSV/values-only XLSX; no PDFs")
    return Decision(True,"REVIEW_REQUIRED","Contents still require validation and a person")
