"""Read-only Gmail history interpretation. No credentials or Gmail send capability.

The injected page_fetch MUST query account-wide history for EVERY approved alias;
it MUST return complete pages and authoritative next_page_token markers. The gateway
does not trust a negative result if those obligations are unverified.
"""
from collections.abc import Mapping
from dataclasses import dataclass
import re

MAIL = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

@dataclass(frozen=True)
class Preflight:
    kind: str  # SENT, UNSENT, BLOCKED
    message_id: str | None = None
    reason: str = ""

def inspect_history(page_fetch, *, aliases, alias_scope_verified, all_mail_verified,
                    sender_account, max_pages=25):
    if not alias_scope_verified or not all_mail_verified:
        return Preflight("BLOCKED",reason="alias or mailbox scope incomplete")
    aliases={str(v).strip().lower() for v in aliases}
    if not aliases or not all(MAIL.fullmatch(v) for v in aliases) or not MAIL.fullmatch(sender_account.lower()):
        return Preflight("BLOCKED",reason="invalid sender/company aliases")
    token=None
    tokens=set()
    messages={}
    try:
        for _ in range(max_pages):
            page=page_fetch(token)
            if not isinstance(page,Mapping) or "next_page_token" not in page or not isinstance(page.get("emails"),list):
                return Preflight("BLOCKED",reason="partial or invalid page")
            for msg in page["emails"]:
                if not isinstance(msg,Mapping) or not isinstance(msg.get("id"),str):
                    return Preflight("BLOCKED",reason="incomplete message")
                to=msg.get("to")
                labels=msg.get("labels")
                if not isinstance(to,list) or not isinstance(labels,list) or not isinstance(msg.get("from"),str):
                    return Preflight("BLOCKED",reason="message direction not verified")
                if not isinstance(msg.get("subject"),str):
                    return Preflight("BLOCKED",reason="message subject missing")
                m_id=msg["id"]
                if m_id in messages and messages[m_id]!=dict(msg):
                    return Preflight("BLOCKED",reason="conflicting Gmail IDs")
                messages[m_id]=dict(msg)
            nxt=page["next_page_token"]
            if nxt in (None,""):
                break
            if not isinstance(nxt,str) or nxt in tokens:
                return Preflight("BLOCKED",reason="pagination stalled")
            tokens.add(nxt)
            token=nxt
        else:
            return Preflight("BLOCKED",reason="page cap exceeded")
    except Exception:
        return Preflight("BLOCKED",reason="provider history unavailable")

    sent=[]
    for m in messages.values():
        to={str(x).strip().lower() for x in m["to"]}
        frm=m["from"].strip().lower()
        labels=set(m["labels"])
        if frm in aliases and "SENT" not in labels:
            return Preflight("BLOCKED",reason="company replied or another inbound signal")
        if "SENT" in labels and to.intersection(aliases):
            sent.append(m["id"])
    if len(sent)==1:
        return Preflight("SENT",message_id=sent[0],reason="existing Gmail Sent")
    if sent:
        return Preflight("BLOCKED",reason="multiple company sends require human review")
    return Preflight("UNSENT",reason="fully paginated no-match snapshot; NOT live send permission")
