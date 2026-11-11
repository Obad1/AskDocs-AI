"""Read-only translation of Gmail connector search results.

No credentials, network calls or send operations. Caller injects the connected
Gmail search action as a function. An UNSENT result is NEVER send authorization.
"""
from email.utils import parseaddr
from collections.abc import Mapping
import re
from gmail_preflight import inspect_history, Preflight

_ADDRESS = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")

def _address(value):
    if not isinstance(value, str):
        raise ValueError("missing sender identity")
    email=parseaddr(value)[1].strip().lower()
    if not _ADDRESS.fullmatch(email):
        raise ValueError("invalid sender identity")
    return email

def _normalized_page(provider_page):
    if not isinstance(provider_page, Mapping):
        raise ValueError("invalid Gmail response")
    # Some connector outputs are enveloped as {result:{...}}.
    page=provider_page.get("result",provider_page)
    if not isinstance(page,Mapping) or "next_page_token" not in page or not isinstance(page.get("emails"),list):
        raise ValueError("missing pagination marker or emails")
    normalized=[]
    for item in page["emails"]:
        if not isinstance(item,Mapping) or not isinstance(item.get("id"),str):
            raise ValueError("missing Gmail message ID")
        # Gmail connector's canonical sender field is from_; a different adapter
        # may supply from. Reject disagreements rather than silently choose.
        raw=item.get("from_")
        alt=item.get("from")
        if raw is None and alt is None: raise ValueError("missing Gmail sender")
        if raw is not None and alt is not None and _address(raw)!=_address(alt):
            raise ValueError("conflicting Gmail sender fields")
        frm=_address(raw if raw is not None else alt)
        to=item.get("to")
        labels=item.get("labels")
        if not isinstance(to,list) or not isinstance(labels,list) or not all(isinstance(x,str) for x in labels):
            raise ValueError("incomplete Gmail headers")
        recipients=[_address(v) for v in to]
        if not isinstance(item.get("subject"),str):
            raise ValueError("missing Gmail subject")
        normalized.append(dict(id=item["id"],**{"from":frm},to=recipients,labels=labels,subject=item["subject"]))
    return dict(emails=normalized,next_page_token=page["next_page_token"])

def read_only_preflight(search, *, sender_account, aliases,
                        alias_scope_verified=False, all_mail_verified=False,
                        max_pages=25):
    """search(query=..., max_results=..., next_page_token=...) -> Gmail response.

    An alias missing from the verified canonical group must make
    alias_scope_verified=False. For now this assertion is external and not
    automatically established by the Gmail API.
    """
    if not alias_scope_verified or not all_mail_verified:
        return Preflight("BLOCKED", reason="alias scope or Gmail visibility unverified")
    try:
        sender=_address(sender_account)
        canonical=sorted({_address(a) for a in aliases})
        if not canonical or len(canonical)>40:
            return Preflight("BLOCKED",reason="alias set incomplete or oversized")
        # No time filter: all history. Both directions, including archived,
        # Spam and Trash, with complete pagination; Gmail search semantics
        # still require independent account/alias group proof.
        query="in:anywhere {"+ " ".join(
            op+addr for addr in canonical for op in ("to:","from:")) +"}"
        def fetch(token):
            result=search(query=query,max_results=100,next_page_token=token or "")
            return _normalized_page(result)
        return inspect_history(fetch, aliases=canonical,
                   alias_scope_verified=True,all_mail_verified=True,
                   sender_account=sender,max_pages=max_pages)
    except Exception:
        return Preflight("BLOCKED",reason="Gmail connector read or normalization failed")
