"""Policy decision for staging-only outreach gateway. No send operation."""
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import FrozenSet

WAT = ZoneInfo("Africa/Lagos")
@dataclass(frozen=True)
class Policy:
    account_verified: bool
    user_authorized: bool
    copy_approved: bool
    recipient_verified: bool
    company_aliases_verified: bool
    sent_history_complete: bool
    replies_checked: bool
    suppressions_checked: bool
    unique_dispatch_lease: bool
    account_capacity_verified: bool
    campaign_stage_approved: bool
    manual_takeover: bool = False
    bounced: bool = False
    opted_out: bool = False
    uncertain_prior_send: bool = False
    blocked: bool = False

def authorize(p: Policy, now: datetime) -> bool:
    if not isinstance(p,Policy) or not isinstance(now,datetime) or now.tzinfo is None:
        return False
    if any((p.manual_takeover,p.bounced,p.opted_out,p.uncertain_prior_send,p.blocked)):
        return False
    if not all((p.account_verified,p.user_authorized,p.copy_approved,
                p.recipient_verified,p.company_aliases_verified,p.sent_history_complete,
                p.replies_checked,p.suppressions_checked,p.unique_dispatch_lease,
                p.account_capacity_verified,p.campaign_stage_approved)):
        return False
    local=now.astimezone(WAT)
    return local.weekday()<5 and 9 <= local.hour < 17
