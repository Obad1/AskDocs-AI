"""Staging-only coordinator. READ-ONLY Gmail + verified policy + fake send.

There is deliberately no real Gmail send API. A result of UNSENT is NOT
sufficient authorization; the caller must provide verified policy evidence.
"""
from datetime import datetime
from dataclasses import replace
from gmail_connector_adapter import read_only_preflight
from policy import authorize, Policy

class StagingCoordinator:
    def __init__(self, gateway, search):
        self.gateway=gateway
        self.search=search

    def simulate(self, *, key, sender, company, stage, recipient, subject,
                 aliases, policy:Policy, now:datetime, fake_send):
        # Assert all safety evidence before allowing the test-only fake sender.
        # A previous Gmail SENT ID must still be recorded as a prior-send BLOCK.
        def history():
            outcome=read_only_preflight(
                self.search,sender_account=sender,aliases=aliases,
                alias_scope_verified=policy.company_aliases_verified,
                all_mail_verified=policy.sent_history_complete)
            if outcome.kind=="SENT":
                return ("SENT",outcome.message_id)
            if outcome.kind=="UNSENT" and authorize(policy,now):
                return ("UNSENT",None)
            return ("BLOCKED",None)
        return self.gateway.run_fake_send(
            key=key,sender=sender,company=company,stage=stage,
            recipient=recipient,subject=subject,
            authorize=lambda:authorize(policy,now),
            preflight=history,fake_send=fake_send)
