import unittest
from datetime import datetime, timezone
from dataclasses import replace
from gmail_connector_adapter import read_only_preflight
from policy import Policy, authorize

SENDER="synthetic@test.invalid"
ADDRESS="team@fictional.example"
def message(mid="s1",direction="sent"):
    if direction=="sent":
        return dict(id=mid,from_="Synthetic Sender <"+SENDER+">",to=[ADDRESS],
                    labels=["SENT"],subject="Initial synthetic email")
    return dict(id=mid,from_="Team <"+ADDRESS+">",to=[SENDER],
                labels=["INBOX"],subject="Re: Initial synthetic email")
def fake_search(pages, queries):
    def search(**kwargs):
        queries.append(kwargs)
        token=kwargs["next_page_token"]
        return pages[token]
    return search
def inspect(pages,**kw):
    return read_only_preflight(fake_search(pages,[]),sender_account=SENDER,aliases=[ADDRESS],
       alias_scope_verified=True,all_mail_verified=True,**kw)

def approved():
    return Policy(**{name:True for name in (
        "account_verified","user_authorized","copy_approved","recipient_verified",
        "company_aliases_verified","sent_history_complete","replies_checked",
        "suppressions_checked","unique_dispatch_lease","account_capacity_verified",
        "campaign_stage_approved")})

class GmailAdapterAcceptance(unittest.TestCase):
    def test_live_shaped_sent_from_accepted(self):
        q=[]
        v=read_only_preflight(fake_search({"":dict(emails=[message()],next_page_token=None)},q),
            sender_account=SENDER,aliases=[ADDRESS],alias_scope_verified=True,all_mail_verified=True)
        self.assertEqual((v.kind,v.message_id),("SENT","s1"))
        self.assertIn("in:anywhere",q[0]["query"])
        self.assertNotIn("after:",q[0]["query"])
    def test_full_two_page_empty(self):
        self.assertEqual(inspect({"":dict(emails=[],next_page_token="p2"),
            "p2":dict(emails=[],next_page_token=None)}).kind,"UNSENT")
    def test_real_shaped_inbound_blocks(self):
        self.assertEqual(inspect({"":dict(emails=[message("r1","inbound")],next_page_token=None)}).kind,"BLOCKED")
    def test_missing_last_page_blocks(self):
        self.assertEqual(inspect({"":dict(emails=[],next_page_token="p2")}).kind,"BLOCKED")
    def test_missing_sender_blocks(self):
        m=message();del m["from_"]
        self.assertEqual(inspect({"":dict(emails=[m],next_page_token=None)}).kind,"BLOCKED")
    def test_sender_disagreement_blocks(self):
        m=message();m["from"]="spoof@fictional.example"
        self.assertEqual(inspect({"":dict(emails=[m],next_page_token=None)}).kind,"BLOCKED")
    def test_missing_alias_verification_blocks(self):
        self.assertEqual(read_only_preflight(lambda **kw:None,sender_account=SENDER,aliases=[ADDRESS]).kind,"BLOCKED")
    def test_multiple_sends_blocks(self):
        self.assertEqual(inspect({"":dict(emails=[message("s1"),message("s2")],next_page_token=None)}).kind,"BLOCKED")
    def test_search_provider_error_blocks(self):
        self.assertEqual(read_only_preflight(lambda **kw:(_ for _ in ()).throw(RuntimeError()),
            sender_account=SENDER,aliases=[ADDRESS],alias_scope_verified=True,all_mail_verified=True).kind,"BLOCKED")

class PolicyAcceptance(unittest.TestCase):
    def test_weekday_in_window(self):
        self.assertTrue(authorize(approved(),datetime(2026,11,9,9,0,tzinfo=timezone.utc)))
    def test_saturday_blocks(self):
        self.assertFalse(authorize(approved(),datetime(2026,11,7,10,0,tzinfo=timezone.utc)))
    def test_after_hours_blocks(self):
        self.assertFalse(authorize(approved(),datetime(2026,11,9,16,0,tzinfo=timezone.utc)))
    def test_missing_any_gate_blocks(self):
        for field in ("account_verified","user_authorized","copy_approved","recipient_verified",
                      "company_aliases_verified","sent_history_complete","replies_checked",
                      "suppressions_checked","unique_dispatch_lease","account_capacity_verified",
                      "campaign_stage_approved"):
            with self.subTest(field=field):
                self.assertFalse(authorize(replace(approved(),**{field:False}),datetime(2026,11,9,10,tzinfo=timezone.utc)))
    def test_any_stop_signal_blocks(self):
        for field in ("manual_takeover","bounced","opted_out","uncertain_prior_send","blocked"):
            with self.subTest(field=field):
                self.assertFalse(authorize(replace(approved(),**{field:True}),datetime(2026,11,9,10,tzinfo=timezone.utc)))
    def test_naive_datetime_blocks(self):
        self.assertFalse(authorize(approved(),datetime(2026,11,9,10,0)))

if __name__=="__main__":unittest.main()
