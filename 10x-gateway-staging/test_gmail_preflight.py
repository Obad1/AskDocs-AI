import unittest
from gmail_preflight import inspect_history

SENDER="synthetic@test.invalid"
ALIAS="ops@fictional.example"

def msg(id,labels=("SENT",),recipient=ALIAS,frm=SENDER,subject="Earlier outreach"):
    return dict(id=id,labels=list(labels),to=[recipient],**{"from":frm},subject=subject)

def check(fetch,**kw):
    opts=dict(aliases={ALIAS},alias_scope_verified=True,all_mail_verified=True,sender_account=SENDER)
    opts.update(kw)
    return inspect_history(fetch,**opts)

def one(messages):
    return lambda _:dict(emails=messages,next_page_token=None)

class GmailPreflightTests(unittest.TestCase):
    def test_verified_empty(self):
        self.assertEqual(check(one([])).kind,"UNSENT")
    def test_prior_sent_different_subject(self):
        x=check(one([msg("m1")]))
        self.assertEqual((x.kind,x.message_id),("SENT","m1"))
    def test_inbound_blocks(self):
        x=check(one([msg("reply",labels=("INBOX",),recipient=SENDER,frm=ALIAS)]))
        self.assertEqual(x.kind,"BLOCKED")
    def test_alias_scope_missing_blocks(self):
        self.assertEqual(check(one([]),alias_scope_verified=False).kind,"BLOCKED")
    def test_allmail_scope_missing_blocks(self):
        self.assertEqual(check(one([]),all_mail_verified=False).kind,"BLOCKED")
    def test_missing_pagination_blocks(self):
        self.assertEqual(check(lambda _:{"emails":[]}).kind,"BLOCKED")
    def test_broken_record_blocks(self):
        self.assertEqual(check(one([{"id":"m1","labels":["SENT"]}])).kind,"BLOCKED")
    def test_multiple_sends_require_manual_reconciliation(self):
        self.assertEqual(check(one([msg("m1"),msg("m2")])).kind,"BLOCKED")
    def test_two_complete_pages(self):
        def page(t):
            if t is None:return dict(emails=[],next_page_token="n2")
            return dict(emails=[msg("m2")],next_page_token=None)
        self.assertEqual(check(page).message_id,"m2")
    def test_pagination_cycle_blocks(self):
        self.assertEqual(check(lambda _:dict(emails=[],next_page_token="loop")).kind,"BLOCKED")
    def test_provider_failure_blocks(self):
        def fail(_):raise TimeoutError("synthetic connection interrupted")
        self.assertEqual(check(fail).kind,"BLOCKED")
    def test_different_sender_not_considered_sent(self):
        self.assertEqual(check(one([msg("other",recipient="else@fictional.example")])).kind,"UNSENT")
    def test_conflicting_message_id_blocks(self):
        def pages(t):
            if t is None:return dict(emails=[msg("m1")],next_page_token="two")
            return dict(emails=[msg("m1",labels=("INBOX",),frm=ALIAS)],next_page_token=None)
        self.assertEqual(check(pages).kind,"BLOCKED")

if __name__=="__main__":
    unittest.main()
