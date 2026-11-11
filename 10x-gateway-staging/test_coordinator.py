"""Synthetic end-to-end checks: PostgreSQL + policy + Gmail-shaped evidence."""
import unittest
from datetime import datetime,timezone
from dataclasses import replace
from test_postgres_gateway import PostgresGatewayIntegration
from test_connector_policy import approved,SENDER,ADDRESS,message
from coordinator import StagingCoordinator

class StagingCoordinatorAcceptance(PostgresGatewayIntegration):
    def dispatch_args(self):
        x=self.args()
        # Gmail read-only adapter expects a validated email sender
        x["sender"]=SENDER
        x["recipient"]=ADDRESS
        return x

    def fake_search(self,pages):
        return lambda **kwargs: pages[kwargs["next_page_token"]]

    def simulate(self,a,policy,pages,call):
        return StagingCoordinator(self.gw,self.fake_search(pages)).simulate(
            **a,aliases=[ADDRESS],policy=policy,
            now=datetime(2026,11,9,10,tzinfo=timezone.utc),fake_send=call)

    def test_existing_real_shaped_sent_not_resent(self):
        a=self.dispatch_args()
        calls=[]
        s=self.simulate(a,approved(),{"":dict(emails=[message("older")],next_page_token=None)},
                        lambda *args:calls.append(args))
        self.assertEqual(s.state,"SENT")
        self.assertEqual(s.gmail_id,"older")
        self.assertFalse(calls)

    def test_approved_empty_history_one_fake_send(self):
        a=self.dispatch_args()
        calls=[]
        def fake(*args):
            calls.append(args)
            return "fake-provider-id"
        s=self.simulate(a,approved(),{"":dict(emails=[],next_page_token=None)},fake)
        self.assertEqual(s.state,"SENT")
        self.assertEqual(len(calls),1)
        self.simulate(a,approved(),{"":dict(emails=[],next_page_token=None)},fake)
        self.assertEqual(len(calls),1)

    def test_incomplete_history_blocks_fake_send(self):
        a=self.dispatch_args()
        calls=[]
        s=self.simulate(a,approved(),{"":dict(emails=[],next_page_token="missing")},
                        lambda *args:calls.append(args))
        self.assertEqual(s.state,"BLOCKED")
        self.assertFalse(calls)

    def test_opt_out_blocks_fake_send(self):
        a=self.dispatch_args()
        calls=[]
        s=self.simulate(a,replace(approved(),opted_out=True),
                        {"":dict(emails=[],next_page_token=None)},
                        lambda *args:calls.append(args))
        self.assertEqual(s.state,"BLOCKED")
        self.assertFalse(calls)

    def test_fake_provider_timeout_blocks_retry(self):
        a=self.dispatch_args()
        counter=[0]
        def fail(*args):
            counter[0]+=1
            raise TimeoutError("provider accepted? response lost")
        with self.assertRaises(TimeoutError):
            self.simulate(a,approved(),{"":dict(emails=[],next_page_token=None)},fail)
        self.assertIn(self.gw.get(a["key"]).state,{"SENDING","UNCERTAIN"})
        self.simulate(a,approved(),{"":dict(emails=[],next_page_token=None)},fail)
        self.assertEqual(counter[0],1)

if __name__=="__main__": unittest.main()
