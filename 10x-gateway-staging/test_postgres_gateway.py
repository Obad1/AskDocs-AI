"""Integration tests run against CI's disposable Postgres service only."""
import os
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4

try:
    import psycopg
except ImportError:
    psycopg = None

from gateway import Gateway, DDL

@unittest.skipUnless(psycopg is not None and os.getenv("TEST_GATEWAY_PG_DSN"),
    "requires explicit disposable CI PostgreSQL DSN and psycopg3")
class PostgresGatewayIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dsn=os.environ["TEST_GATEWAY_PG_DSN"]
        if os.environ.get("GATEWAY_TEST_ISOLATED") != "YES":
            raise RuntimeError("GATEWAY_TEST_ISOLATED=YES required; no production DB permitted")
        def connect():
            return psycopg.connect(cls.dsn)
        cls.connect=staticmethod(connect)
        with connect() as c:
            c.execute(DDL)
        cls.gw=Gateway(connect)

    def key(self):
        return "fixture-"+uuid4().hex

    def args(self):
        k=self.key()
        return dict(key=k,sender="synthetic@test.invalid",company=k+".invalid",
                    stage="INITIAL",recipient="buyer@test.invalid",subject="Synthetic Only")

    def test_concurrent_claim_one_winner(self):
        a=self.args()
        with ThreadPoolExecutor(max_workers=8) as pool:
            results=list(pool.map(lambda _: self.gw.claim(**a),range(8)))
        self.assertEqual(sum(f for f,_ in results),1)
        self.assertEqual({s.state for _,s in results},{"CHECKING"})
        with self.connect() as c:
            n=c.execute("SELECT count(*) FROM gateway_ci.dispatch WHERE idempotency_key=%s",(a["key"],)).fetchone()[0]
        self.assertEqual(n,1)

    def test_alternate_key_same_company_rejected(self):
        a=self.args()
        self.gw.claim(**a)
        with self.assertRaises(ValueError):
            self.gw.claim(**{**a,"key":self.key()})

    def test_preexisting_sent_no_send(self):
        a=self.args()
        calls=[]
        s=self.gw.run_fake_send(**a,authorize=lambda:True,
                preflight=lambda:("SENT","synthetic-existing-id"),
                fake_send=lambda *args:calls.append(args))
        self.assertEqual(s.state,"SENT")
        self.assertEqual(calls,[])

    def test_duplicate_retry_does_not_send_again(self):
        a=self.args()
        calls=[]
        def send(*args):
            calls.append(args)
            return "synthetic-sent-id"
        self.assertEqual(self.gw.run_fake_send(**a,authorize=lambda:True,
                          preflight=lambda:("UNSENT",None),fake_send=send).state,"SENT")
        self.assertEqual(self.gw.run_fake_send(**a,authorize=lambda:True,
                          preflight=lambda:("UNSENT",None),fake_send=send).state,"SENT")
        self.assertEqual(len(calls),1)

    def test_provider_timeout_leaves_no_replay(self):
        a=self.args()
        counter=[0]
        def fail(*args):
            counter[0]+=1
            raise TimeoutError("simulated Gmail accepted but response lost")
        with self.assertRaises(TimeoutError):
            self.gw.run_fake_send(**a,authorize=lambda:True,
                preflight=lambda:("UNSENT",None),fake_send=fail)
        self.assertIn(self.gw.get(a["key"]).state,("UNCERTAIN","SENDING"))
        self.gw.run_fake_send(**a,authorize=lambda:True,
                    preflight=lambda:("UNSENT",None),fake_send=fail)
        self.assertEqual(counter[0],1)

    def test_permission_and_incomplete_history_fail_closed(self):
        for authorize,history in [(lambda:False,lambda:("UNSENT",None)),
                                   (lambda:True,lambda:("BLOCKED",None))]:
            a=self.args()
            count=[]
            s=self.gw.run_fake_send(**a,authorize=authorize,preflight=history,
                fake_send=lambda *args:count.append(args))
            self.assertEqual(s.state,"BLOCKED")
            self.assertFalse(count)

    def test_rollback_on_audit_failure(self):
        a=self.args()
        self.gw.claim(**a)
        with self.connect() as c:
            with c.cursor() as cur:
                cur.execute("SAVEPOINT before_test")
                try:
                    cur.execute("UPDATE gateway_ci.dispatch SET state='SENDING' WHERE idempotency_key=%s",(a["key"],))
                    cur.execute("INSERT INTO gateway_ci.events(key,state) VALUES ('not-existent-fixture','SENDING')")
                except psycopg.Error:
                    cur.execute("ROLLBACK TO SAVEPOINT before_test")
            c.commit()
        self.assertEqual(self.gw.get(a["key"]).state,"CHECKING")

    def test_audit_is_immutable(self):
        a=self.args()
        self.gw.claim(**a)
        with self.assertRaises(psycopg.Error):
            with self.connect() as c:
                c.execute("UPDATE gateway_ci.events SET state='SENT' WHERE key=%s",(a["key"],))
        self.assertEqual(self.gw.get(a["key"]).state,"CHECKING")

    def test_stale_transition_rejected(self):
        a=self.args()
        self.gw.claim(**a)
        self.gw.transition(a["key"],"CHECKING","SENDING")
        with self.assertRaises((ValueError,psycopg.Error)):
            self.gw.transition(a["key"],"CHECKING","SENDING")
        self.assertEqual(self.gw.get(a["key"]).state,"SENDING")

if __name__=="__main__":
    unittest.main()
