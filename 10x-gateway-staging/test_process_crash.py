"""Real process-termination testing against disposable CI PostgreSQL.

These tests never call Gmail or production. Child process commits state then is
forcibly terminated. A second independent process cannot start a new claim.
"""
import multiprocessing as mp
import os
import unittest
from uuid import uuid4

from gateway import Gateway
from test_postgres_gateway import PostgresGatewayIntegration

def child_claim_and_hang(dsn, key, conn):
    import psycopg
    gw=Gateway(lambda:psycopg.connect(dsn))
    gw.claim(key,"synthetic@test.invalid",key+".invalid","INITIAL","buyer@test.invalid","Synthetic only")
    gw.transition(key,"CHECKING","SENDING")
    conn.send("COMMITTED_SENDING")
    conn.close()
    # Freeze until parent deliberately terminates process.
    import time
    while True:
        time.sleep(60)

def child_retry(dsn,key,result_conn):
    import psycopg
    gw=Gateway(lambda:psycopg.connect(dsn))
    fresh,status=gw.claim(key,"synthetic@test.invalid",key+".invalid","INITIAL","buyer@test.invalid","Synthetic only")
    result_conn.send((fresh,status.state))
    result_conn.close()

@unittest.skipUnless(os.getenv("TEST_GATEWAY_PG_DSN") and os.getenv("GATEWAY_TEST_ISOLATED")=="YES",
                     "explicit isolated Postgres only")
class ProcessCrashRecovery(unittest.TestCase):
    def test_kill_after_send_intent_cannot_reclaim(self):
        dsn=os.environ["TEST_GATEWAY_PG_DSN"]
        key="crash-"+uuid4().hex
        ctx=mp.get_context("spawn")
        parent,child=ctx.Pipe(duplex=False)
        process=ctx.Process(target=child_claim_and_hang,args=(dsn,key,child))
        process.start()
        child.close()
        try:
            self.assertTrue(parent.poll(15),"child did not confirm durable SENDING")
            self.assertEqual(parent.recv(),"COMMITTED_SENDING")
        finally:
            process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
            parent.close()
        self.assertNotEqual(process.exitcode,0)
        read,write=ctx.Pipe(duplex=False)
        p2=ctx.Process(target=child_retry,args=(dsn,key,write))
        p2.start()
        write.close()
        try:
            self.assertTrue(read.poll(15),"retry worker did not respond")
            fresh,state=read.recv()
            self.assertFalse(fresh)
            self.assertEqual(state,"SENDING")
        finally:
            p2.join(timeout=10)
            if p2.is_alive():
                p2.terminate()
                p2.join(timeout=5)
            read.close()
        self.assertEqual(p2.exitcode,0)

if __name__=="__main__":
    unittest.main()
