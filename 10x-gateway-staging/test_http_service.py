"""HTTP smoke/failure acceptance checks; no external network or credentials."""
import json
import threading
import unittest
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

from http_service import Handler

class TestHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server=ThreadingHTTPServer(("127.0.0.1",0),Handler)
        cls.thread=threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()
    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2)

    def req(self,method,path,body=None):
        c=HTTPConnection("127.0.0.1",self.server.server_port,timeout=3)
        try:
            h={"Content-Type":"application/json"}
            c.request(method,path,body=body,headers=h)
            response=c.getresponse()
            data=json.loads(response.read().decode())
            return response.status,data
        finally:
            c.close()

    def test_health(self):
        status,d=self.req("GET","/healthz")
        self.assertEqual(status,200)
        self.assertFalse(d["sending_enabled"])
    def test_readiness_is_not_production_ready(self):
        status,d=self.req("GET","/readyz")
        self.assertEqual(status,200)
        self.assertFalse(d["gmail_live"])
        self.assertFalse(d["production_db"])
    def test_synthetic_cases_never_enable_dispatch(self):
        for case,expected in [("empty","UNSENT"),("prior_sent","SENT"),("inbound_reply","BLOCKED"),
                              ("truncated_history","BLOCKED"),("provider_failure","BLOCKED")]:
            with self.subTest(case=case):
                status,d=self.req("POST","/v1/fixture-preflight",json.dumps({"case":case}))
                self.assertEqual(status,200)
                self.assertEqual(d["history"],expected)
                self.assertFalse(d["dispatch_allowed"])
    def test_attacker_supplied_approval_flags_rejected(self):
        status,d=self.req("POST","/v1/fixture-preflight",json.dumps({"case":"empty","user_authorized":True}))
        self.assertEqual(status,400)
    def test_unknown_cases_rejected(self):
        self.assertEqual(self.req("POST","/v1/fixture-preflight",json.dumps({"case":"live"}))[0],400)
    def test_missing_body_rejected(self):
        self.assertEqual(self.req("POST","/v1/fixture-preflight","")[0],400)
    def test_oversized_body_rejected(self):
        self.assertEqual(self.req("POST","/v1/fixture-preflight","x"*3000)[0],413)
    def test_no_send_endpoint(self):
        self.assertEqual(self.req("POST","/v1/send",json.dumps({}))[0],404)
        self.assertEqual(self.req("PUT","/v1/fixture-preflight",json.dumps({"case":"empty"}))[0],405)
        self.assertEqual(self.req("DELETE","/v1/send")[0],405)
    def test_unknown_path(self):
        self.assertEqual(self.req("GET","/secret")[0],404)

if __name__=="__main__":
    unittest.main()
