"""Non-publishing 10X staging HTTP service.

No Gmail credentials, send function, database writes, or user-supplied approval flags.
Only synthetic fixture evaluation is exposed. Defaults to loopback binding.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
import json
import os

from policy import Policy, authorize
from gmail_connector_adapter import read_only_preflight
from datetime import datetime, timezone

SYNTHETIC_SENDER="sender@synthetic.invalid"
SYNTHETIC_RECIPIENT="buyer@synthetic.invalid"
CASES={"empty","prior_sent","inbound_reply","truncated_history","provider_failure"}
MAX_BODY=2048

def decision(case):
    """Fixed, synthetic-only scenarios. No user-provided email/policy evidence."""
    def lookup(**kwargs):
        if case=="provider_failure":
            raise TimeoutError("synthetic failure")
        if case=="truncated_history":
            return {"emails":[],"next_page_token":"missing"}
        if case=="empty":
            return {"emails":[],"next_page_token":None}
        if case=="prior_sent":
            return {"emails":[{"id":"fake-gmail-sent-001","from_":SYNTHETIC_SENDER,
                    "to":[SYNTHETIC_RECIPIENT],"labels":["SENT"],"subject":"Existing synthetic outreach"}],
                    "next_page_token":None}
        return {"emails":[{"id":"fake-gmail-reply-001","from_":SYNTHETIC_RECIPIENT,
                "to":[SYNTHETIC_SENDER],"labels":["INBOX"],"subject":"Re: Synthetic outreach"}],
                "next_page_token":None}
    result=read_only_preflight(lookup, sender_account=SYNTHETIC_SENDER,
        aliases=[SYNTHETIC_RECIPIENT],alias_scope_verified=True,all_mail_verified=True)
    # Server policy denies permission regardless of what a remote caller requests.
    return {"fixture":case,"history":result.kind,
            "existing_id":result.message_id,"dispatch_allowed":False,
            "mode":"synthetic-read-only","reason":result.reason}

class Handler(BaseHTTPRequestHandler):
    server_version="10XStagingReadOnly/1"
    def log_message(self,*args):
        pass # Do not write arbitrary request contents to shared logs.
    def _reply(self,status,data):
        raw=json.dumps(data,sort_keys=True).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type","application/json")
        self.send_header("Cache-Control","no-store")
        self.send_header("X-Content-Type-Options","nosniff")
        self.send_header("Content-Length",str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
    def do_GET(self):
        if urlsplit(self.path).path=="/healthz":
            return self._reply(200,{"status":"ok","mode":"synthetic-read-only","sending_enabled":False})
        if urlsplit(self.path).path=="/readyz":
            return self._reply(200,{"status":"staging-only","gmail_live":False,
                                    "production_db":False,"sending_enabled":False})
        return self._reply(404,{"error":"not_found"})
    def do_POST(self):
        if urlsplit(self.path).path!="/v1/fixture-preflight":
            return self._reply(404,{"error":"not_found"})
        try:
            length=int(self.headers.get("Content-Length","-1"))
            if not 0<=length<=MAX_BODY:
                return self._reply(413,{"error":"invalid_content_length"})
            value=json.loads(self.rfile.read(length))
            if not isinstance(value,dict) or set(value)!={"case"} or value["case"] not in CASES:
                return self._reply(400,{"error":"synthetic_case_only"})
            return self._reply(200,decision(value["case"]))
        except (ValueError,TypeError,json.JSONDecodeError):
            return self._reply(400,{"error":"invalid_json"})
    def do_PUT(self):
        return self._reply(405,{"error":"method_not_allowed"})
    do_PATCH=do_PUT
    do_DELETE=do_PUT

def main():
    host=os.environ.get("HOST","127.0.0.1")
    if host not in ("127.0.0.1","localhost"):
        raise RuntimeError("public listening disabled for synthetic staging service")
    port=int(os.environ.get("PORT","8787"))
    with ThreadingHTTPServer((host,port),Handler) as server:
        server.serve_forever()

if __name__=="__main__":
    main()
