"""10X gateway staging proof. No production Gmail or send API is exposed."""
from dataclasses import dataclass
from typing import Callable

SCHEMA = "gateway_ci"
DDL = """
CREATE SCHEMA IF NOT EXISTS gateway_ci;
CREATE TABLE IF NOT EXISTS gateway_ci.dispatch (
 idempotency_key TEXT PRIMARY KEY,
 sender TEXT NOT NULL, company TEXT NOT NULL, stage TEXT NOT NULL,
 recipient TEXT NOT NULL, subject TEXT NOT NULL,
 state TEXT NOT NULL DEFAULT 'CHECKING'
   CHECK (state IN ('CHECKING','SENDING','SENT','BLOCKED','UNCERTAIN')),
 gmail_id TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
 UNIQUE(sender,company,stage),
 CONSTRAINT sent_requires_id CHECK ((state='SENT' AND gmail_id IS NOT NULL AND length(btrim(gmail_id))>0)
 OR (state<>'SENT' AND gmail_id IS NULL))
);
CREATE TABLE IF NOT EXISTS gateway_ci.events (
 seq BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 key TEXT NOT NULL REFERENCES gateway_ci.dispatch(idempotency_key),
 state TEXT NOT NULL, gmail_id TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE OR REPLACE FUNCTION gateway_ci.guarded_dispatch() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
 IF TG_OP='INSERT' THEN
   IF NEW.state<>'CHECKING' OR NEW.gmail_id IS NOT NULL THEN RAISE EXCEPTION 'new claim must be CHECKING'; END IF;
 ELSE
   IF OLD.state='SENT' THEN RAISE EXCEPTION 'SENT immutable'; END IF;
   IF OLD.state='SENDING' AND NEW.state NOT IN ('SENT','UNCERTAIN') THEN RAISE EXCEPTION 'SENDING cannot reopen'; END IF;
   IF OLD.state IN ('BLOCKED','UNCERTAIN') AND NEW.state <> 'SENT' THEN RAISE EXCEPTION 'hold cannot reopen'; END IF;
   IF OLD.state='CHECKING' AND NEW.state NOT IN ('SENDING','SENT','BLOCKED','UNCERTAIN') THEN RAISE EXCEPTION 'invalid claim transition'; END IF;
   IF NEW.state='SENT' AND (NEW.gmail_id IS NULL OR length(btrim(NEW.gmail_id))=0) THEN RAISE EXCEPTION 'SENT requires Gmail id'; END IF;
 END IF;
 RETURN NEW;
END $$;
DROP TRIGGER IF EXISTS guarded_dispatch ON gateway_ci.dispatch;
CREATE TRIGGER guarded_dispatch BEFORE INSERT OR UPDATE ON gateway_ci.dispatch
 FOR EACH ROW EXECUTE FUNCTION gateway_ci.guarded_dispatch();
CREATE OR REPLACE FUNCTION gateway_ci.immutable_event() RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN RAISE EXCEPTION 'audit is append only'; END $$;
DROP TRIGGER IF EXISTS immutable_event ON gateway_ci.events;
CREATE TRIGGER immutable_event BEFORE UPDATE OR DELETE ON gateway_ci.events
 FOR EACH ROW EXECUTE FUNCTION gateway_ci.immutable_event();
"""

@dataclass(frozen=True)
class Status:
    state: str
    gmail_id: str | None

class Gateway:
    """Caller supplies isolated psycopg connection factory. No implicit DB or Gmail setup."""
    def __init__(self, connect: Callable):
        self.connect = connect

    def claim(self, key, sender, company, stage, recipient, subject):
        fields=(key,sender,company,stage,recipient,subject)
        if any(not isinstance(v,str) or not v.strip() for v in fields):
            raise ValueError("canonical send identifiers required")
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""INSERT INTO gateway_ci.dispatch
                  (idempotency_key,sender,company,stage,recipient,subject)
                  VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING RETURNING idempotency_key""",fields)
                fresh=cur.fetchone() is not None
                if fresh:
                    cur.execute("INSERT INTO gateway_ci.events(key,state) VALUES (%s,'CHECKING')",(key,))
                cur.execute("""SELECT idempotency_key,recipient,subject,state,gmail_id
                   FROM gateway_ci.dispatch
                   WHERE idempotency_key=%s OR (sender=%s AND company=%s AND stage=%s)""",
                   (key,sender,company,stage))
                hits=cur.fetchall()
            conn.commit()
        if len(hits)!=1 or tuple(hits[0][:3])!=(key,recipient,subject):
            raise ValueError("conflicting sender/company/stage or contact: manual review")
        return fresh,Status(hits[0][3],hits[0][4])

    def transition(self, key, expected, next_state, gmail_id=None):
        """State transition and audit commit atomically. Never creates send retries."""
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("""UPDATE gateway_ci.dispatch SET state=%s,gmail_id=%s
                   WHERE idempotency_key=%s AND state=%s RETURNING idempotency_key""",
                   (next_state,gmail_id,key,expected))
                if cur.fetchone() is None:
                    raise ValueError("stale or missing gateway state")
                cur.execute("INSERT INTO gateway_ci.events(key,state,gmail_id) VALUES (%s,%s,%s)",
                            (key,next_state,gmail_id))
            conn.commit()

    def get(self,key):
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT state,gmail_id FROM gateway_ci.dispatch WHERE idempotency_key=%s",(key,))
                row=cur.fetchone()
        if row is None: raise KeyError(key)
        return Status(*row)

    def run_fake_send(self, *, key, sender, company, stage, recipient, subject, authorize, preflight, fake_send):
        """Synthetic ONLY: blocks uncertain requests; never performs automated retries.
        preflight must return ('UNSENT',None), ('SENT',id), or ('BLOCKED',None).
        A real adapter must prove fully paginated Gmail history and opt-out checks."""
        fresh,old=self.claim(key,sender,company,stage,recipient,subject)
        if not fresh: return old
        try:
            if authorize() is not True:
                self.transition(key,'CHECKING','BLOCKED')
                return self.get(key)
            state,existing=preflight()
            if state=='SENT' and isinstance(existing,str) and existing:
                self.transition(key,'CHECKING','SENT',existing)
                return self.get(key)
            if state!='UNSENT' or existing is not None:
                self.transition(key,'CHECKING','BLOCKED')
                return self.get(key)
            self.transition(key,'CHECKING','SENDING')
            try:
                message_id=fake_send(recipient,subject)
                if not isinstance(message_id,str) or not message_id.strip():
                    raise ValueError('missing message id')
                self.transition(key,'SENDING','SENT',message_id)
            except BaseException:
                try:
                    self.transition(key,'SENDING','UNCERTAIN')
                except Exception:
                    pass  # durable SENDING also prevents automatic retries
                raise
            return self.get(key)
        except BaseException:
            # Fail closed even if authorization/preflight failed unexpectedly.
            try:
                if self.get(key).state=='CHECKING': self.transition(key,'CHECKING','BLOCKED')
            except Exception:
                pass
            raise
