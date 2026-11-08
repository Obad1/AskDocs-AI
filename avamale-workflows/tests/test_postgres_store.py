"""Real PostgreSQL integration and restart tests for synthetic pilot.

Run against a disposable CI Postgres service. Does not use or modify Neon
production/staging databases. The endpoint is never authorized for real files.
"""
from __future__ import annotations

import os
from pathlib import Path
import uuid

import pytest
from fastapi.testclient import TestClient

from product_app.api import create_app
from product_app.postgres_store import PostgresStore

URL = os.getenv("TEST_POSTGRES_URL", "")
ROOT = Path(__file__).resolve().parents[1]
PASSWORD = "Test_Password_Long_2026!"
SECRET = "a-b-c-d-local-test-operator-cookie-signature-key-2026"

pytestmark = pytest.mark.skipif(not URL, reason="Requires disposable TEST_POSTGRES_URL")

def client():
    return TestClient(create_app(database_url=URL, admin_password=PASSWORD,
                                 session_secret=SECRET, allow_leads=False))

def login(c):
    response = c.post("/api/auth/login", json={"password": PASSWORD})
    assert response.status_code == 200, response.text
    return {"X-CSRF-Token": response.json()["csrf"]}

@pytest.mark.parametrize("product", ["AA01", "AA02", "AA03", "AA04"])
def test_review_survives_restart_with_real_postgres(product):
    name = "freight_synthetic.csv" if product == "AA03" else product+".csv"
    files = (
        {"bank": ("bank_synthetic.csv", (ROOT/"samples"/"bank_synthetic.csv").read_bytes(), "text/csv"),
         "ledger": ("ledger_synthetic.csv", (ROOT/"samples"/"ledger_synthetic.csv").read_bytes(), "text/csv")}
        if product == "AA02" else
        {"data": (name, (ROOT/"samples"/name).read_bytes(), "text/csv")}
    )
    with client() as c1:
        auth = login(c1)
        run = c1.post(f"/api/runs/{product}", data={"synthetic":"true"},
                      files=files, headers=auth)
        assert run.status_code == 200, run.text
        ident = run.json()["id"]
        before = c1.get(f"/api/runs/{ident}")
        assert before.status_code == 200
        assert before.json()["synthetic"] is True
        assert before.json()["decisions"]
        updated = c1.post(f"/api/runs/{ident}/findings/0",
                          json={"status":"accepted","note":"Invented source checked",
                                "reviewer":"synthetic CI reviewer"},headers=auth)
        assert updated.status_code == 200, updated.text
    # A new FastAPI instance means new DB connections, not a reused in-memory store.
    with client() as c2:
        assert c2.get("/healthz").json()["ok"] is True
        assert isinstance(c2.app.state.store, PostgresStore)
        login(c2)
        persisted = c2.get(f"/api/runs/{ident}").json()
        assert persisted["decisions"][0]["state"] == "accepted"
        assert persisted["input_hashes"]
        assert c2.get(f"/api/runs/{ident}/history").status_code == 200
        assert len(c2.get(f"/api/runs/{ident}/history").json()) == 2
        assert c2.get(f"/api/runs/{ident}/export.csv").status_code == 200

def test_real_files_blocked_and_not_persisted():
    with client() as c:
        auth=login(c)
        csv=(ROOT/"samples"/"freight_synthetic.csv").read_bytes()
        before=len(c.get("/api/runs").json())
        r=c.post("/api/runs/AA03",data={"synthetic":"false"},
                 files={"data":("real.csv",csv,"text/csv")},headers=auth)
        assert r.status_code == 400
        assert "synthetic" in r.text.lower()
        assert len(c.get("/api/runs").json()) == before

def test_authentication_and_lead_intake_fail_closed():
    with client() as c:
        assert c.get("/api/runs").status_code == 401
        assert c.get("/api/enquiries").status_code == 401
        response=c.post("/api/enquiries",json={
            "product":"AA03","organization":"Invented Co",
            "contact_email":"pilot@example.test",
            "problem":"Fictional comparison only","consent":True,
        })
        assert response.status_code == 503
        auth=login(c)
        assert c.post("/api/runs/AA03",data={"synthetic":"true"}).status_code == 403
        assert c.get("/api/enquiries").status_code == 200

def test_database_transaction_rollback_does_not_leak_partial_run():
    store=PostgresStore(URL)
    identifier=str(uuid.uuid4())
    before=len(store.list_runs(limit=100))
    with pytest.raises(Exception):
        with store.conn() as db:
            db.execute("""INSERT INTO runs
               (id,product,title,created_at,updated_at,input_hashes,results,
                source_names,synthetic) VALUES
               (%s,'AA01','controlled rollback','now','now','{}','[]','{}',1)""",
               (identifier,))
            raise RuntimeError("Intentional rollback")
    assert store.get_run(identifier) is None
    assert len(store.list_runs(limit=100)) == before

def test_public_mode_requires_database_url(monkeypatch):
    monkeypatch.setenv("AVAMALE_PUBLIC","1")
    monkeypatch.setenv("AVAMALE_COOKIE_SECURE","1")
    monkeypatch.delenv("DATABASE_URL",raising=False)
    with pytest.raises(RuntimeError,match="requires DATABASE_URL"):
        create_app(admin_password=PASSWORD,session_secret=SECRET)
