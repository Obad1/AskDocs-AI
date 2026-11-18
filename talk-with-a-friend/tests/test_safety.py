"""Fail-closed release and safety regression, without real users or payments."""
import os

import pytest
from fastapi.testclient import TestClient

import app as talk
import safety


@pytest.fixture()
def production_hold(monkeypatch):
    monkeypatch.setenv("RENDER", "true")
    monkeypatch.setenv("TALK_PUBLIC_URL", "https://talk-with-a-friend.onrender.com")
    monkeypatch.delenv("TALK_PUBLIC_MATCHING_ENABLED", raising=False)
    monkeypatch.delenv("REPORT_DATABASE_URL", raising=False)
    monkeypatch.delenv("TALK_MODERATOR_SECRET", raising=False)
    return TestClient(talk.app)


def test_prod_matching_closed_without_ready_safety(production_hold):
    assert safety.public_matching_enabled() is False
    status = production_hold.get("/api/safety/status").json()
    assert status["matching_open"] is False
    assert status["age_verification"] == "self_attested_only"
    assert status["moderation"] == "manual_unstaffed"
    assert production_hold.get("/api/match/status").json()["availability"] == "safety_hold"
    denied = production_hold.post("/api/sessions", json={"adult_confirmed":True,"rules_accepted":True})
    assert denied.status_code == 503
    assert "safety" in denied.text.lower()


def test_moderation_requires_actual_secret(production_hold):
    assert production_hold.get("/api/moderation/reports").status_code == 403
    assert production_hold.post("/api/moderation/reports/R-0000000000/review",
                                json={"status":"resolved"}).status_code == 403


def test_legal_pages_visible_as_pages_not_fake_certifications(production_hold):
    for path, expected in [
        ("/privacy", "Privacy notice"), ("/community", "Community rules"),
        ("/safety", "Safety and reporting")]:
        response = production_hold.get(path)
        assert response.status_code == 200
        assert expected in response.text
    assert "/privacy" in production_hold.get("/").text
    assert "/community" in production_hold.get("/").text


def test_both_config_and_health_required_before_prod_joins(monkeypatch):
    monkeypatch.setenv("RENDER","true")
    monkeypatch.setenv("TALK_PUBLIC_URL","https://talk-with-a-friend.onrender.com")
    monkeypatch.setenv("TALK_PUBLIC_MATCHING_ENABLED", "1")
    monkeypatch.setenv("REPORT_DATABASE_URL", "postgresql://unused:unused@example.invalid/talkfriend")
    monkeypatch.setenv("TALK_MODERATOR_SECRET", "a"*40)
    monkeypatch.setattr(talk,"storage_healthy",lambda:False)
    with TestClient(talk.app) as client:
        assert safety.public_matching_enabled() is True
        assert client.post("/api/sessions",
                           json={"adult_confirmed":True,"rules_accepted":True}).status_code==503
