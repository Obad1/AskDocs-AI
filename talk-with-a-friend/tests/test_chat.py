"""No real-user, live-payment, or network calls: anonymous matching regression."""
import os

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import app as talk
import advice
from legal import LEGAL_VERSION


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.delenv("RENDER", raising=False)
    monkeypatch.delenv("TALK_PUBLIC_URL", raising=False)
    for key in ("ADVICE_BOOKINGS_ENABLED", "ADVICE_HOST_AVAILABLE", "PAYSTACK_SECRET_KEY",
                "ADVICE_HOST_SECRET", "ADVICE_PRICE_NGN", "DATABASE_URL"):
        monkeypatch.delenv(key, raising=False)
    talk.strangers.clear()
    talk.waitlist.clear()
    talk.session_rate.clear()
    talk.reports.clear()
    with TestClient(talk.app) as api:
        api._opened_chat_sessions = []
        try:
            yield api
        finally:
            for session in reversed(api._opened_chat_sessions):
                session.__exit__(None, None, None)
    talk.strangers.clear()
    talk.waitlist.clear()
    talk.session_rate.clear()
    talk.reports.clear()


def accepted_policy():
    return {
        "adult_confirmed": True,
        "terms_accepted": True,
        "privacy_acknowledged": True,
        "pledge_accepted": True,
        "safety_acknowledged": True,
        "legal_version": LEGAL_VERSION,
    }


def session(c):
    result = c.post("/api/sessions", json=accepted_policy())
    assert result.status_code == 200, result.text
    return result.json()


def join(c, session_data):
    socket = c.websocket_connect("/ws")
    socket.__enter__()
    c._opened_chat_sessions.append(socket)
    socket.send_json({"type": "join", "session_id": session_data["session_id"],
                      "token": session_data["token"]})
    assert socket.receive_json()["type"] == "ready"
    return socket


def test_age_and_rules_required(client):
    assert client.post("/api/sessions", json={}).status_code == 400
    assert client.post("/api/sessions", json={"adult_confirmed": True}).status_code == 400
    assert client.post("/api/sessions", json=accepted_policy()).status_code == 200


def test_matching_strangers_and_messaging(client):
    first = session(client)
    second = session(client)
    one = join(client, first)
    assert one.receive_json()["type"] == "searching"
    two = join(client, second)
    assert one.receive_json()["type"] == "matched"
    assert two.receive_json()["type"] == "matched"
    one.send_json({"type": "message", "text": "Hello, stranger."})
    assert two.receive_json()["text"] == "Hello, stranger."
    assert one.receive_json()["type"] == "sent"
    two.send_json({"type": "message", "text": "<b>not markup</b>"})
    assert one.receive_json()["text"] == "<b>not markup</b>"
    assert two.receive_json()["type"] == "sent"
    assert not hasattr(talk.strangers[first["session_id"]], "messages")
    one.close()
    assert two.receive_json()["type"] == "partner_left"
    assert two.receive_json()["type"] == "searching"
    two.close()


def test_next_requeues_and_never_repeats_same_stranger(client):
    a,b,c=(session(client) for _ in range(3))
    one=join(client,a)
    assert one.receive_json()["type"]=="searching"
    two=join(client,b)
    assert one.receive_json()["type"]=="matched"
    assert two.receive_json()["type"]=="matched"
    one.send_json({"type":"next"})
    assert two.receive_json()["type"]=="partner_left"
    assert two.receive_json()["type"]=="searching"
    assert one.receive_json()["type"]=="searching"
    three=join(client,c)
    assert three.receive_json()["type"]=="matched"
    assert two.receive_json()["type"]=="matched"
    assert talk.strangers[a["session_id"]].waiting is True
    one.close();two.close();three.close()


def test_report_is_ephemeral_and_blocks_same_peer(client):
    a,b=session(client),session(client)
    one=join(client,a);one.receive_json()
    two=join(client,b);one.receive_json();two.receive_json()
    one.send_json({"type":"report","reason":"harassment"})
    assert two.receive_json()["type"]=="partner_left"
    assert two.receive_json()["type"]=="searching"
    assert one.receive_json()["type"]=="reported"
    assert one.receive_json()["type"]=="searching"
    assert len(talk.reports)==1
    assert talk.reports[0]["reason"]=="harassment"
    assert b["session_id"] in talk.strangers[a["session_id"]].blocked
    one.close();two.close()


def test_no_message_without_match(client):
    a=session(client)
    one=join(client,a);one.receive_json()
    one.send_json({"type":"message","text":"Hello"})
    assert one.receive_json()["type"]=="error"
    one.close()


def test_wrong_bearer_rejected(client):
    a=session(client)
    with client.websocket_connect("/ws") as sock:
        sock.send_json({"type":"join","session_id":a["session_id"],
                        "token":"a-wrong-token-of-sufficient-length-for-test"})
        assert sock.receive_json()["type"]=="error"
        with pytest.raises(WebSocketDisconnect):
            sock.receive_json()


def test_third_session_rate_limit(client):
    for _ in range(4):
        assert session(client)
    assert client.post("/api/sessions",json=accepted_policy()).status_code==429


def test_origin_restricted_when_configured(client,monkeypatch):
    monkeypatch.setenv("TALK_PUBLIC_URL","https://talk-with-a-friend.onrender.com")
    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/ws",headers={"origin":"https://bad.example"}) as sock:
            sock.receive_json()


def test_paid_advice_fail_closed_without_configuration(client):
    config=client.get("/api/advice/config")
    assert config.status_code==200
    assert config.json()["open"] is False
    assert config.json()["therapy_available"] is False
    assert client.post("/api/advice/bookings",json={"email":"test@example.com","adult_confirmed":True}).status_code==503
    assert client.get("/api/advice/host/bookings").status_code==403
    assert client.get("/host").status_code==200
    assert advice.enabled() is False




def test_paystack_amount_and_currency_guards_without_real_charges(monkeypatch):
    reference = 'ADVICE_' + 'a'*24
    row = {'status': 'pending', 'amount_kobo': 1500000}
    # Provider success alone is insufficient when the payment amount differs.
    monkeypatch.setattr(advice, '_paystack', lambda path: {
        'reference': reference, 'status': 'success', 'currency': 'NGN',
        'amount': 1})
    assert advice._verify_payment(reference, row) == {'paid': False, 'status': 'pending'}
    # Correct amount in the wrong currency must also be rejected.
    monkeypatch.setattr(advice, '_paystack', lambda path: {
        'reference': reference, 'status': 'success', 'currency': 'USD',
        'amount': 1500000})
    assert advice._verify_payment(reference, row)['paid'] is False
    # Reference substitution must not unlock a paid chat.
    monkeypatch.setattr(advice, '_paystack', lambda path: {
        'reference': 'ADVICE_' + 'b'*24, 'status': 'success',
        'currency': 'NGN', 'amount': 1500000})
    assert advice._verify_payment(reference, row)['paid'] is False

def test_price_is_public_but_payment_stays_disabled(client, monkeypatch):
    monkeypatch.setenv("ADVICE_PRICE_NGN", "10000")
    result = client.get("/api/advice/config")
    assert result.status_code == 200
    assert result.json()["price_kobo"] == 1000000
    assert result.json()["minutes"] == 30
    assert result.json()["delivery"] == "live_text"
    assert result.json()["open"] is False
    booking = client.post("/api/advice/bookings", json={
        "email": "buyer@example.com", "adult_confirmed": True
    })
    assert booking.status_code == 503


def test_policy_routes_and_version_are_readable(client):
    route = client.get("/api/legal/version")
    assert route.status_code == 200
    assert route.json()["version"] == LEGAL_VERSION
    assert set(route.json()["documents"]) == {
        "/legal/terms", "/legal/privacy", "/legal/pledge",
        "/legal/safety", "/legal/advice",
    }
    for name in ("terms", "privacy", "pledge", "safety", "advice"):
        result = client.get("/legal/" + name)
        assert result.status_code == 200
        assert LEGAL_VERSION in result.text
        assert "DRAFT" in result.text
        assert "Return to Talk with a Friend" in result.text
    assert "Nigeria Data Protection Act 2023" in client.get("/legal/privacy").text
    assert "does not exclude" not in client.get("/legal/terms").text.lower()
    assert client.get("/legal/unknown").status_code == 404


def test_clickwrap_requires_every_acknowledgement_and_current_version(client):
    good = accepted_policy()
    for key in list(good):
        bad = dict(good)
        bad[key] = False if key != "legal_version" else "2020-legacy"
        assert client.post("/api/sessions", json=bad).status_code == 400
    missing = dict(good)
    del missing["privacy_acknowledged"]
    assert client.post("/api/sessions", json=missing).status_code == 400
    result = client.post("/api/sessions", json=good)
    assert result.status_code == 200
    assert result.json()["agreement_version"] == LEGAL_VERSION
    assert result.json()["acceptance_storage"] == "ephemeral"
    record = talk.strangers[result.json()["session_id"]]
    assert record.legal_version == LEGAL_VERSION
    assert record.terms_accepted_at > 0


def test_paid_booking_cannot_skip_clickwrap_even_if_configured(client, monkeypatch):
    monkeypatch.setattr(advice, "enabled", lambda: True)
    assert client.post("/api/advice/bookings", json={"email":"buyer@example.com",
                       "adult_confirmed":True}).status_code == 400
    invalid = accepted_policy()
    invalid.update({"email": "buyer@example.com", "advice_terms_accepted":False})
    assert client.post("/api/advice/bookings",json=invalid).status_code == 400
    invalid["advice_terms_accepted"] = True
    invalid["legal_version"] = "outdated"
    assert client.post("/api/advice/bookings",json=invalid).status_code == 400


def test_static_security_and_health(client):
    assert client.get("/").status_code==200
    assert "Meet a stranger" in client.get("/").text
    assert client.get("/healthz").json()["mode"]=="anonymous-random-match"
    assert client.get("/assets/style.css").status_code==200
    assert client.get("/assets/app.js").status_code==200
    assert client.get("/").headers["Cache-Control"]=="no-store"
