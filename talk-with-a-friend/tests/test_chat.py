"""No real-user, live-payment, or network calls: anonymous matching regression."""
import os

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import app as talk
import advice


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
        yield api
    talk.strangers.clear()
    talk.waitlist.clear()
    talk.session_rate.clear()
    talk.reports.clear()


def session(c):
    result = c.post("/api/sessions", json={"adult_confirmed": True, "rules_accepted": True})
    assert result.status_code == 200, result.text
    return result.json()


def join(c, session_data):
    socket = c.websocket_connect("/ws")
    socket.__enter__()
    socket.send_json({"type": "join", "session_id": session_data["session_id"],
                      "token": session_data["token"]})
    assert socket.receive_json()["type"] == "ready"
    return socket


def test_age_and_rules_required(client):
    assert client.post("/api/sessions", json={}).status_code == 400
    assert client.post("/api/sessions", json={"adult_confirmed": True}).status_code == 400
    assert client.post("/api/sessions", json={"adult_confirmed": True,
                                              "rules_accepted": True}).status_code == 200


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
    assert client.post("/api/sessions",json={"adult_confirmed":True,"rules_accepted":True}).status_code==429


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


def test_static_security_and_health(client):
    assert client.get("/").status_code==200
    assert "Meet a stranger" in client.get("/").text
    assert client.get("/healthz").json()["mode"]=="anonymous-random-match"
    assert client.get("/assets/style.css").status_code==200
    assert client.get("/assets/app.js").status_code==200
    assert client.get("/").headers["Cache-Control"]=="no-store"
