"""Optional paid general-advice sessions. Independent of free stranger matching.

Strictly fail-closed: requires an explicitly published price, host availability,
a persistent PostgreSQL URL, a host secret, and Paystack credentials. No therapy
or other health service offered here.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request as URLRequest, urlopen

from fastapi import HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

import psycopg
from psycopg.rows import dict_row

from legal import LEGAL_VERSION

sessions = {}  # connected paid chat seats, temporary; no chat transcripts


def enabled() -> bool:
    price = os.getenv("ADVICE_PRICE_NGN", "")
    return bool(
        os.getenv("ADVICE_BOOKINGS_ENABLED") == "1"
        and os.getenv("ADVICE_HOST_AVAILABLE") == "1"
        # Legal drafts must NOT be treated as launch-ready merely because the
        # payment keys were accidentally supplied.
        and os.getenv("ADVICE_LEGAL_REVIEW_APPROVED") == "1"
        and os.getenv("ADVICE_REFUND_POLICY_PUBLISHED") == "1"
        and bool(re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", os.getenv("PRIVACY_CONTACT_EMAIL", "")))
        and os.getenv("PAYSTACK_SECRET_KEY", "").startswith(("sk_test_", "sk_live_"))
        and os.getenv("ADVICE_HOST_SECRET", "") and len(os.getenv("ADVICE_HOST_SECRET", "")) >= 32
        and os.getenv("DATABASE_URL", "").startswith(("postgres://", "postgresql://"))
        and price.isdecimal() and 1000 <= int(price) <= 1000000
    )


def get_db():
    uri = os.environ.get("DATABASE_URL", "")
    if not uri.startswith(("postgres://", "postgresql://")):
        raise HTTPException(503, "Persistent bookings database is unavailable")
    return psycopg.connect(uri, row_factory=dict_row, connect_timeout=5)


def ensure_schema():
    with get_db() as db:
        with db.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS talk_advice_bookings (
                    reference TEXT PRIMARY KEY, customer_email TEXT NOT NULL,
                    topic TEXT NOT NULL DEFAULT '', amount_kobo INTEGER NOT NULL,
                    currency TEXT NOT NULL DEFAULT 'NGN',
                    customer_access_hash TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    paid_at TIMESTAMPTZ, status TEXT NOT NULL DEFAULT 'pending',
                    provider_accepted_at TIMESTAMPTZ, completed_at TIMESTAMPTZ,
                    legal_version TEXT,
                    terms_accepted_at TIMESTAMPTZ,
                    privacy_acknowledged_at TIMESTAMPTZ,
                    pledge_accepted_at TIMESTAMPTZ,
                    safety_acknowledged_at TIMESTAMPTZ,
                    advice_terms_accepted_at TIMESTAMPTZ
                )
            """)
            # Safe additive migration for existing empty and pre-release tables.
            for column in ("legal_version TEXT", "terms_accepted_at TIMESTAMPTZ",
                           "privacy_acknowledged_at TIMESTAMPTZ",
                           "pledge_accepted_at TIMESTAMPTZ",
                           "safety_acknowledged_at TIMESTAMPTZ",
                           "advice_terms_accepted_at TIMESTAMPTZ"):
                cursor.execute("ALTER TABLE talk_advice_bookings ADD COLUMN IF NOT EXISTS " + column)
        db.commit()


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf8")).hexdigest()


def _host_allowed(secret: str) -> bool:
    configured = os.getenv("ADVICE_HOST_SECRET", "")
    return len(configured) >= 32 and hmac.compare_digest(secret, configured)


def _paystack(path: str, payload: dict | None = None) -> dict:
    key = os.getenv("PAYSTACK_SECRET_KEY", "")
    if not key.startswith(("sk_test_", "sk_live_")):
        raise HTTPException(503, "Payment service not available")
    body = json.dumps(payload, separators=(",", ":")).encode("utf8") if payload is not None else None
    request = URLRequest(
        "https://api.paystack.co" + path,
        data=body, headers={"Authorization": "Bearer " + key,
                            "Content-Type": "application/json"}, method="POST" if body else "GET",
    )
    try:
        with urlopen(request, timeout=8) as response:
            result = json.load(response)
    except (HTTPError, URLError, TimeoutError, ValueError):
        raise HTTPException(502, "Payment provider unavailable; no payment was confirmed")
    if not result.get("status") or not isinstance(result.get("data"), dict):
        raise HTTPException(502, "Payment provider could not complete this operation")
    return result["data"]


def _booking(reference: str):
    if not re.fullmatch(r"ADVICE_[A-Za-z0-9]{24}", reference or ""):
        raise HTTPException(404, "Booking not found")
    with get_db() as db:
        row = db.execute("SELECT * FROM talk_advice_bookings WHERE reference=%s", (reference,)).fetchone()
    if not row:
        raise HTTPException(404, "Booking not found")
    return row


def _verify_payment(reference: str, row: dict) -> dict:
    if row["status"] not in ("pending", "paid", "accepted", "completed"):
        raise HTTPException(409, "Booking is not payable")
    verified = _paystack("/transaction/verify/" + reference)
    # Never infer success from callback, browser, webhook, or displayed checkout.
    if not (verified.get("reference") == reference and verified.get("status") == "success"
            and verified.get("currency") == "NGN"
            and verified.get("amount") == row["amount_kobo"]):
        return {"paid": False, "status": row["status"]}
    with get_db() as db:
        db.execute("""
            UPDATE talk_advice_bookings
            SET paid_at=COALESCE(paid_at,NOW()),
                status=CASE WHEN status='pending' THEN 'paid' ELSE status END
            WHERE reference=%s AND amount_kobo=%s
        """, (reference, row["amount_kobo"]))
        db.commit()
    return {"paid": True, "status": "paid"}


def install(app):
    @app.get("/api/advice/config")
    def config():
        good = enabled()
        raw_price = os.environ.get("ADVICE_PRICE_NGN", "").strip()
        published = raw_price.isdecimal() and 1000 <= int(raw_price) <= 1000000
        # Display a researched public price even before checkout launches;
        # publishing a price alone must NEVER enable payment collection.
        return {"open": good, "price_kobo": int(raw_price) * 100 if published else None,
                "currency": "NGN", "minutes": 30, "service": "general_advice",
                "delivery": "live_text", "therapy_available": False,
                "legal_version": LEGAL_VERSION}

    @app.post("/api/advice/bookings")
    async def create_booking(request: Request):
        if not enabled():
            raise HTTPException(503, "Advice bookings are not open")
        try:
            data = await request.json()
        except (ValueError, TypeError):
            raise HTTPException(400, "Invalid booking data")
        required = ("adult_confirmed", "terms_accepted", "privacy_acknowledged",
                    "pledge_accepted", "safety_acknowledged", "advice_terms_accepted")
        if (not isinstance(data, dict) or any(data.get(flag) is not True for flag in required)
                or data.get("legal_version") != LEGAL_VERSION):
            raise HTTPException(400, "Current Terms, Privacy, Pledge, Safety and Paid Advice terms must be individually acknowledged")
        email = data.get("email", "")
        topic = data.get("topic", "")
        if not isinstance(email, str) or not re.fullmatch(r"[^@\s]{1,64}@[^@\s]+\.[^@\s]+", email) or len(email)>254:
            raise HTTPException(400, "Valid email required")
        if not isinstance(topic, str) or len(topic)>300:
            raise HTTPException(400, "Topic too long")
        # Reject obvious secrets/health topics rather than soliciting sensitive details.
        topic = topic.strip()
        token = secrets.token_urlsafe(32)
        reference = "ADVICE_" + secrets.token_hex(12)
        price = int(os.environ["ADVICE_PRICE_NGN"]) * 100
        ensure_schema()
        with get_db() as db:
            db.execute("""INSERT INTO talk_advice_bookings
                (reference, customer_email, topic, amount_kobo, customer_access_hash,
                 legal_version, terms_accepted_at, privacy_acknowledged_at,
                 pledge_accepted_at, safety_acknowledged_at, advice_terms_accepted_at)
                VALUES (%s,%s,%s,%s,%s,%s,NOW(),NOW(),NOW(),NOW(),NOW())""",
                (reference,email.lower().strip(),topic,price,_hash(token),LEGAL_VERSION))
            db.commit()
        base = os.getenv("TALK_PUBLIC_URL", "").rstrip("/")
        if not base.startswith("https://"):
            raise HTTPException(503, "Booking callback URL must use HTTPS")
        checkout = _paystack("/transaction/initialize", {
            "email": email.lower().strip(),
            "amount": price,
            "currency": "NGN",
            "reference": reference,
            "callback_url": base + "/?reference=" + reference,
            "metadata": {"service": "general_advice", "minutes": 30},
        })
        auth_url = checkout.get("authorization_url", "")
        if not isinstance(auth_url, str) or not auth_url.startswith("https://checkout.paystack.com/"):
            raise HTTPException(502, "Secure checkout link unavailable")
        return {"checkout_url": auth_url, "reference": reference, "customer_access_token": token,
                "amount_kobo": price, "currency": "NGN"}

    @app.post("/api/advice/bookings/{reference}/verify")
    async def verify_booking(reference: str, request: Request):
        try:
            obj = await request.json()
        except (ValueError, TypeError):
            raise HTTPException(400, "Invalid verification request")
        row = _booking(reference)
        supplied = obj.get("access_token", "") if isinstance(obj,dict) else ""
        if not isinstance(supplied,str) or not hmac.compare_digest(_hash(supplied),row["customer_access_hash"]):
            raise HTTPException(403, "Not authorized for this booking")
        confirmation = _verify_payment(reference,row)
        return {"paid": confirmation["paid"], "status": confirmation["status"],
                "reference": reference, "provider_confirmed": row["provider_accepted_at"] is not None}

    @app.get("/api/advice/host/bookings")
    def host_bookings(request: Request):
        if not _host_allowed(request.headers.get("X-Advice-Host-Secret","")):
            raise HTTPException(403, "Host access denied")
        ensure_schema()
        with get_db() as db:
            rows=db.execute("""SELECT reference,customer_email,topic,amount_kobo,created_at,
                                paid_at,status,provider_accepted_at FROM talk_advice_bookings
                                ORDER BY created_at DESC LIMIT 100""").fetchall()
        return {"bookings":[{**r,"created_at":r["created_at"].isoformat(),
                             "paid_at":r["paid_at"].isoformat() if r["paid_at"] else None,
                             "provider_accepted_at":r["provider_accepted_at"].isoformat() if r["provider_accepted_at"] else None}
                             for r in rows]}

    @app.post("/api/advice/host/bookings/{reference}/accept")
    def host_accept(reference: str, request: Request):
        if not _host_allowed(request.headers.get("X-Advice-Host-Secret","")):
            raise HTTPException(403, "Host access denied")
        row = _booking(reference)
        if not _verify_payment(reference, row)["paid"]:
            raise HTTPException(409, "Payment not verified")
        with get_db() as db:
            db.execute("""UPDATE talk_advice_bookings
                          SET provider_accepted_at=COALESCE(provider_accepted_at,NOW()),status='accepted'
                          WHERE reference=%s AND status IN ('paid','accepted')""",(reference,))
            db.commit()
        return {"accepted":True,"reference":reference}

    @app.websocket("/ws/advice/{reference}")
    async def advice_chat(ws: WebSocket, reference: str):
        configured = os.getenv("TALK_PUBLIC_URL","").rstrip("/")
        if not configured or ws.headers.get("origin")!=configured:
            await ws.close(code=1008)
            return
        await ws.accept()
        seat = ""
        try:
            try:
                hello = await asyncio.wait_for(ws.receive_json(), timeout=12)
            except (asyncio.TimeoutError, ValueError, TypeError):
                await ws.close(code=1008)
                return
            if not isinstance(hello,dict):
                await ws.close(code=1008)
                return
            role,secret = hello.get("role"),hello.get("token")
            if not isinstance(secret,str):
                await ws.close(code=1008)
                return
            row = _booking(reference)
            if row["status"] != "accepted":
                await ws.close(code=1008)
                return
            if role == "host" and _host_allowed(secret):
                seat = "host"
            elif role == "customer" and hmac.compare_digest(_hash(secret),row["customer_access_hash"]):
                seat = "customer"
            else:
                await ws.close(code=1008)
                return
            members=sessions.setdefault(reference,{})
            if seat in members:
                await ws.close(code=1008)
                return
            members[seat]=ws
            await ws.send_json({"type":"joined","role":seat})
            other = members.get("customer" if seat=="host" else "host")
            if other:
                await other.send_json({"type":"connected"})
                await ws.send_json({"type":"connected"})
            history: list[float] = []
            while True:
                raw = await ws.receive_text()
                if len(raw)>2300:
                    continue
                packet=json.loads(raw)
                if not isinstance(packet,dict):continue
                if packet.get("type")=="end":
                    break
                if packet.get("type")!="message":continue
                value=packet.get("text","")
                if not isinstance(value,str) or not 1<=len(value.strip())<=1000:continue
                now=time.monotonic()
                history=[x for x in history if now-x<10]
                if len(history)>=12:continue
                history.append(now)
                counterpart=members.get("customer" if seat=="host" else "host")
                if counterpart:
                    await counterpart.send_json({"type":"message","text":value.strip()})
                    await ws.send_json({"type":"sent","text":value.strip()})
        except (WebSocketDisconnect, RuntimeError, ValueError, TypeError):
            pass
        finally:
            members=sessions.get(reference)
            if members and members.get(seat) is ws:
                members.pop(seat,None)
                peer=members.get("customer" if seat=="host" else "host")
                if peer:
                    try:await peer.send_json({"type":"disconnected"})
                    except (WebSocketDisconnect,RuntimeError,OSError):pass
                if not members:sessions.pop(reference,None)
