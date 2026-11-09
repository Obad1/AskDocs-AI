"""Talk with a Friend: 18+ anonymous, randomly paired, text-only strangers.

No email, phone, contact upload, payments, image upload or chat message storage.
Matching and temporary abuse reports use process memory; Render Free restarts
clear all state. This is not a substitute for an audited moderation service.
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
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).resolve().parent
app = FastAPI(title="Talk with a Friend", docs_url=None, redoc_url=None)
app.mount("/assets", StaticFiles(directory=str(BASE / "static")), name="assets")

SESSION_TTL = 2 * 60 * 60
MAX_USERS = 150
MAX_MESSAGE = 1000
ALLOWED_REPORTS = {"harassment", "sexual", "hate", "threats", "spam", "other"}
state_lock = asyncio.Lock()


@dataclass
class Stranger:
    id: str
    token_hash: str
    created: float = field(default_factory=time.monotonic)
    socket: WebSocket | None = None
    partner: str | None = None
    waiting: bool = False
    blocked: set[str] = field(default_factory=set)
    reported: set[str] = field(default_factory=set)
    switch_times: deque[float] = field(default_factory=deque)


strangers: dict[str, Stranger] = {}
waitlist: deque[str] = deque()
session_rate: dict[str, deque[float]] = defaultdict(deque)
# Short-lived metadata only (NOT a durable abuse-report queue or moderation).
reports: deque[dict] = deque(maxlen=200)


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _origin_allowed(socket: WebSocket) -> bool:
    configured = os.getenv("TALK_PUBLIC_URL", "").strip().rstrip("/")
    if configured:
        return configured.startswith("https://") and socket.headers.get("origin") == configured
    if os.getenv("RENDER"):
        return False
    origin = socket.headers.get("origin", "")
    if not origin:
        return True  # local test client; session token still mandatory
    return origin.rstrip("/") == (
        ("https" if socket.url.scheme == "wss" else "http")
        + "://" + socket.headers.get("host", "")
    )


def _purge_expired() -> None:
    now = time.monotonic()
    for identifier, person in list(strangers.items()):
        if not person.socket and now - person.created > SESSION_TTL:
            person.waiting = False
            strangers.pop(identifier, None)


def _client_key(request: Request) -> str:
    # Best-effort per-node protection only: Render may share reverse-proxy IPs.
    # Never trust raw client-supplied X-Forwarded-For.
    return request.client.host if request.client else "unknown"


def _connected(person: Stranger) -> bool:
    return strangers.get(person.id) is person and person.socket is not None


async def _send(socket: WebSocket | None, payload: dict) -> bool:
    if socket is None:
        return False
    try:
        await asyncio.wait_for(socket.send_json(payload), timeout=3)
        return True
    except (WebSocketDisconnect, RuntimeError, OSError, asyncio.TimeoutError):
        return False


async def _queue_or_match(person: Stranger) -> None:
    """Only pair unrelated, connected, waiting sessions on this single node."""
    outbound = []
    async with state_lock:
        if not _connected(person) or person.partner or person.waiting:
            return
        peer = None
        deferred = []
        for _ in range(len(waitlist)):
            identifier = waitlist.popleft()
            other = strangers.get(identifier)
            if not other or not other.waiting or other.partner or not _connected(other):
                continue
            if other.id == person.id or other.id in person.blocked or person.id in other.blocked:
                deferred.append(identifier)
                continue
            peer = other
            other.waiting = False
            break
        waitlist.extendleft(reversed(deferred))
        if peer:
            person.partner = peer.id
            peer.partner = person.id
            outbound.append((person.socket, {"type": "matched"}))
            outbound.append((peer.socket, {"type": "matched"}))
        else:
            person.waiting = True
            waitlist.append(person.id)
            outbound.append((person.socket, {"type": "searching"}))
    for socket, message in outbound:
        await _send(socket, message)


async def _unpair(person: Stranger, *, requeue: bool, reason: str | None = None) -> None:
    """Atomic unpair and no-repeat matching for both strangers."""
    peer = None
    async with state_lock:
        person.waiting = False
        if person.partner:
            peer = strangers.get(person.partner)
            person.partner = None
            if peer and peer.partner == person.id:
                peer.partner = None
                peer.waiting = False
                person.blocked.add(peer.id)
                peer.blocked.add(person.id)
                if reason and peer.id not in person.reported:
                    person.reported.add(peer.id)
                    reports.append({"at": int(time.time()), "reason": reason,
                                    "reporter_session": person.id, "reported_session": peer.id})
    if peer and _connected(peer):
        await _send(peer.socket, {"type": "partner_left"})
    if reason and _connected(person):
        await _send(person.socket, {"type": "reported",
                                     "message": "Skipped and blocked for this session. Report metadata is temporary; no staff review is guaranteed."})
    if peer and _connected(peer):
        await _queue_or_match(peer)
    if requeue and _connected(person):
        await _queue_or_match(person)


async def _disconnect(person: Stranger, socket: WebSocket) -> None:
    peer = None
    async with state_lock:
        if person.socket is not socket:
            return
        person.socket = None
        person.waiting = False
        if person.partner:
            peer = strangers.get(person.partner)
            person.partner = None
            if peer and peer.partner == person.id:
                peer.partner = None
                peer.waiting = False
                peer.blocked.add(person.id)
                person.blocked.add(peer.id)
    if peer and _connected(peer):
        await _send(peer.socket, {"type": "partner_left"})
        await _queue_or_match(peer)


@app.middleware("http")
async def secure_headers(request: Request, call_next):
    result = await call_next(request)
    result.headers["Cache-Control"] = "no-store"
    result.headers["X-Content-Type-Options"] = "nosniff"
    result.headers["Referrer-Policy"] = "no-referrer"
    result.headers["X-Frame-Options"] = "DENY"
    result.headers["Content-Security-Policy"] = (
        "default-src 'self'; connect-src 'self' ws: wss:; "
        "script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "object-src 'none'; base-uri 'none'; form-action 'self'"
    )
    return result


@app.get("/")
async def home():
    return FileResponse(BASE / "static/index.html")


@app.get("/healthz")
async def health():
    _purge_expired()
    return {"ok": True, "mode": "anonymous-random-match",
            "online": sum(bool(p.socket) for p in strangers.values()),
            "waiting": sum(bool(p.socket) and p.waiting for p in strangers.values()),
            "persisted": False}


@app.post("/api/sessions")
async def start_session(request: Request):
    try:
        payload = await request.json()
    except (TypeError, ValueError):
        raise HTTPException(400, "Confirm the age requirement and chat rules")
    if not isinstance(payload, dict) or payload.get("adult_confirmed") is not True or payload.get("rules_accepted") is not True:
        raise HTTPException(400, "Adults 18+ must agree to the chat rules")
    address = _client_key(request)
    now = time.monotonic()
    async with state_lock:
        attempts = session_rate[address]
        while attempts and now - attempts[0] > 60:
            attempts.popleft()
        for key in list(session_rate):
            if key != address and (not session_rate[key] or now - session_rate[key][-1] > 90):
                del session_rate[key]
        if len(attempts) >= 4:
            raise HTTPException(429, "Too many sessions; please retry in a minute")
        _purge_expired()
        if len(strangers) >= MAX_USERS:
            raise HTTPException(503, "All chat seats are currently occupied")
        attempts.append(now)
        identifier = secrets.token_hex(12)
        raw_token = secrets.token_urlsafe(32)
        strangers[identifier] = Stranger(id=identifier, token_hash=_hash(raw_token))
    return {"session_id": identifier, "token": raw_token, "expires_minutes": 120,
            "stranger_matching": True, "persisted": False}


@app.websocket("/ws")
async def chat(socket: WebSocket):
    if not _origin_allowed(socket):
        await socket.close(code=1008)
        return
    await socket.accept()
    person = None
    try:
        try:
            raw = await asyncio.wait_for(socket.receive_text(), timeout=12)
            if len(raw) > 1500:
                await socket.close(code=1009)
                return
            hello = json.loads(raw)
        except (asyncio.TimeoutError, json.JSONDecodeError, TypeError, ValueError):
            await socket.close(code=1008)
            return
        if not isinstance(hello, dict) or hello.get("type") != "join":
            await socket.close(code=1008)
            return
        identifier, token = hello.get("session_id"), hello.get("token")
        if not isinstance(identifier, str) or not re.fullmatch("[0-9a-f]{24}", identifier) or not isinstance(token, str) or not 32 <= len(token) <= 128:
            await socket.close(code=1008)
            return
        async with state_lock:
            _purge_expired()
            candidate = strangers.get(identifier)
            if (candidate and candidate.socket is None
                    and hmac.compare_digest(candidate.token_hash, _hash(token))
                    and time.monotonic() - candidate.created < SESSION_TTL):
                person = candidate
                person.socket = socket
            else:
                await _send(socket, {"type": "error", "message": "Session expired or in use. Start again."})
                await socket.close(code=1008)
                return
        await _send(socket, {"type": "ready"})
        await _queue_or_match(person)
        message_times: deque[float] = deque()
        last_typing = 0.0
        while True:
            raw = await socket.receive_text()
            if len(raw) > 2300:
                await _send(socket, {"type": "error", "message": "Message too long"})
                continue
            try:
                payload = json.loads(raw)
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            if time.monotonic() - person.created > SESSION_TTL:
                await _send(socket, {"type": "error", "message": "Session expired. Start again."})
                break
            typ = payload.get("type")
            if typ == "message":
                msg = payload.get("text")
                if not isinstance(msg, str):
                    continue
                msg = msg.strip()
                if not 1 <= len(msg) <= MAX_MESSAGE:
                    await _send(socket, {"type": "error", "message": "Message must be 1–1000 characters"})
                    continue
                now = time.monotonic()
                while message_times and now - message_times[0] > 10:
                    message_times.popleft()
                if len(message_times) >= 12:
                    await _send(socket, {"type": "error", "message": "Please slow down"})
                    continue
                message_times.append(now)
                peer = strangers.get(person.partner or "")
                if peer is None or peer.partner != person.id or not peer.socket:
                    await _send(socket, {"type": "error", "message": "No stranger connected"})
                    continue
                await _send(peer.socket, {"type": "message", "text": msg, "at": int(time.time())})
                await _send(socket, {"type": "sent", "at": int(time.time())})
            elif typ == "typing":
                now = time.monotonic()
                if now - last_typing < 0.8:
                    continue
                last_typing = now
                peer = strangers.get(person.partner or "")
                if peer and peer.partner == person.id:
                    await _send(peer.socket, {"type": "typing", "active": bool(payload.get("active"))})
            elif typ in {"next", "report"}:
                now = time.monotonic()
                while person.switch_times and now - person.switch_times[0] > 60:
                    person.switch_times.popleft()
                if len(person.switch_times) >= 8:
                    await _send(socket, {"type": "error", "message": "Too many switches. Retry shortly."})
                    continue
                reason = payload.get("reason") if typ == "report" else None
                if typ == "report" and reason not in ALLOWED_REPORTS:
                    await _send(socket, {"type": "error", "message": "Choose a valid report reason"})
                    continue
                person.switch_times.append(now)
                await _unpair(person, requeue=True, reason=reason)
            elif typ == "leave":
                break
    except (WebSocketDisconnect, RuntimeError, OSError, ValueError, TypeError):
        pass
    finally:
        if person:
            await _disconnect(person, socket)
        try:
            await socket.close(code=1000)
        except (RuntimeError, OSError):
            pass
