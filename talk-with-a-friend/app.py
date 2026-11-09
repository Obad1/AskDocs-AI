"""Talk with a Friend: free, invite-only two-person ephemeral chat.

No AI model, account provider, payment provider or durable message storage.
Rooms and messages disappear when the server process restarts. This is an
intentional privacy and free-tier trade-off, NOT production-grade availability.
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import secrets
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, urlsplit

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

BASE = Path(__file__).resolve().parent
app = FastAPI(title="Talk with a Friend", docs_url=None, redoc_url=None)
app.mount("/assets", StaticFiles(directory=str(BASE / "static")), name="assets")

MAX_ROOMS = 150
MAX_AGE_SECONDS = 12 * 60 * 60
MAX_MESSAGE = 1000
_room_lock = asyncio.Lock()
rooms: dict[str, "Room"] = {}
creation_rate = defaultdict(deque)


@dataclass
class Room:
    id: str
    host_digest: str
    guest_digest: str
    created: float = field(default_factory=time.monotonic)
    activity: float = field(default_factory=time.monotonic)
    connected: dict[str, WebSocket] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)
    ended: bool = False


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _same(secret: str, digest: str) -> bool:
    return hmac.compare_digest(_digest(secret), digest)


def _purge_expired() -> None:
    now = time.monotonic()
    for room_id, room in list(rooms.items()):
        if room.ended or now - room.created > MAX_AGE_SECONDS:
            rooms.pop(room_id, None)


def _client_addr(req: Request) -> str:
    # Ignore spoofable forwarded IP headers; Render's actual upstream addresses
    # may group several users, so this is best-effort abuse throttling.
    return req.client.host if req.client else "unknown"


def _public_origin(request: Request) -> str:
    configured = os.getenv("TALK_PUBLIC_URL", "").strip().rstrip("/")
    if configured:
        if not configured.startswith("https://"):
            raise HTTPException(503, "TALK_PUBLIC_URL must use HTTPS")
        return configured
    # Local development only. Production should always configure TALK_PUBLIC_URL.
    if os.getenv("RENDER"):
        raise HTTPException(503, "Public site URL not configured")
    return str(request.base_url).rstrip("/")


@app.middleware("http")
async def harden(request: Request, call_next):
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; connect-src 'self' ws: wss:; "
        "script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "object-src 'none'; base-uri 'self'; form-action 'none'"
    )
    return response


@app.get("/")
def home():
    return FileResponse(BASE / "static/index.html")


@app.get("/r/{room_id}")
def room_page(room_id: str):
    if len(room_id) > 50:
        raise HTTPException(404)
    return FileResponse(BASE / "static/index.html")


@app.get("/healthz")
def health():
    _purge_expired()
    return {"ok": True, "mode": "ephemeral-invite-chat", "active_rooms": len(rooms)}


@app.post("/api/rooms")
async def create_room(request: Request):
    address = _client_addr(request)
    now = time.monotonic()
    async with _room_lock:
        attempts = creation_rate[address]
        while attempts and now - attempts[0] > 60:
            attempts.popleft()
        if len(attempts) >= 5:
            raise HTTPException(429, "Too many rooms created; try again later")
        attempts.append(now)
        _purge_expired()
        if len(rooms) >= MAX_ROOMS:
            raise HTTPException(503, "Chat is at capacity; please retry")
        room_id = secrets.token_hex(7)
        host_key = secrets.token_urlsafe(24)
        invite_key = secrets.token_urlsafe(24)
        rooms[room_id] = Room(id=room_id, host_digest=_digest(host_key),
                              guest_digest=_digest(invite_key))
    origin = _public_origin(request)
    return {
        "room_id": room_id, "host_key": host_key,
        "invite_url": f"{origin}/r/{room_id}?invite={quote(invite_key, safe='')}",
        "expires_minutes": 720, "persisted": False,
    }


@app.get("/api/rooms/{room_id}")
def room_info(room_id: str):
    _purge_expired()
    room = rooms.get(room_id)
    if not room:
        raise HTTPException(404, "This room has expired or was lost on restart")
    return {"room_id": room.id, "connected": len(room.connected),
            "expires_minutes": max(0, int((MAX_AGE_SECONDS - (time.monotonic() - room.created)) / 60)),
            "persisted": False}


async def _safe_send(socket: WebSocket, payload: dict) -> bool:
    try:
        await socket.send_json(payload)
        return True
    except (WebSocketDisconnect, RuntimeError, OSError):
        return False


async def _other(room: Room, seat: str, payload: dict):
    recipient = room.connected.get("guest" if seat == "host" else "host")
    if recipient:
        await _safe_send(recipient, payload)


@app.websocket("/ws/{room_id}")
async def chat(socket: WebSocket, room_id: str):
    # Browser WebSockets carry Origin. Deny third-party origins to reduce
    # drive-by token abuse; do not rely on Origin as the sole authorization.
    allowed = os.getenv("TALK_PUBLIC_URL", "").strip().rstrip("/")
    origin = socket.headers.get("origin", "")
    if allowed and origin != allowed:
        await socket.close(code=1008)
        return
    await socket.accept()
    seat = None
    room = None
    try:
        try:
            hello = await asyncio.wait_for(socket.receive_json(), timeout=12)
        except (asyncio.TimeoutError, ValueError, TypeError):
            await socket.close(code=1008)
            return
        if not isinstance(hello, dict) or hello.get("type") != "join":
            await socket.close(code=1008)
            return
        raw = hello.get("token")
        if not isinstance(raw, str) or not 24 <= len(raw) <= 128:
            await socket.close(code=1008)
            return
        clean_name = hello.get("name", "")
        if not isinstance(clean_name, str):
            clean_name = ""
        clean_name = clean_name.strip()[:24] or "Friend"
        async with _room_lock:
            _purge_expired()
            room = rooms.get(room_id)
            if not room:
                await _safe_send(socket, {"type": "error", "message": "Room expired. Start a new one."})
                await socket.close(code=1008)
                return
            seat = ("host" if _same(raw, room.host_digest) else
                    "guest" if _same(raw, room.guest_digest) else None)
            if seat is None or seat in room.connected:
                await _safe_send(socket, {"type": "error", "message": "Invite invalid or seat already occupied."})
                await socket.close(code=1008)
                return
            room.connected[seat] = socket
            room.names[seat] = clean_name
            room.activity = time.monotonic()
            count = len(room.connected)
        await _safe_send(socket, {"type": "joined", "seat": seat, "connected": count})
        await _other(room, seat, {"type": "presence", "online": True, "name": clean_name})
        await _safe_send(socket, {"type": "presence", "online": count == 2,
                                  "name": room.names.get("guest" if seat == "host" else "host", "Friend")})
        sent_at = deque()
        while True:
            packet = await socket.receive_json()
            if not isinstance(packet, dict):
                continue
            kind = packet.get("type")
            if kind == "message":
                message = packet.get("text")
                if not isinstance(message, str):
                    continue
                message = message.strip()
                if not message or len(message) > MAX_MESSAGE:
                    await _safe_send(socket, {"type": "error", "message": "Messages must be 1–1000 characters."})
                    continue
                now = time.monotonic()
                while sent_at and now - sent_at[0] > 10:
                    sent_at.popleft()
                if len(sent_at) >= 20:
                    await _safe_send(socket, {"type": "error", "message": "Sending too quickly. Try again shortly."})
                    continue
                sent_at.append(now)
                if room.ended:
                    await _safe_send(socket, {"type": "error", "message": "This chat has ended."})
                    continue
                if len(room.connected) != 2:
                    await _safe_send(socket, {"type": "error", "message": "Waiting for your friend to join."})
                    continue
                room.activity = now
                await _other(room, seat, {"type": "message", "text": message,
                                          "name": clean_name, "at": int(time.time())})
                await _safe_send(socket, {"type": "sent", "at": int(time.time())})
            elif kind == "typing":
                await _other(room, seat, {"type": "typing", "active": bool(packet.get("active"))})
            elif kind == "end" and seat == "host":
                async with _room_lock:
                    room.ended = True
                    rooms.pop(room.id, None)
                await _other(room, seat, {"type": "ended"})
                await _safe_send(socket, {"type": "ended"})
                await socket.close(code=1000)
                return
            elif kind == "leave":
                return
    except (WebSocketDisconnect, RuntimeError, ValueError, TypeError):
        pass
    finally:
        if room and seat:
            async with _room_lock:
                # Avoid disconnecting a newly reconnected seat.
                if room.connected.get(seat) is socket:
                    room.connected.pop(seat, None)
                    await _other(room, seat, {"type": "presence", "online": False})
