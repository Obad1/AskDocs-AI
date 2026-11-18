"""Durable, minimal anonymous-abuse queue with authenticated human triage.

No chat messages, names, IPs, emails or session identifiers are written to Postgres.
Hosting this module does not imply reports are continuously staffed or legally certified.
"""
from __future__ import annotations

import hmac
import os
import re
import logging
from datetime import datetime
from fastapi import HTTPException, Request
import psycopg
from psycopg.rows import dict_row

log = logging.getLogger("talk.friend.safety")
STATUSES = {"new", "reviewing", "resolved", "dismissed"}
RECEIPT = re.compile(r"R-[0-9A-F]{10}\Z")
REASONS = {"harassment", "sexual", "hate", "threats", "spam", "other"}


def database_ready() -> bool:
    """Configured connection, not a claim of active moderation."""
    return os.environ.get("REPORT_DATABASE_URL", "").startswith(("postgresql://", "postgres://"))


def moderator_configured() -> bool:
    return len(os.environ.get("TALK_MODERATOR_SECRET", "")) >= 32


def public_matching_enabled() -> bool:
    # Local tests historically run without RENDER and TALK_PUBLIC_URL.
    production = bool(os.environ.get("RENDER") or os.environ.get("TALK_PUBLIC_URL"))
    default = "0" if production else "1"
    return (os.environ.get("TALK_PUBLIC_MATCHING_ENABLED", default) == "1"
            and (not production or (database_ready() and moderator_configured())))


def connect():
    if not database_ready():
        raise RuntimeError("Report storage not provisioned")
    return psycopg.connect(os.environ["REPORT_DATABASE_URL"], row_factory=dict_row, connect_timeout=5)


def storage_healthy() -> bool:
    if not database_ready():
        return False
    try:
        with connect() as db:
            db.execute("SELECT 1 FROM talk_friend_reports LIMIT 1").fetchone()
        return True
    except (psycopg.Error, OSError) as exc:
        log.error("report_storage_health_failure type=%s", type(exc).__name__)
        return False


def persist_report(receipt: str, reason: str) -> bool:
    """Returns True only after committed durable storage; no raw chat data."""
    if not RECEIPT.fullmatch(receipt) or reason not in REASONS:
        raise ValueError("Invalid report")
    if not database_ready():
        return False
    try:
        with connect() as db:
            db.execute(
                "INSERT INTO talk_friend_reports (receipt,reason) VALUES (%s,%s) ON CONFLICT DO NOTHING",
                (receipt, reason),
            )
            db.execute(
                "INSERT INTO talk_friend_moderation_events (receipt,action) VALUES (%s,'created')",
                (receipt,),
            )
            db.commit()
        return True
    except (psycopg.Error, OSError) as exc:
        log.error("report_persistence_failure type=%s", type(exc).__name__)
        return False


def authorize(request: Request):
    secret = os.environ.get("TALK_MODERATOR_SECRET", "")
    supplied = request.headers.get("X-Talk-Moderator-Secret", "")
    if len(secret) < 32 or not hmac.compare_digest(supplied, secret):
        raise HTTPException(403, "Moderator authorization required")


def install(app):
    @app.get("/api/safety/status")
    def safety_status():
        return {"matching_open": public_matching_enabled(),
                "reports_persistent_configured": database_ready(),
                "moderation_access_configured": moderator_configured(),
                "age_verification": "self_attested_only",
                "moderation": "manual_unstaffed",
                "therapy_available": False}

    @app.get("/api/moderation/reports")
    def queue(request: Request, status: str = "new"):
        authorize(request)
        if status not in STATUSES and status != "all":
            raise HTTPException(400, "Unknown status")
        if not database_ready():
            raise HTTPException(503, "Report database unavailable")
        try:
            with connect() as db:
                if status == "all":
                    rows = db.execute(
                        "SELECT receipt,reason,status,created_at,reviewed_at,review_note "
                        "FROM talk_friend_reports ORDER BY created_at DESC LIMIT 100"
                    ).fetchall()
                else:
                    rows = db.execute(
                        "SELECT receipt,reason,status,created_at,reviewed_at,review_note "
                        "FROM talk_friend_reports WHERE status=%s ORDER BY created_at LIMIT 100",
                        (status,),
                    ).fetchall()
        except (psycopg.Error, OSError) as exc:
            log.error("moderation_read_failure type=%s", type(exc).__name__)
            raise HTTPException(503, "Report queue unavailable")
        return {"reports": [{k:(v.isoformat() if isinstance(v,datetime) else v) for k,v in row.items()}
                            for row in rows], "continuous_monitoring": False}

    @app.post("/api/moderation/reports/{receipt}/review")
    async def update_report(receipt: str, request: Request):
        authorize(request)
        if not RECEIPT.fullmatch(receipt):
            raise HTTPException(404, "No such receipt")
        try:
            body = await request.json()
        except (TypeError, ValueError):
            raise HTTPException(400, "Invalid review")
        if not isinstance(body,dict) or body.get("status") not in (STATUSES - {"new"}):
            raise HTTPException(400, "Choose reviewing, resolved or dismissed")
        note = body.get("note", "")
        if not isinstance(note,str) or len(note) > 500:
            raise HTTPException(400, "Note exceeds limit")
        if not database_ready():
            raise HTTPException(503, "Report database unavailable")
        try:
            with connect() as db:
                result = db.execute(
                    "UPDATE talk_friend_reports SET status=%s, review_note=%s, reviewed_at=NOW() "
                    "WHERE receipt=%s RETURNING receipt",
                    (body["status"],note.strip(),receipt),
                ).fetchone()
                if not result:
                    raise HTTPException(404, "Report not found")
                db.execute("INSERT INTO talk_friend_moderation_events(receipt,action) VALUES(%s,%s)",
                           (receipt,body["status"]))
                db.commit()
        except HTTPException:
            raise
        except (psycopg.Error, OSError) as exc:
            log.error("moderation_write_failure type=%s", type(exc).__name__)
            raise HTTPException(503, "Could not update report")
        return {"receipt": receipt, "status": body["status"], "updated": True}
