"""Opt-in structured beta reviews, without storing chat transcripts or free text.

Events go to provider application logs with a short receipt ID. Those logs have
the provider's retention policy and are NOT a durable support-ticket system.
"""
from __future__ import annotations

import logging
import secrets
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

logger = logging.getLogger("talk.friend.feedback")
ALLOWED_ISSUES = {"match", "buttons", "layout", "safety", "advice", "other"}
ALLOWED_STAGES = {"home", "waiting", "matched", "reconnecting", "advice", "other"}
_rate: dict[str, deque[float]] = defaultdict(deque)


def install(app):
    @app.post("/api/feedback")
    async def submit_feedback(request: Request):
        # Deliberately no identity, transcript, names, emails or free-form text.
        try:
            payload = await request.json()
        except (ValueError, TypeError):
            raise HTTPException(400, "Choose an issue and rating")
        if not isinstance(payload, dict):
            raise HTTPException(400, "Invalid feedback")
        issue = payload.get("issue")
        stage = payload.get("stage", "other")
        rating = payload.get("rating")
        if (issue not in ALLOWED_ISSUES or stage not in ALLOWED_STAGES
                or type(rating) is not int or not 1 <= rating <= 5):
            raise HTTPException(400, "Choose an issue, a rating from 1–5, and a valid stage")
        address = request.client.host if request.client else "unknown"
        now = time.monotonic()
        bucket = _rate[address]
        while bucket and now - bucket[0] > 3600:
            bucket.popleft()
        for key in list(_rate):
            if key != address and (not _rate[key] or now - _rate[key][-1] > 3700):
                del _rate[key]
        if len(bucket) >= 6:
            raise HTTPException(429, "Feedback limit reached. Please try again later")
        bucket.append(now)
        reference = "TF-" + secrets.token_hex(5).upper()
        # Structured only, no identifying data; staff can inspect Render logs.
        logger.warning("beta_feedback receipt=%s issue=%s rating=%d stage=%s",
                       reference, issue, rating, stage)
        return {"received": True, "receipt": reference,
                "note": "Thanks. This beta logs anonymous ratings and issue categories temporarily; it cannot guarantee follow-up."}
