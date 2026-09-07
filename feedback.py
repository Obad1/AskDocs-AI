import secrets
import json
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
FEEDBACK_PATH = Path("data/feedback.jsonl")

@dataclass
class FeedbackEntry:
    feature: str
    rating: str
    description: str = ""
    output_excerpt: str = ""
    reference_code: str = field(default_factory=lambda: secrets.token_urlsafe(6).upper()[:8])
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_reviewed: bool = False

def submit(feature: str, rating: str, description: str = "", output_excerpt: str = "") -> str:
    entry = FeedbackEntry(
        feature=feature, rating=rating,
        description=description[:1000],
        output_excerpt=output_excerpt[:500]
    )
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(entry)) + "\n")
    logger.info(f"[Feedback] {entry.reference_code} — {feature} / {rating}")
    return entry.reference_code

def get_by_ref(ref: str) -> dict | None:
    if not FEEDBACK_PATH.exists():
        return None
    with open(FEEDBACK_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                e = json.loads(line)
                if e.get("reference_code") == ref.upper():
                    return e
            except Exception:
                pass
    return None

def get_summary() -> dict:
    if not FEEDBACK_PATH.exists():
        return {"total": 0}
    entries = []
    with open(FEEDBACK_PATH, encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except Exception:
                pass
    by_feature = {}
    by_rating = {}
    for e in entries:
        by_feature[e["feature"]] = by_feature.get(e["feature"], 0) + 1
        by_rating[e["rating"]] = by_rating.get(e["rating"], 0) + 1
    return {
        "total": len(entries),
        "by_feature": by_feature,
        "by_rating": by_rating,
        "unreviewed": sum(1 for e in entries if not e.get("is_reviewed"))
    }
