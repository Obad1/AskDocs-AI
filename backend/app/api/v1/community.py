"""Community API router (stub).

Read-only, local-first sharing. Generates a shareable public link that points
to an exported workspace snapshot on the local server. No third-party
accounts, no external upload. This is intentionally minimal but valid.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from app.core.config import get_settings
from app.core.security import LocalOnly, sanitize_filename
from app.models.schemas import ExportRequest, ExportResponse

router = APIRouter(prefix="/community", tags=["community"])


@router.post("/share", response_model=ExportResponse)
async def create_share_link(
    req: ExportRequest, _: str = Depends(LocalOnly)
) -> ExportResponse:
    """Create a local, read-only share snapshot of a workspace.

    The returned ``file_path`` is a path under the exports directory that can
    be served by the backend. No data leaves the machine.
    """
    settings = get_settings()
    exports = Path(settings.exports_dir)
    exports.mkdir(parents=True, exist_ok=True)

    token = hashlib.sha256(
        (str(req.workspace_id or "default") + req.deck_name).encode()
    ).hexdigest()[:16]
    filename = sanitize_filename(f"share-{token}.json")
    path = exports / filename
    snapshot = {
        "workspace_id": req.workspace_id,
        "format": req.format.value,
        "deck_name": req.deck_name,
        "shared_at": json.dumps(None),
    }
    path.write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return ExportResponse(
        file_path=str(path), format=req.format, size_bytes=path.stat().st_size
    )


@router.get("/share/{token}")
async def get_shared_workspace(token: str, _: str = Depends(LocalOnly)) -> dict:
    """Resolve a previously created local share link (read-only)."""
    settings = get_settings()
    exports = Path(settings.exports_dir)
    safe_token = sanitize_filename(token)
    path = exports / sanitize_filename(f"share-{safe_token}.json")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Shared workspace not found.")
    return json.loads(path.read_text(encoding="utf-8"))
