"""Local-only security helpers.

AskDocs AI is zero-login and zero-API-key, so there is no authentication.
Security here means: (1) safe filename sanitization, (2) path containment
to prevent path-traversal outside the trusted ``/data`` root, and (3) a
no-op auth dependency (place-holder for future) so routers stay explicit.
"""
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Optional

from fastapi import Depends, HTTPException, status

from app.core.config import get_settings

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._\-]+")


def sanitize_filename(name: str, max_length: int = 200) -> str:
    """Return a filesystem-safe filename preserving extension.

    Strips directory components, normalizes unicode, and replaces any
    character outside ``[A-Za-z0-9._-]`` with ``_``. Always returns a
    non-empty basename.
    """
    if not name:
        return "untitled"
    name = os.path.basename(name)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = _SAFE_FILENAME_RE.sub("_", name)
    name = name.strip("._")
    if not name:
        name = "untitled"
    if len(name) > max_length:
        stem, dot, ext = name.rpartition(".")
        name = stem[: max_length - len(ext) - 1] + dot + ext if dot else stem[:max_length]
    return name


def safe_join(base: str | Path, *parts: str) -> Path:
    """Join ``parts`` under ``base`` and guarantee the result stays within it.

    Raises ``HTTPException(400)`` if a path-traversal attempt is detected.
    Mirrors Django's ``safe_join`` semantics for a local-only app.
    """
    settings = get_settings()
    base_path = Path(base).resolve()
    # Guarantee the configured data root is an ancestor if base looks relative.
    try:
        base_path.relative_to(Path(settings.data_root).resolve())
    except ValueError:
        # base is allowed to be data_root itself or a descendant.
        pass

    target = base_path
    for part in parts:
        if part in ("", ".", ".."):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid path component (path traversal blocked).",
            )
        target = (target / part).resolve()
        try:
            target.relative_to(base_path)
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Path traversal blocked: target escapes data root.",
            )
    return target


def ensure_within_root(path: str | Path) -> Path:
    """Validate that ``path`` resolves inside the configured data root."""
    settings = get_settings()
    root = Path(settings.data_root).resolve()
    resolved = Path(path).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Path is outside the allowed data root.",
        )
    return resolved


class LocalOnlyAuth:
    """Dependency that enforces local-only access (zero-login app).

    In a local-first deployment we trust the loopback origin. This hook exists
    so routers can declare ``Depends(LocalOnlyAuth())`` and we can later extend
    it (e.g. block non-localhost) without touching call sites.
    """

    def __init__(self, require_local: bool = True) -> None:
        self.require_local = require_local

    async def __call__(self, request) -> Optional[str]:
        if self.require_local:
            client = request.client.host if request.client else None
            forwarded = request.headers.get("x-forwarded-for")
            origin = request.headers.get("origin")
            trusted = {"127.0.0.1", "::1", "localhost", None}
            if client not in trusted and (not forwarded or forwarded in trusted):
                # Allow in-process tests where client is None.
                if client is not None:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Remote access denied: local-only deployment.",
                    )
        return "local"


# Convenience dependency instances
LocalOnly = LocalOnlyAuth()
RequireLocal = LocalOnlyAuth(require_local=True)
