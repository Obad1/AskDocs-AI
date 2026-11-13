"""Local-only security helpers.

The API is intended for loopback use, not as a substitute for authentication.
Forwarding headers are rejected because this application has no configured
trusted-proxy boundary. Filesystem helpers resolve paths before checking that
they remain beneath the configured data root.
"""
from __future__ import annotations

import ipaddress
import os
import re
import unicodedata
from pathlib import Path, PureWindowsPath
from typing import Optional

from fastapi import HTTPException, Request, status

from app.core.config import get_settings

_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._\-]+")
_FORWARDING_HEADERS = ("forwarded", "x-forwarded-for", "x-real-ip")


def sanitize_filename(name: str, max_length: int = 200) -> str:
    """Return a filesystem-safe filename preserving its extension."""
    if max_length < 1:
        raise ValueError("max_length must be positive")
    if not name:
        return "untitled"
    name = os.path.basename(name)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = _SAFE_FILENAME_RE.sub("_", name).strip("._") or "untitled"
    if len(name) > max_length:
        stem, dot, ext = name.rpartition(".")
        if dot and len(ext) + 1 < max_length:
            name = stem[: max_length - len(ext) - 1] + dot + ext
        else:
            name = name[:max_length]
    return name


def _data_root() -> Path:
    return Path(get_settings().data_root).resolve()


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _bad_path() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Path is outside the allowed data root or contains an invalid component.",
    )


def safe_join(base: str | Path, *parts: str) -> Path:
    """Resolve *parts* below a base directory inside the configured data root.

    Absolute paths, drive-qualified paths, separators inside a component, and
    traversal (including symlink escapes) are rejected.
    """
    root = _data_root()
    base_path = Path(base).resolve()
    if not _inside(base_path, root):
        raise _bad_path()

    target = base_path
    for part in parts:
        if (
            not isinstance(part, str)
            or not part
            or part in {".", ".."}
            or "\x00" in part
            or "/" in part
            or "\\" in part
            or Path(part).is_absolute()
            or PureWindowsPath(part).drive
        ):
            raise _bad_path()
        target = (target / part).resolve()
        if not _inside(target, base_path) or not _inside(target, root):
            raise _bad_path()
    return target


def ensure_within_root(path: str | Path) -> Path:
    """Resolve *path* and ensure it is within the configured data root."""
    root = _data_root()
    resolved = Path(path).resolve()
    if not _inside(resolved, root):
        raise _bad_path()
    return resolved


class LocalOnlyAuth:
    """Allow only direct loopback peers; never infer identity from headers.

    Forwarded headers are not accepted: support for a reverse proxy must be
    deliberately configured with an explicit trusted-proxy policy.
    """

    def __init__(self, require_local: bool = True) -> None:
        self.require_local = require_local

    async def __call__(self, request: Request) -> Optional[str]:
        if not self.require_local:
            return "local"

        if any(request.headers.get(name) for name in _FORWARDING_HEADERS):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Forwarded requests are not supported by this local-only service.",
            )

        host = request.client.host if request.client else None
        try:
            address = ipaddress.ip_address(host) if host else None
        except ValueError:
            address = None
        if address is None or not address.is_loopback:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Remote access denied: local-only deployment.",
            )
        return "local"


# Convenience dependency instances.
LocalOnly = LocalOnlyAuth()
RequireLocal = LocalOnlyAuth(require_local=True)
