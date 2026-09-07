"""Central API router.

Aggregates every v1 sub-router into a single ``APIRouter`` mounted at
``/api/v1`` by ``main.py``.

Sibling routers (ingest, retrieval, study, audio, export) are owned by other
agents and mounted here. Each is imported defensively: if a module is not yet
present (e.g. during isolated test runs) it is skipped and recorded so the
app still boots. They are expected to be present in the integrated build.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.api.v1 import community, workspace

api_router = APIRouter()

# Sibling-owned routers to mount (module -> attribute "router")
_SIBLING_ROUTERS = [
    ("app.api.v1.ingest", "router"),
    ("app.api.v1.retrieval", "router"),
    ("app.api.v1.study", "router"),
    ("app.api.v1.audio", "router"),
    ("app.api.v1.export", "router"),
]

_mounted: list[str] = []
_skipped: list[str] = []


def _try_mount(module_name: str, attr: str) -> None:
    try:
        import importlib

        mod = importlib.import_module(module_name)
        router_obj = getattr(mod, attr)
        api_router.include_router(router_obj)
        _mounted.append(module_name)
    except Exception as exc:  # noqa: BLE001 - defensive at import time
        _skipped.append(f"{module_name} ({exc.__class__.__name__})")


for _mod, _attr in _SIBLING_ROUTERS:
    _try_mount(_mod, _attr)

# Locally-owned routers (always present)
api_router.include_router(workspace.router)
api_router.include_router(community.router)
_mounted.extend(["app.api.v1.workspace", "app.api.v1.community"])


def router_status() -> dict:
    return {"mounted": _mounted, "skipped": _skipped}
