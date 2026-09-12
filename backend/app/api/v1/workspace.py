"""Workspace API router (Z §7.2 AskDocsWorkspace).

Provides save/load of the full workspace state and document listing.
State is persisted as JSON under the data root (zero external accounts).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException

from app.core.benchmark import run_benchmark
from app.core.config import get_settings
from app.core.model_registry import get_registry
from app.core.security import (
    LocalOnly,
    ensure_within_root,
    safe_join,
    sanitize_filename,
)
from app.models.domain import AskDocsWorkspace, WorkspaceID
from app.models.schemas import (
    WorkspaceListResponse,
    WorkspaceResponse,
)

router = APIRouter(prefix="/workspace", tags=["workspace"])

_WORKSPACE_FILE = "workspace.json"


def _workspace_path(workspace_id: Optional[str] = None) -> Path:
    settings = get_settings()
    base = Path(settings.data_root)
    base.mkdir(parents=True, exist_ok=True)
    if workspace_id:
        return safe_join(base, "workspaces", sanitize_filename(workspace_id + ".json"))
    return base / _WORKSPACE_FILE


def _load(workspace_id: Optional[str]) -> AskDocsWorkspace:
    path = _workspace_path(workspace_id)
    if not path.exists():
        return AskDocsWorkspace()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return AskDocsWorkspace.model_validate(raw)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Corrupt workspace: {exc}")


def _save(ws: AskDocsWorkspace) -> None:
    path = _workspace_path(ws.workspace_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ws.model_dump_json(indent=2), encoding="utf-8")


@router.get("", response_model=WorkspaceResponse)
async def get_workspace(
    workspace_id: Optional[str] = None, _: str = Depends(LocalOnly)
) -> WorkspaceResponse:
    try:
        return WorkspaceResponse(workspace=_load(workspace_id))
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Workspace read unavailable: {exc}")


@router.put("", response_model=WorkspaceResponse)
async def put_workspace(
    ws: AskDocsWorkspace, _: str = Depends(LocalOnly)
) -> WorkspaceResponse:
    try:
        _save(ws)
        return WorkspaceResponse(workspace=ws)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Workspace write unavailable: {exc}")


@router.get("/documents", response_model=WorkspaceListResponse)
async def list_documents(
    workspace_id: Optional[str] = None, _: str = Depends(LocalOnly)
) -> WorkspaceListResponse:
    try:
        ws = _load(workspace_id)
        return WorkspaceListResponse(documents=ws.documents, workspace_id=ws.workspace_id)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Workspace documents unavailable: {exc}")


@router.post("/benchmark/apply", response_model=WorkspaceResponse)
async def apply_benchmark_tier(
    workspace_id: Optional[str] = None, _: str = Depends(LocalOnly)
) -> WorkspaceResponse:
    """Apply the latest benchmark tier to the workspace engine state."""
    try:
        settings = get_settings()
        bench = run_benchmark(settings.data_root)
        ws = _load(workspace_id)
        reg = get_registry()
        reg.apply_tier(bench.tier)
        ws.engine.tier = bench.tier
        ws.engine.backend = reg._state.backend
        ws.engine.ollama_host = settings.ollama_host
        _save(ws)
        return WorkspaceResponse(workspace=ws)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503, detail=f"Benchmark service unavailable on this instance: {exc}"
        )
