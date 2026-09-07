"""Export API router (spec §4.6). All responses are built locally; no third-party
API. Endpoints mirror the frontend exporters.
"""

from __future__ import annotations

from fastapi import APIRouter, Response
from pydantic import BaseModel

from app.services.export import (
    anki_builder,
    obsidian_builder,
    office_builder,
    notion_export,
)

router = APIRouter(prefix="/export", tags=["export"])


class AnkiRequest(BaseModel):
    cards: list[tuple[str, str]]
    deck_name: str = "AskDocs AI Study Deck"


class ObsidianRequest(BaseModel):
    notes: list[tuple[str, str, list[str]]]
    vault_name: str = "askdocs-vault"


class PptxRequest(BaseModel):
    slides: list[tuple[str, list[str]]]


class XlsxRequest(BaseModel):
    rows: list[list[str]]


class NotionRequest(BaseModel):
    pages: list[tuple[str, str]]
    db_rows: list[dict[str, str]] = []


@router.post("/anki")
async def anki(req: AnkiRequest):
    data = anki_builder.build_deck(req.cards, req.deck_name)
    return Response(content=data, media_type="application/octet-stream")


@router.post("/obsidian")
async def obsidian(req: ObsidianRequest):
    data = obsidian_builder.build_vault(req.notes, req.vault_name)
    return Response(content=data, media_type="application/zip")


@router.post("/pptx")
async def pptx(req: PptxRequest):
    data = office_builder.build_pptx(req.slides)
    return Response(content=data, media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation")


@router.post("/xlsx")
async def xlsx(req: XlsxRequest):
    data = office_builder.build_xlsx(req.rows)
    return Response(content=data, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@router.post("/notion")
async def notion(req: NotionRequest):
    data = notion_export.build_bundle(req.pages, req.db_rows)
    return Response(content=data, media_type="application/zip")
