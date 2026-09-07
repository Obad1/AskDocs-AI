"""Anki deck builder (spec §4.6). genanki produces a real SQLite-backed .apkg
with no Anki desktop app or AnkiConnect API required.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterable

import genanki


def build_deck(
    cards: Iterable[tuple[str, str]], deck_name: str = "AskDocs AI Study Deck"
) -> bytes:
    deck_id = abs(hash(deck_name)) % (2**31)
    model = genanki.Model(
        1607392319,
        "AskDocs Model",
        fields=[{"name": "Front"}, {"name": "Back"}],
        templates=[
            {
                "name": "Card",
                "qfmt": "{{Front}}",
                "afmt": "{{Front}}<hr id=\"answer\">{{Back}}",
            }
        ],
    )
    deck = genanki.Deck(deck_id, deck_name)
    for front, back in cards:
        deck.add_note(genanki.Note(model=model, fields=[front, back]))

    with tempfile.NamedTemporaryFile(suffix=".apkg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        genanki.Package(deck).write_to_file(tmp_path)
        return Path(tmp_path).read_bytes()
    finally:
        Path(tmp_path).unlink(missing_ok=True)
