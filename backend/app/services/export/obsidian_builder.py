"""Obsidian vault exporter (spec §4.6). Markdown + YAML frontmatter, zipped so
Obsidian can open the folder natively as a vault. No API.
"""

from __future__ import annotations

import io
import zipfile
from typing import Iterable


def _slug(s: str) -> str:
    out = "".join(c if c.isalnum() or c in " -" else "" for c in s.lower()).strip()
    out = out.replace(" ", "-")[:80] or "untitled"
    return out


def build_vault(
    notes: Iterable[tuple[str, str, list[str]]], vault_name: str = "askdocs-vault"
) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        seen: set[str] = set()
        for i, (title, body, tags) in enumerate(notes):
            base = _slug(title)
            if base in seen:
                base = f"{base}-{i}"
            seen.add(base)
            tag_yaml = "\n".join(f"  - {t}" for t in tags) or "  []"
            fm = f"---\ntitle: {title!r}\ntags:\n{tag_yaml}\nsource: AskDocs AI\n---\n\n"
            z.writestr(f"{vault_name}/{base}.md", fm + f"# {title}\n\n{body}\n")
        z.writestr(
            f"{vault_name}/README.md",
            f"# {vault_name}\n\nExported from AskDocs AI. Open this folder as an Obsidian vault.\n",
        )
    return buf.getvalue()
