"""Credential-free Notion export (spec §4.6 / §4.7). Produces a .zip of Markdown
pages + a CSV matching Notion's native CSV import schema. No OAuth token / API.
"""

from __future__ import annotations

import csv
import io
import zipfile
from typing import Iterable


def _csv_cell(v: str) -> str:
    v = str(v if v is not None else "")
    return f'"{v.replace(chr(34), chr(34) * 2)}"' if any(c in v for c in ',\n"') else v


def build_bundle(
    pages: Iterable[tuple[str, str]],
    db_rows: list[dict[str, str]],
    bundle_name: str = "askdocs-notion-bundle",
) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for i, (title, md) in enumerate(pages):
            name = (title or f"page-{i + 1}").replace(" ", "-")[:80] or f"page-{i + 1}"
            z.writestr(f"{bundle_name}/pages/{name}.md", f"# {title}\n\n{md}\n")

        cols = ["Name"] + sorted({k for r in db_rows for k in r if k != "Name"})
        lines = [",".join([_csv_cell(c) for c in cols])]
        for r in db_rows:
            lines.append(
                ",".join(_csv_cell(r.get(c, "")) for c in cols)
            )
        z.writestr(f"{bundle_name}/database.csv", "\n".join(lines))
        z.writestr(
            f"{bundle_name}/README.md",
            "# Notion import bundle\n\nImport pages/ as Markdown and database.csv "
            "via Notion's native importer. No credentials required.\n",
        )
    return buf.getvalue()
