"""ODS (OpenDocument Spreadsheet) parser."""
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict
from .document_parser import DocumentParser


class ODSParser(DocumentParser):

    def parse(self, file_path: Path) -> Dict[str, str]:
        try:
            with zipfile.ZipFile(file_path) as z:
                content = z.read("content.xml")
            root = ET.fromstring(content)
            ns = {"table": "urn:oasis:names:tc:opendocument:xmlns:table:1.0",
                  "text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0"}
            tables = root.findall(".//{urn:oasis:names:tc:opendocument:xmlns:table:1.0}table")
            parts = []
            for table in tables:
                name = table.get("{urn:oasis:names:tc:opendocument:xmlns:table:1.0}name") or "Sheet"
                rows = table.findall(".//{urn:oasis:names:tc:opendocument:xmlns:table:1.0}table-row")
                data = []
                for row in rows:
                    cells = []
                    for cell in row.findall("{urn:oasis:names:tc:opendocument:xmlns:table:1.0}table-cell"):
                        p = cell.find("{urn:oasis:names:tc:opendocument:xmlns:text:1.0}p")
                        cells.append("".join(p.itertext()).strip() if p is not None else "")
                    data.append("\t".join(cells))
                parts.append(f"--- Sheet: {name} ---\n" + "\n".join(data))
            text = "\n\n".join(parts)
            return {"text": text, "metadata": {"method": "ods_parse", "tables": len(tables)}}
        except Exception as e:
            return {"text": f"Error parsing ODS: {e}", "metadata": {"method": "error"}}
