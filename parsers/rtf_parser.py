"""RTF (Rich Text Format) parser."""
import re
from pathlib import Path
from typing import Dict
from .document_parser import DocumentParser


class RTFParser(DocumentParser):

    def parse(self, file_path: Path) -> Dict[str, str]:
        try:
            raw = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            raw = file_path.read_text(encoding="latin-1", errors="replace")

        text = raw
        text = re.sub(r'\{\\rtf1.*?\\ansi', '', text, count=1)
        text = re.sub(r'\\([a-z]+)(-?\d+)?', '', text)
        text = re.sub(r'\{|\}', '', text)
        text = re.sub(r'\\[\'’][0-9a-f]{2}', '', text)
        text = re.sub(r'\\\n', '\n', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return {"text": text, "metadata": {"method": "rtf_strip"}}
