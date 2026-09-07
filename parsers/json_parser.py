"""JSON file parser."""
import json
from pathlib import Path
from typing import Dict
from .document_parser import DocumentParser


class JSONParser(DocumentParser):

    def parse(self, file_path: Path) -> Dict[str, str]:
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            text = json.dumps(data, indent=2, ensure_ascii=False)
            return {"text": text, "metadata": {"method": "json_pretty"}}
        except Exception as e:
            return {"text": f"Error parsing JSON: {e}", "metadata": {"method": "error"}}
