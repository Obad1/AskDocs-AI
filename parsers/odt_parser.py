"""ODT (OpenDocument Text) parser."""
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict
from .document_parser import DocumentParser


class ODTParser(DocumentParser):

    def parse(self, file_path: Path) -> Dict[str, str]:
        try:
            with zipfile.ZipFile(file_path) as z:
                content = z.read("content.xml")
            root = ET.fromstring(content)
            ns = {"text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0"}
            paragraphs = []
            for p in root.iter("{urn:oasis:names:tc:opendocument:xmlns:text:1.0}p"):
                text = "".join(p.itertext())
                if text.strip():
                    paragraphs.append(text.strip())
            text = "\n\n".join(paragraphs)
            return {"text": text, "metadata": {"method": "odt_parse", "paragraphs": len(paragraphs)}}
        except Exception as e:
            return {"text": f"Error parsing ODT: {e}", "metadata": {"method": "error"}}
