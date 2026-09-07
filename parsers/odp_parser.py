"""ODP (OpenDocument Presentation) parser."""
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict
from .document_parser import DocumentParser


class ODPParser(DocumentParser):

    def parse(self, file_path: Path) -> Dict[str, str]:
        try:
            with zipfile.ZipFile(file_path) as z:
                content = z.read("content.xml")
            root = ET.fromstring(content)
            ns = {"text": "urn:oasis:names:tc:opendocument:xmlns:text:1.0",
                  "draw": "urn:oasis:names:tc:opendocument:xmlns:drawing:1.0"}
            pages = root.findall(".//{urn:oasis:names:tc:opendocument:xmlns:drawing:1.0}page")
            slides = []
            for page in pages:
                title = page.get("{urn:oasis:names:tc:opendocument:xmlns:text:1.0}p") or "Slide"
                texts = []
                for p in page.iter("{urn:oasis:names:tc:opendocument:xmlns:text:1.0}p"):
                    t = "".join(p.itertext()).strip()
                    if t:
                        texts.append(t)
                slides.append(f"--- {title} ---\n" + "\n".join(texts))
            text = "\n\n".join(slides)
            return {"text": text, "metadata": {"method": "odp_parse", "slides": len(slides)}}
        except Exception as e:
            return {"text": f"Error parsing ODP: {e}", "metadata": {"method": "error"}}
