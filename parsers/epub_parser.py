"""EPUB e-book parser."""
import zipfile
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict
from .document_parser import DocumentParser


class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self._text = []
    def handle_data(self, data):
        self._text.append(data)
    def get_text(self):
        return " ".join(self._text)


class EPUBParser(DocumentParser):

    def parse(self, file_path: Path) -> Dict[str, str]:
        try:
            with zipfile.ZipFile(file_path) as z:
                names = z.namelist()
                html_files = [n for n in names if n.endswith((".xhtml", ".html", ".htm"))]

                if not html_files:
                    try:
                        container = ET.fromstring(z.read("META-INF/container.xml"))
                        ns = {"c": "urn:oasis:names:tc:opendocument:xmlns:container"}
                        rootfile = container.find(".//c:rootfile", ns)
                        if rootfile is not None:
                            opf_path = rootfile.get("full-path", "")
                            opf = ET.fromstring(z.read(opf_path))
                            ns_m = {"m": "http://www.idpf.org/2007/opf"}
                            base = str(Path(opf_path).parent) + "/" if Path(opf_path).parent else ""
                            for ref in opf.findall(".//m:itemref", ns_m):
                                idref = ref.get("idref", "")
                                item = opf.find(f".//m:item[@id='{idref}']", ns_m)
                                if item is not None:
                                    href = item.get("href", "")
                                    html_files.append(base + href)
                    except Exception:
                        pass

                chapters = []
                for hf in html_files:
                    try:
                        raw = z.read(hf).decode("utf-8", errors="replace")
                        stripper = _HTMLStripper()
                        stripper.feed(raw)
                        text = stripper.get_text().strip()
                        if text:
                            chapters.append(text)
                    except Exception:
                        pass

            text = "\n\n".join(chapters)
            return {"text": text, "metadata": {"method": "epub_parse", "chapters": len(chapters)}}
        except Exception as e:
            return {"text": f"Error parsing EPUB: {e}", "metadata": {"method": "error"}}
