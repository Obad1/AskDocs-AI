"""XML file parser."""
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict
from .document_parser import DocumentParser


class XMLParser(DocumentParser):

    def parse(self, file_path: Path) -> Dict[str, str]:
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()

            def _recurse(node, depth=0):
                parts = []
                indent = "  " * depth
                tag = node.tag.split("}")[-1] if "}" in node.tag else node.tag
                text = (node.text or "").strip()
                if text:
                    parts.append(f"{indent}{tag}: {text}")
                else:
                    parts.append(f"{indent}{tag}")
                for child in node:
                    parts.extend(_recurse(child, depth + 1))
                tail = (node.tail or "").strip()
                if tail:
                    parts.append(f"{indent}{tail}")
                return parts

            lines = _recurse(root)
            text = "\n".join(lines)
            return {"text": text, "metadata": {"method": "xml_parse", "root_tag": root.tag}}
        except Exception as e:
            return {"text": f"Error parsing XML: {e}", "metadata": {"method": "error"}}
