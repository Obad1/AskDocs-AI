"""Plain text file parser."""
from pathlib import Path
from typing import Dict
from .document_parser import DocumentParser

class TextParser(DocumentParser):
    """Parser for plain text files."""
    
    def parse(self, file_path: Path) -> Dict[str, str]:
        """Extract text from plain text file."""
        text = ""
        metadata = {}
        
        try:
            # Try UTF-8 first, then fall back to other encodings
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        text = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if not text.strip():
                text = "❌ No readable text found."
        except Exception as e:
            text = f"❌ Error extracting text: {str(e)}"
        
        return {"text": text.strip(), "metadata": metadata}

