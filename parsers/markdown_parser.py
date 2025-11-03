"""Markdown file parser."""
from pathlib import Path
from typing import Dict
from .document_parser import DocumentParser

class MarkdownParser(DocumentParser):
    """Parser for Markdown files."""
    
    def parse(self, file_path: Path) -> Dict[str, str]:
        """Extract text from Markdown file."""
        text = ""
        metadata = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            if not text.strip():
                text = "❌ No readable text found in Markdown file."
        except Exception as e:
            text = f"❌ Error extracting text from Markdown: {str(e)}"
        
        return {"text": text.strip(), "metadata": metadata}

