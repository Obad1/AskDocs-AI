"""PDF document parser."""
import re
from pathlib import Path
from typing import Dict
import pdfplumber
from .document_parser import DocumentParser

class PDFParser(DocumentParser):
    """Parser for PDF files."""
    
    def parse(self, file_path: Path) -> Dict[str, str]:
        """Extract text from PDF file."""
        text = ""
        metadata = {"pages": 0}
        
        try:
            with pdfplumber.open(file_path) as pdf:
                metadata["pages"] = len(pdf.pages)
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        # Clean up whitespace
                        cleaned = re.sub(r'\s+', ' ', extracted).strip()
                        if cleaned:
                            text += cleaned + "\n\n"
            
            if not text.strip():
                text = "❌ No readable text found in PDF."
        except Exception as e:
            text = f"❌ Error extracting text from PDF: {str(e)}"
        
        return {"text": text.strip(), "metadata": metadata}

