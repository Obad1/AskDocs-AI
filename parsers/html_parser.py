"""HTML file parser."""
from pathlib import Path
from typing import Dict
from bs4 import BeautifulSoup
from .document_parser import DocumentParser

class HTMLParser(DocumentParser):
    """Parser for HTML files."""
    
    def parse(self, file_path: Path) -> Dict[str, str]:
        """Extract text from HTML file."""
        text = ""
        metadata = {}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text(separator='\n\n', strip=True)
            
            # Get title if available
            title_tag = soup.find('title')
            if title_tag:
                metadata["title"] = title_tag.get_text()
            
            if not text.strip():
                text = "❌ No readable text found in HTML."
        except Exception as e:
            text = f"❌ Error extracting text from HTML: {str(e)}"
        
        return {"text": text.strip(), "metadata": metadata}

