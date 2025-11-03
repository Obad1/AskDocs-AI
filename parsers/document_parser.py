"""Base document parser interface."""
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional

class DocumentParser(ABC):
    """Base class for document parsers."""
    
    @abstractmethod
    def parse(self, file_path: Path) -> Dict:
        """
        Parse a document and return extracted text and metadata.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with 'text' and 'metadata' keys
        """
        pass
    
    @staticmethod
    def get_parser_for_file(file_path: Path) -> Optional['DocumentParser']:
        """Get the appropriate parser for a file based on its extension."""
        from .pdf_parser import PDFParser
        from .docx_parser import DOCXParser
        from .pptx_parser import PPTXParser
        from .text_parser import TextParser
        from .markdown_parser import MarkdownParser
        from .csv_parser import CSVParser
        from .html_parser import HTMLParser
        
        ext = file_path.suffix.lower()
        parsers = {
            '.pdf': PDFParser,
            '.docx': DOCXParser,
            '.pptx': PPTXParser,
            '.ppt': PPTXParser,
            '.txt': TextParser,
            '.md': MarkdownParser,
            '.markdown': MarkdownParser,
            '.csv': CSVParser,
            '.html': HTMLParser,
            '.htm': HTMLParser,
        }
        
        parser_class = parsers.get(ext)
        if parser_class:
            return parser_class()
        return None

