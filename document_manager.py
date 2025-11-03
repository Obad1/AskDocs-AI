"""Document management and processing."""
import logging
import hashlib
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from parsers.document_parser import DocumentParser
from vectorstores.base import VectorStoreBase
from text_processor import chunk_text
from config import DOCUMENTS_DIR

logger = logging.getLogger(__name__)

class DocumentManager:
    """Manages documents, their storage, and processing."""
    
    def __init__(self, vector_store: VectorStoreBase):
        """Initialize the document manager."""
        self.vector_store = vector_store
        self.documents: Dict[str, Dict] = {}  # doc_id -> document info
    
    def add_document(self, file_path: Path) -> Dict[str, any]:
        """
        Process and add a document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dictionary with document information
        """
        # Get parser
        parser = DocumentParser.get_parser_for_file(file_path)
        if not parser:
            return {"error": f"Unsupported file type: {file_path.suffix}"}
        
        # Parse document
        try:
            result = parser.parse(file_path)
            text = result["text"]
            metadata = result.get("metadata", {})
        except Exception as e:
            logger.error(f"Error parsing document: {e}")
            return {"error": f"Error parsing document: {str(e)}"}
        
        if not text or text.startswith("❌"):
            return {"error": "Could not extract text from document."}
        
        # Generate document ID
        doc_id = self._generate_doc_id(file_path, text)
        
        # Chunk text for vector store
        chunks = chunk_text(text)
        
        # Add to vector store
        doc_metadata = {
            "file_name": file_path.name,
            "file_path": str(file_path),
            "uploaded_at": datetime.now().isoformat(),
            "chunks": len(chunks),
            "text_length": len(text),
            **metadata
        }
        
        self.vector_store.add_document(doc_id, chunks, doc_metadata)
        
        # Store document info
        self.documents[doc_id] = {
            "id": doc_id,
            "name": file_path.name,
            "path": str(file_path),
            "text": text,
            "metadata": doc_metadata,
            "uploaded_at": datetime.now().isoformat()
        }
        
        logger.info(f"Added document: {doc_id} ({file_path.name})")
        
        return self.documents[doc_id]
    
    def _generate_doc_id(self, file_path: Path, text: str) -> str:
        """Generate a unique document ID."""
        # Use file name + content hash
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        file_hash = hashlib.md5(file_path.name.encode()).hexdigest()[:8]
        return f"{file_path.stem}_{file_hash}_{content_hash}"
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """Get document information by ID."""
        return self.documents.get(doc_id)
    
    def list_documents(self) -> List[Dict]:
        """List all documents."""
        return list(self.documents.values())
    
    def delete_document(self, doc_id: str):
        """Delete a document."""
        if doc_id in self.documents:
            self.vector_store.delete_document(doc_id)
            del self.documents[doc_id]
            logger.info(f"Deleted document: {doc_id}")
    
    def get_document_text(self, doc_id: str) -> Optional[str]:
        """Get the full text of a document."""
        doc = self.get_document(doc_id)
        return doc.get("text") if doc else None

