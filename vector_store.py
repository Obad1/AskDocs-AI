"""Vector store using ChromaDB for local-first storage."""
import logging
from pathlib import Path
from typing import List, Dict, Optional
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from config import VECTOR_DB_PATH, EMBEDDING_MODEL, DEVICE

logger = logging.getLogger(__name__)

class VectorStore:
    """Local vector store using ChromaDB."""
    
    def __init__(self):
        """Initialize the vector store."""
        self.client = chromadb.PersistentClient(
            path=str(VECTOR_DB_PATH),
            settings=Settings(anonymized_telemetry=False)
        )
        self.embedder = SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)
        self.collections: Dict[str, chromadb.Collection] = {}
    
    def get_collection(self, doc_id: str) -> chromadb.Collection:
        """Get or create a collection for a document."""
        if doc_id not in self.collections:
            try:
                self.collections[doc_id] = self.client.get_collection(name=doc_id)
            except:
                self.collections[doc_id] = self.client.create_collection(
                    name=doc_id,
                    metadata={"description": f"Document collection for {doc_id}"}
                )
        return self.collections[doc_id]
    
    def add_document(self, doc_id: str, chunks: List[str], metadata: Optional[Dict] = None):
        """Add document chunks to the vector store."""
        if not chunks:
            logger.warning(f"No chunks to add for document {doc_id}")
            return
        
        collection = self.get_collection(doc_id)
        
        # Generate embeddings
        embeddings = self.embedder.encode(chunks, normalize_embeddings=True)
        
        # Add to collection
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"chunk_index": i, **(metadata or {})} for i in range(len(chunks))]
        
        collection.add(
            embeddings=embeddings.tolist(),
            documents=chunks,
            ids=ids,
            metadatas=metadatas
        )
        
        logger.info(f"Added {len(chunks)} chunks to document {doc_id}")
    
    def search(self, doc_id: str, query: str, top_k: int = 5) -> List[Dict]:
        """Search for similar chunks in a document."""
        collection = self.get_collection(doc_id)
        
        # Generate query embedding
        query_embedding = self.embedder.encode([query], normalize_embeddings=True)[0]
        
        # Search
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k
        )
        
        # Format results
        formatted_results = []
        if results['documents'] and len(results['documents'][0]) > 0:
            for doc, dist, metadata in zip(
                results['documents'][0],
                results['distances'][0],
                results['metadatas'][0]
            ):
                formatted_results.append({
                    "text": doc,
                    "distance": dist,
                    "metadata": metadata
                })
        
        return formatted_results
    
    def delete_document(self, doc_id: str):
        """Delete a document from the vector store."""
        try:
            if doc_id in self.collections:
                self.client.delete_collection(name=doc_id)
                del self.collections[doc_id]
                logger.info(f"Deleted document {doc_id}")
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")

