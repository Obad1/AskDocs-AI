from typing import List, Dict, Optional


class VectorStoreBase:
    """Abstract interface for vector stores."""

    def add_document(self, doc_id: str, chunks: List[str], metadata: Optional[Dict] = None):
        raise NotImplementedError

    def search(self, doc_id: str, query: str, top_k: int = 5) -> List[Dict]:
        raise NotImplementedError

    def delete_document(self, doc_id: str):
        raise NotImplementedError


