from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Dict, Any, Optional, Tuple

@dataclass
class VectorDocument:
    id: str
    text: str
    metadata: Dict[str, Any]

@dataclass
class SearchResult:
    id: str
    text: str
    score: float
    metadata: Dict[str, Any]

class VectorStore(ABC):
    """Abstract vector store interface (multi-tenant safe wrapper)."""

    @abstractmethod
    def ensure_collections(self) -> None:
        """Create underlying collection(s) or indexes if missing."""
        raise NotImplementedError

    @abstractmethod
    def upsert(
        self,
        tenant_id: str,
        docs: Sequence[VectorDocument],
        embeddings: Sequence[Sequence[float]],
        skip_if_exists: bool = True,
    ) -> int:
        """
        Insert or update documents (idempotent).
        Returns number of newly inserted docs (skipped duplicates not counted).
        """
        raise NotImplementedError

    @abstractmethod
    def search(
        self,
        tenant_id: str,
        embedding: Sequence[float],
        top_k: int,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Vector similarity search restricted to tenant."""
        raise NotImplementedError

    @abstractmethod
    def delete_tenant(
        self,
        tenant_id: str,
    ) -> int:
        """Delete all docs for a tenant. Returns number deleted."""
        raise NotImplementedError

    @abstractmethod
    def count_tenant(self, tenant_id: str) -> int:
        """Count docs for tenant."""
        raise NotImplementedError

    @abstractmethod
    def doc_exists_hash(
        self,
        tenant_id: str,
        content_hashes: Sequence[str],
    ) -> Dict[str, bool]:
        """Return map of content_hash -> exists for given tenant."""
        raise NotImplementedError

# Utility to chunk sequences (used by concrete implementations)
def chunk_iter(iterable: Iterable[Any], size: int) -> Iterable[List[Any]]:
    batch: List[Any] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
