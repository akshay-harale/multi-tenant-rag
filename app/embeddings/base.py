from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Sequence, List
import hashlib
import threading
import time

class EmbeddingsProvider(ABC):
    """Abstract embeddings provider."""

    @abstractmethod
    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed multiple documents (batch)."""
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query string."""
        raise NotImplementedError


class InMemoryEmbeddingCache:
    """
    Simple thread-safe in-memory cache.
    Key derivation includes model name (provided by caller).
    Not persistent; promote to Redis / disk later if needed.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._store: dict[str, List[float]] = {}

    @staticmethod
    def _hash_key(model: str, text: str) -> str:
        h = hashlib.sha256()
        h.update(model.encode("utf-8"))
        h.update(b"\x00")
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    def get(self, model: str, text: str) -> List[float] | None:
        k = self._hash_key(model, text)
        with self._lock:
            return self._store.get(k)

    def put(self, model: str, text: str, embedding: List[float]) -> None:
        k = self._hash_key(model, text)
        with self._lock:
            self._store[k] = embedding

    def batch_get(self, model: str, texts: Sequence[str]) -> list[List[float] | None]:
        return [self.get(model, t) for t in texts]

    def batch_put(self, model: str, texts: Sequence[str], embeddings: Sequence[List[float]]) -> None:
        for t, e in zip(texts, embeddings):
            self.put(model, t, e)


def retry_with_backoff(fn, *, max_attempts: int = 5, base_delay: float = 0.5, max_delay: float = 8.0):
    """
    Generic retry helper for transient network errors.
    """
    attempt = 0
    while True:
        try:
            return fn()
        except Exception as e:  # noqa
            attempt += 1
            if attempt >= max_attempts:
                raise
            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
            time.sleep(delay)
