from __future__ import annotations
from typing import Sequence, List
import httpx
import logging
from .base import EmbeddingsProvider, InMemoryEmbeddingCache, retry_with_backoff
from app.core.config import get_settings

logger = logging.getLogger(__name__)
_settings = get_settings()

_cache = InMemoryEmbeddingCache()

class OllamaLocalEmbeddings(EmbeddingsProvider):
    def __init__(self, base_url: str | None = None, model: str | None = None, timeout: float = 60.0):
        self.base_url = (base_url or _settings.ollama_base_url).rstrip("/")
        self.model = model or _settings.embed_model
        self.timeout = timeout

    # ------------- Public API -------------

    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        # cache lookup
        cached = _cache.batch_get(self.model, texts)
        to_compute_idx: list[int] = [i for i, c in enumerate(cached) if c is None]
        if to_compute_idx:
            to_compute_texts = [texts[i] for i in to_compute_idx]
            computed = self._batch_embed(to_compute_texts)
            for i, emb in zip(to_compute_idx, computed):
                cached[i] = emb
                _cache.put(self.model, texts[i], emb)
        # type ignore guarded by filling above
        return [c for c in cached if c is not None]  # type: ignore

    def embed_query(self, text: str) -> List[float]:
        cached = _cache.get(self.model, text)
        if cached is not None:
            return cached
        emb = self._single_embed(text)
        _cache.put(self.model, text, emb)
        return emb

    # ------------- Internal -------------

    def _single_embed(self, text: str) -> List[float]:
        def _call():
            return self._request_embedding(text)
        return retry_with_backoff(_call)

    def _batch_embed(self, texts: Sequence[str]) -> List[List[float]]:
        # Ollama embeddings API currently processes one prompt per call.
        # For now loop sequentially; could parallelize with threads if needed.
        out: List[List[float]] = []
        for t in texts:
            out.append(self._single_embed(t))
        return out

    def _request_embedding(self, text: str) -> List[float]:
        url = f"{self.base_url}/api/embeddings"
        payload = {"model": self.model, "prompt": text}
        try:
            with httpx.Client(timeout=self.timeout) as client:
                r = client.post(url, json=payload)
                r.raise_for_status()
                data = r.json()
                if "embedding" not in data:
                    raise RuntimeError(f"Unexpected Ollama response keys: {list(data.keys())}")
                emb = data["embedding"]
                if not isinstance(emb, list):
                    raise RuntimeError("Ollama embedding not a list")
                return emb
        except Exception as e:  # noqa
            logger.error("Ollama embedding request failed: %s", e)
            raise

# Multi-model registry (keyed by base_url + model) for dynamic overrides
_embeddings_registry: dict[str, OllamaLocalEmbeddings] = {}

def get_embeddings_provider(model: str | None = None, base_url: str | None = None) -> OllamaLocalEmbeddings:
    """
    Retrieve (or create) an embeddings provider for the given model/base_url.
    Backwards compatible: if called without arguments returns default model instance.
    """
    effective_base = (base_url or _settings.ollama_base_url).rstrip("/")
    effective_model = model or _settings.embed_model
    key = f"{effective_base}::{effective_model}"
    provider = _embeddings_registry.get(key)
    if provider is None:
        provider = OllamaLocalEmbeddings(base_url=effective_base, model=effective_model)
        _embeddings_registry[key] = provider
    return provider
