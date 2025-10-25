from __future__ import annotations
"""
Embedding provider factory.

Supported providers (env-driven):
  provider_embed=ollama (default)
  provider_embed=openai

Environment fields (Settings):
  provider_embed
  embed_model
  openai_api_key
  ollama_base_url

Add more providers (google, anthropic, databricks) by extending _build_provider.
"""
from typing import List, Sequence, Protocol
from openai import OpenAI  # only used if provider=openai (ensure package installed)
from app.core.config import get_settings
from app.embeddings.ollama_local import get_embeddings_provider as get_ollama_embeddings
import logging

logger = logging.getLogger(__name__)

class EmbeddingsLike(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> List[List[float]]: ...
    def embed_query(self, text: str) -> List[float]: ...

_singletons: dict[str, EmbeddingsLike] = {}

def _build_provider(settings) -> EmbeddingsLike:
    prov = settings.provider_embed.lower()
    model = settings.embed_model

    if prov == "ollama":
        return get_ollama_embeddings(model=model)

    if prov == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set but provider_embed=openai")
        client = OpenAI(api_key=settings.openai_api_key)

        class OpenAIEmbeddingsAdapter:
            def __init__(self, model: str):
                self.model = model
            def embed_texts(self, texts: Sequence[str]) -> List[List[float]]:
                if not texts:
                    return []
                # OpenAI embeddings API
                resp = client.embeddings.create(model=self.model, input=list(texts))
                # new client returns .data list with .embedding
                return [d.embedding for d in resp.data]
            def embed_query(self, text: str) -> List[float]:
                resp = client.embeddings.create(model=self.model, input=[text])
                return resp.data[0].embedding

        return OpenAIEmbeddingsAdapter(model)

    raise ValueError(f"Unsupported embedding provider '{prov}'")

def get_embedding_service() -> EmbeddingsLike:
    settings = get_settings()
    key = f"{settings.provider_embed}:{settings.embed_model}"
    inst = _singletons.get(key)
    if inst is None:
        inst = _build_provider(settings)
        _singletons[key] = inst
        logger.info("Initialized embedding provider %s (model=%s)", settings.provider_embed, settings.embed_model)
    return inst
