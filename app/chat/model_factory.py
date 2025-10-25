from __future__ import annotations
"""
Chat model factory supporting multiple providers via environment configuration.

Settings fields used:
  provider_chat: ollama | openai (extensible)
  chat_model: model name for selected provider
  ollama_base_url
  openai_api_key

Add future providers (google, anthropic, databricks) by extending _build_client.
"""
from typing import List, Dict, Any, Protocol
from app.core.config import get_settings
import httpx
import logging

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI  # lightweight; only used if provider_chat=openai
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore

class ChatClientLike(Protocol):
    def complete(self, messages: List[Dict[str, str]]) -> str: ...

_singletons: dict[str, ChatClientLike] = {}


class OllamaChatClient:
    def __init__(self, base_url: str, model: str, timeout: float = 120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def complete(self, messages: List[Dict[str, str]]) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False
        }
        url = f"{self.base_url}/api/chat"
        with httpx.Client(timeout=self.timeout) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
        if isinstance(data, dict):
            if "message" in data and isinstance(data["message"], dict):
                return data["message"].get("content", "").strip()
            if "response" in data:
                return str(data["response"]).strip()
        return str(data)[:8000]


class OpenAIChatClient:
    def __init__(self, api_key: str, model: str, timeout: float = 120):
        if OpenAI is None:
            raise RuntimeError("openai package not installed (pip install openai)")
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.timeout = timeout

    def complete(self, messages: List[Dict[str, str]]) -> str:
        # Convert to OpenAI chat format; already role/content
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0
        )
        try:
            return resp.choices[0].message.content.strip()
        except Exception:
            return str(resp)[:8000]


def _build_client(settings) -> ChatClientLike:
    prov = settings.provider_chat.lower()
    model = settings.chat_model

    if prov == "ollama":
        return OllamaChatClient(base_url=settings.ollama_base_url, model=model)

    if prov == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY not set but provider_chat=openai")
        return OpenAIChatClient(api_key=settings.openai_api_key, model=model)

    raise ValueError(f"Unsupported chat provider '{prov}'")


def get_chat_client() -> ChatClientLike:
    s = get_settings()
    key = f"{s.provider_chat}:{s.chat_model}"
    inst = _singletons.get(key)
    if inst is None:
        inst = _build_client(s)
        _singletons[key] = inst
        logger.info("Initialized chat provider %s (model=%s)", s.provider_chat, s.chat_model)
    return inst


def chat_complete(messages: List[Dict[str, str]]) -> str:
    """
    Convenience function returning final assistant text for a list of role/content messages.
    """
    client = get_chat_client()
    return client.complete(messages)
