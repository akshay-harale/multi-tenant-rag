from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Provider selection (env-only)
    provider_embed: str = Field(default="ollama")   # ollama | openai | google | anthropic | databricks
    provider_chat: str = Field(default="ollama")    # same enum
    embed_model: str = Field(default="nomic-embed-text")  # embedding model name
    chat_model: str = Field(default="llama2")             # chat model name

    # Provider base URLs (only used if provider requires it)
    ollama_base_url: str = Field(default="http://localhost:11434")

    # OpenAI
    openai_api_key: str | None = None

    # Anthropic
    anthropic_api_key: str | None = None

    # Google Gemini
    google_api_key: str | None = None

    # Databricks
    databricks_host: str | None = None
    databricks_token: str | None = None

    # Vector store (Qdrant)
    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_api_key: str | None = Field(default=None)  # if you later secure Qdrant
    qdrant_collection: str = Field(default="documents")

    # Ingestion / retrieval
    embedding_batch_size: int = 64
    chunk_size: int = 800
    chunk_overlap: int = 80
    max_search_k: int = 50       # initial vector recall set
    top_k: int = 8               # final results returned
    min_score_threshold: float | None = None  # set (e.g., 0.5) to filter weak hits

    # Chat / RAG
    enable_streaming: bool = False
    max_context_docs: int = 8

    # Storage
    storage_root: str = "storage"

    # Tenancy
    tenant_id_pattern: str = r"^[a-zA-Z0-9_-]{3,64}$"

    # Postgres (for persistent tenants + conversations)
    pg_host: str = Field(default="localhost")
    pg_port: int = Field(default=5432)
    pg_db: str = Field(default="ragdb")
    pg_user: str = Field(default="raguser")
    pg_password: str = Field(default="ragpass")
    pg_minconn: int = 1
    pg_maxconn: int = 5

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
