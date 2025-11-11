from __future__ import annotations
import psycopg
from psycopg.rows import dict_row
from contextlib import contextmanager
from typing import Iterator, Any
import logging
from app.core.config import get_settings
import os
import time

logger = logging.getLogger(__name__)
settings = get_settings()

_POOL: psycopg.Connection | None = None

DDL = """
CREATE TABLE IF NOT EXISTS tenants (
    tenant_id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sources (
    tenant_id TEXT NOT NULL,
    source_id UUID NOT NULL,
    source_name TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (tenant_id, source_id),
    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE,
    UNIQUE (tenant_id, source_name)
);

CREATE TABLE IF NOT EXISTS documents (
    tenant_id TEXT NOT NULL,
    source_id UUID NOT NULL,
    document_id UUID NOT NULL,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    uploaded_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (tenant_id, source_id, document_id),
    FOREIGN KEY (tenant_id, source_id) REFERENCES sources(tenant_id, source_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    tenant_id TEXT NOT NULL,
    session_id UUID NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (tenant_id, session_id),
    FOREIGN KEY (tenant_id) REFERENCES tenants(tenant_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS chat_messages (
    tenant_id TEXT NOT NULL,
    session_id UUID NOT NULL,
    turn_index INT NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('system','user','assistant')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (tenant_id, session_id, turn_index),
    FOREIGN KEY (tenant_id, session_id) REFERENCES chat_sessions(tenant_id, session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_session ON chat_messages(tenant_id, session_id, turn_index);
"""

def _dsn() -> str:
    return (
        f"host={settings.pg_host} "
        f"port={settings.pg_port} "
        f"dbname={settings.pg_db} "
        f"user={settings.pg_user} "
        f"password={settings.pg_password}"
    )

def init_pool(retries: int = 20, delay: float = 1.0) -> None:
    global _POOL
    if _POOL is not None:
        return
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            _POOL = psycopg.connect(_dsn(), autocommit=True)
            logger.info("Postgres connection established.")
            return
        except Exception as e:  # noqa
            last_err = e
            logger.warning("Postgres connect attempt %d failed: %s", attempt, e)
            time.sleep(delay)
    raise RuntimeError(f"Failed to connect to Postgres after {retries} attempts: {last_err}")

def run_migrations() -> None:
    assert _POOL is not None, "Pool not initialized"
    with _POOL.cursor() as cur:
        cur.execute(DDL)
    logger.info("Database migrations ensured.")

@contextmanager
def get_conn() -> Iterator[psycopg.Connection]:
    if _POOL is None:
        init_pool()
    assert _POOL is not None
    yield _POOL

def fetch_all(query: str, *params: Any) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, params)
            return list(cur.fetchall())

def fetch_one(query: str, *params: Any) -> dict | None:
    with get_conn() as conn:
        with conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query, params)
            return cur.fetchone()

def execute(query: str, *params: Any) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)

def execute_returning(query: str, *params: Any) -> Any:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            try:
                return cur.fetchone()
            except Exception:
                return None
