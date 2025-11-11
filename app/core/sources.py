from __future__ import annotations
import uuid
from dataclasses import dataclass
from typing import List, Optional
from app.core.config import get_settings
from app.core.db import fetch_one, fetch_all, execute

_settings = get_settings()

@dataclass(frozen=True)
class SourceContext:
    tenant_id: str
    source_id: str
    source_name: str

@dataclass
class Source:
    source_id: str
    source_name: str
    created_at: str

@dataclass
class Document:
    document_id: str
    filename: str
    file_path: str
    uploaded_at: str

# ------------- Source Management -------------

def create_source(tenant_id: str, source_name: str) -> str:
    """
    Create a new source for a tenant. Returns the source_id.
    """
    if not source_name or not source_name.strip():
        raise ValueError("source_name is required")
    
    source_name = source_name.strip()
    
    # Check if source already exists
    existing = fetch_one(
        "SELECT source_id FROM sources WHERE tenant_id=%s AND source_name=%s",
        tenant_id, source_name
    )
    if existing:
        raise ValueError(f"Source '{source_name}' already exists for this tenant")
    
    source_id = str(uuid.uuid4())
    execute(
        "INSERT INTO sources (tenant_id, source_id, source_name) VALUES (%s, %s, %s)",
        tenant_id, source_id, source_name
    )
    return source_id

def list_sources(tenant_id: str) -> List[Source]:
    """
    List all sources for a tenant.
    """
    rows = fetch_all(
        "SELECT source_id, source_name, created_at FROM sources WHERE tenant_id=%s ORDER BY created_at DESC",
        tenant_id
    )
    return [
        Source(
            source_id=r["source_id"],
            source_name=r["source_name"],
            created_at=str(r["created_at"])
        )
        for r in rows
    ]

def get_source(tenant_id: str, source_id: str) -> Optional[Source]:
    """
    Get a specific source by ID.
    """
    row = fetch_one(
        "SELECT source_id, source_name, created_at FROM sources WHERE tenant_id=%s AND source_id=%s",
        tenant_id, source_id
    )
    if not row:
        return None
    return Source(
        source_id=row["source_id"],
        source_name=row["source_name"],
        created_at=str(row["created_at"])
    )

def get_source_by_name(tenant_id: str, source_name: str) -> Optional[Source]:
    """
    Get a specific source by name.
    """
    row = fetch_one(
        "SELECT source_id, source_name, created_at FROM sources WHERE tenant_id=%s AND source_name=%s",
        tenant_id, source_name
    )
    if not row:
        return None
    return Source(
        source_id=row["source_id"],
        source_name=row["source_name"],
        created_at=str(row["created_at"])
    )

def delete_source(tenant_id: str, source_id: str) -> bool:
    """
    Delete a source and all its documents. Returns True if deleted, False if not found.
    """
    row = fetch_one(
        "SELECT source_id FROM sources WHERE tenant_id=%s AND source_id=%s",
        tenant_id, source_id
    )
    if not row:
        return False
    
    execute(
        "DELETE FROM sources WHERE tenant_id=%s AND source_id=%s",
        tenant_id, source_id
    )
    return True

def get_source_context(tenant_id: str, source_id: str) -> SourceContext:
    """
    Get source context, validating it exists.
    """
    source = get_source(tenant_id, source_id)
    if not source:
        raise ValueError(f"Source '{source_id}' not found for tenant '{tenant_id}'")
    
    return SourceContext(
        tenant_id=tenant_id,
        source_id=source_id,
        source_name=source.source_name
    )

# ------------- Document Tracking -------------

def register_document(
    tenant_id: str,
    source_id: str,
    filename: str,
    file_path: str
) -> str:
    """
    Register a document in a source. Returns document_id.
    """
    document_id = str(uuid.uuid4())
    execute(
        "INSERT INTO documents (tenant_id, source_id, document_id, filename, file_path) VALUES (%s, %s, %s, %s, %s)",
        tenant_id, source_id, document_id, filename, file_path
    )
    return document_id

def list_documents(tenant_id: str, source_id: str) -> List[Document]:
    """
    List all documents in a source.
    """
    rows = fetch_all(
        "SELECT document_id, filename, file_path, uploaded_at FROM documents WHERE tenant_id=%s AND source_id=%s ORDER BY uploaded_at DESC",
        tenant_id, source_id
    )
    return [
        Document(
            document_id=r["document_id"],
            filename=r["filename"],
            file_path=r["file_path"],
            uploaded_at=str(r["uploaded_at"])
        )
        for r in rows
    ]

def count_documents(tenant_id: str, source_id: str) -> int:
    """
    Count documents in a source.
    """
    row = fetch_one(
        "SELECT COUNT(*) as cnt FROM documents WHERE tenant_id=%s AND source_id=%s",
        tenant_id, source_id
    )
    return row["cnt"] if row else 0
