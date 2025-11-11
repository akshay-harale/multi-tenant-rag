from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Optional

# ---------- Tenant ----------

class TenantCreateRequest(BaseModel):
    tenant_id: str = Field(..., description="Unique tenant identifier")

class TenantCreateResponse(BaseModel):
    tenant_id: str
    status: str = "created"

class TenantListResponse(BaseModel):
    tenants: List[str]

# ---------- Source ----------

class SourceCreateRequest(BaseModel):
    source_name: str = Field(..., description="Name of the source")

class SourceCreateResponse(BaseModel):
    tenant_id: str
    source_id: str
    source_name: str
    status: str = "created"

class SourceResponse(BaseModel):
    source_id: str
    source_name: str
    created_at: str

class SourceListResponse(BaseModel):
    tenant_id: str
    sources: List[SourceResponse]

class SourceDeleteResponse(BaseModel):
    tenant_id: str
    source_id: str
    status: str = "deleted"

class DocumentResponse(BaseModel):
    document_id: str
    filename: str
    file_path: str
    uploaded_at: str

class DocumentListResponse(BaseModel):
    tenant_id: str
    source_id: str
    documents: List[DocumentResponse]

# ---------- Ingestion ----------

class IngestDirectoryRequest(BaseModel):
    directory: str = Field(..., description="Path containing PDF files")

class IngestionStatsResponse(BaseModel):
    tenant_id: str
    pdf_files: int
    pages: int
    raw_chunks: int
    new_chunks: int
    skipped_duplicates: int
    elapsed_sec: float

# ---------- Search ----------

class SearchRequest(BaseModel):
    query: str
    top_k: int = 8
    score_threshold: Optional[float] = None
    source_ids: Optional[List[str]] = Field(None, description="Optional list of source IDs to filter search")

class SearchHit(BaseModel):
    id: str
    text: str
    score: float
    source: Optional[str] = None
    page: Optional[int] = None
    chunk_index: Optional[int] = None

class SearchResponse(BaseModel):
    tenant_id: str
    query: str
    hits: List[SearchHit]

# ---------- Chat (RAG) ----------

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    top_k: int = 6
    include_history: bool = True
    source_ids: Optional[List[str]] = Field(None, description="Optional list of source IDs to filter search")

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: List[str] = []
    used_chunks: int = 0
    sources: List[str] = []
