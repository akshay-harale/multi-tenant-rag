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

class ChatResponse(BaseModel):
    session_id: str
    answer: str
    citations: List[str] = []
    used_chunks: int = 0
    sources: List[str] = []
