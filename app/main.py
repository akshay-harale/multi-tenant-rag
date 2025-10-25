from __future__ import annotations
import logging
from fastapi import FastAPI, HTTPException, Header, Depends, UploadFile, File
from fastapi.responses import RedirectResponse, FileResponse
from typing import Optional, List
from app.core.config import get_settings
from app.core.tenancy import (
    register_tenant,
    list_tenants,
    get_tenant_context,
    authorize_api_key,
)
from app.models.dto import (
    TenantCreateRequest,
    TenantCreateResponse,
    TenantListResponse,
    IngestDirectoryRequest,
    IngestionStatsResponse,
    SearchRequest,
    SearchResponse,
    SearchHit,
    ChatRequest,
    ChatResponse,
)
from app.ingestion.pipeline import ingest_directory
from app.vector.qdrant_store import get_vector_store
from app.chat.service import rag_chat
from app.embeddings.factory import get_embedding_service
from app.core.db import init_pool, run_migrations
from fastapi.staticfiles import StaticFiles
import os
from app.ingestion.pipeline import ingest_single_file

logger = logging.getLogger("app")
logging.basicConfig(level=logging.INFO)

settings = get_settings()
app = FastAPI(title="Multi-tenant RAG API", version="0.1.0")

# ---------------- Dependencies ----------------

def get_api_key(x_api_key: Optional[str] = Header(default=None)) -> Optional[str]:
    return x_api_key

def tenant_guard(tenant_id: str, api_key: Optional[str] = Depends(get_api_key)):
    # If an API key is supplied, authorize it (in-memory mapping for now)
    if api_key:
        try:
            authorize_api_key(api_key, tenant_id)
        except PermissionError as e:
            raise HTTPException(status_code=403, detail=str(e))
    try:
        return get_tenant_context(tenant_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# ---------------- Tenant Endpoints ----------------

@app.post("/tenants", response_model=TenantCreateResponse)
def create_tenant(req: TenantCreateRequest):
    try:
        register_tenant(req.tenant_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return TenantCreateResponse(tenant_id=req.tenant_id)

@app.get("/tenants", response_model=TenantListResponse)
def list_all_tenants():
    tenants = sorted(list_tenants())
    return TenantListResponse(tenants=tenants)

# ---------------- Ingestion & Upload ----------------

@app.post("/tenants/{tenant_id}/ingest", response_model=IngestionStatsResponse)
def ingest_directory_endpoint(
    tenant_id: str,
    req: IngestDirectoryRequest,
    tenant = Depends(tenant_guard),
):
    stats = ingest_directory(tenant, req.directory)
    return IngestionStatsResponse(**stats.__dict__)

@app.post("/tenants/{tenant_id}/upload", response_model=IngestionStatsResponse)
async def upload_file_endpoint(
    tenant_id: str,
    file: UploadFile = File(...),
    tenant = Depends(tenant_guard),
):
    settings = get_settings()
    upload_dir = os.path.join(settings.storage_root, "uploads", tenant.tenant_id)
    os.makedirs(upload_dir, exist_ok=True)
    target_path = os.path.join(upload_dir, file.filename)
    raw = await file.read()
    with open(target_path, "wb") as f:
        f.write(raw)

    ext = file.filename.lower()
    if ext.endswith(".pdf"):
        stats = ingest_single_file(tenant, target_path)
    elif ext.endswith(".txt"):
        from app.vector.qdrant_store import get_vector_store
        from app.vector.base import VectorDocument
        import time, hashlib, uuid

        text_content = raw.decode("utf-8", errors="ignore")
        if not text_content.strip():
            raise HTTPException(status_code=400, detail="Empty text file.")

        # Chunk text similar to PDF pipeline
        chunk_size = settings.chunk_size
        overlap = settings.chunk_overlap
        chunks: list[str] = []
        start = 0
        while start < len(text_content):
            end = min(start + chunk_size, len(text_content))
            seg = text_content[start:end].strip()
            if seg:
                chunks.append(seg)
            if end == len(text_content):
                break
            start = end - overlap

        embeddings = get_embedding_service()
        vs = get_vector_store()
        docs: list[VectorDocument] = []
        for idx, chunk in enumerate(chunks):
            h = hashlib.sha256(chunk.encode("utf-8")).hexdigest()
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{h}:{idx}"))
            docs.append(
                VectorDocument(
                    id=doc_id,
                    text=chunk,
                    metadata={
                        "text": chunk,
                        "source": file.filename,
                        "source_path": target_path,
                        "page": 0,
                        "chunk_index": idx,
                        "hash": h,
                        "created_at": int(time.time())
                    }
                )
            )
        # Deduplicate
        final_docs: list[VectorDocument] = []
        seen = set()
        for d in docs:
            h = d.metadata["hash"]
            if h in seen:
                continue
            seen.add(h)
            final_docs.append(d)
        embeddings_out = embeddings.embed_texts([d.text for d in final_docs])
        inserted = vs.upsert(
            tenant_id=tenant.tenant_id,
            docs=final_docs,
            embeddings=embeddings_out,
            skip_if_exists=True
        )
        from app.ingestion.pipeline import IngestionStats
        stats = IngestionStats(
            tenant_id=tenant.tenant_id,
            pdf_files=0,
            pages=1,
            raw_chunks=len(docs),
            new_chunks=inserted,
            skipped_duplicates=len(docs) - inserted,
            elapsed_sec=0.0
        )
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type (only .pdf or .txt).")

    return IngestionStatsResponse(**stats.__dict__)

# ---------------- Search ----------------

@app.post("/tenants/{tenant_id}/search", response_model=SearchResponse)
def search_endpoint(
    tenant_id: str,
    req: SearchRequest,
    tenant = Depends(tenant_guard),
):
    embeddings = get_embedding_service()
    vector_store = get_vector_store()

    emb = embeddings.embed_query(req.query)
    results = vector_store.search(
        tenant_id=tenant.tenant_id,
        embedding=emb,
        top_k=min(req.top_k, settings.max_search_k),
        score_threshold=req.score_threshold
    )

    hits: List[SearchHit] = []
    for r in results:
        hits.append(
            SearchHit(
                id=r.id,
                text=r.text,
                score=r.score,
                source=r.metadata.get("source"),
                page=r.metadata.get("page"),
                chunk_index=r.metadata.get("chunk_index"),
            )
        )

    return SearchResponse(
        tenant_id=tenant.tenant_id,
        query=req.query,
        hits=hits
    )

# ---------------- Chat (RAG) ----------------

@app.post("/tenants/{tenant_id}/chat", response_model=ChatResponse)
def chat_endpoint(
    tenant_id: str,
    req: ChatRequest,
    tenant = Depends(tenant_guard),
):
    data = rag_chat(
        tenant_id=tenant.tenant_id,
        user_message=req.message,
        session_id=req.session_id,
        top_k=req.top_k,
        include_history=req.include_history
    )
    return ChatResponse(**data)

# ---------------- Health ----------------

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------------- Startup ----------------

@app.on_event("startup")
def on_startup():
    # Initialize Postgres + run migrations
    init_pool()
    run_migrations()
    # Mount UI (served at /ui/)
    if not any(r.path == "/ui" for r in app.routes):
        if os.path.isdir("web"):
            app.mount("/ui", StaticFiles(directory="web", html=True), name="ui")
    logger.info("API startup complete (db + vector ready + ui).")

# ---------------- UI Routes ----------------
# Serve index directly at root to avoid redirect loops; rely on StaticFiles for /ui/ assets.
@app.get("/", include_in_schema=False)
def root_index():
    return FileResponse("web/index.html")

"""
Run with:
  uvicorn app.main:app --reload

Example flow:
  curl -X POST http://localhost:8000/tenants -H "Content-Type: application/json" -d '{"tenant_id":"acme"}'
  curl -X POST http://localhost:8000/tenants/acme/ingest -H "Content-Type: application/json" -d '{"directory":"data"}'
  curl -X POST http://localhost:8000/tenants/acme/search -H "Content-Type: application/json" -d '{"query":"invoice"}'
"""
