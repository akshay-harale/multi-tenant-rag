from __future__ import annotations
import os
import time
import hashlib
import uuid
from typing import List, Iterable, Tuple
from dataclasses import dataclass
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import get_settings
from app.core.tenancy import TenantContext
from app.vector.base import VectorDocument
from app.vector.qdrant_store import get_vector_store
from app.embeddings.factory import get_embedding_service

settings = get_settings()

# ---------------- Data Structures ----------------

@dataclass
class IngestionStats:
    tenant_id: str
    pdf_files: int
    pages: int
    raw_chunks: int
    new_chunks: int
    skipped_duplicates: int
    elapsed_sec: float

# ---------------- Utilities ----------------

def _hash_text(text: str) -> str:
    h = hashlib.sha256()
    h.update(text.encode("utf-8"))
    return h.hexdigest()

def _normalize_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.strip().splitlines() if line.strip())

def _load_pdf_texts(file_path: str) -> List[Tuple[int, str]]:
    reader = PdfReader(file_path)
    pages: List[Tuple[int, str]] = []
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        txt = _normalize_text(txt)
        if txt:
            pages.append((i, txt))
    return pages

def _chunk_page_texts(pages: List[Tuple[int, str]]) -> List[Tuple[int, int, str]]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    out: List[Tuple[int, int, str]] = []
    for page_num, page_text in pages:
        # splitter works across docs; give list with single element
        chunks = splitter.split_text(page_text)
        for idx, chunk in enumerate(chunks):
            norm = _normalize_text(chunk)
            if not norm:
                continue
            out.append((page_num, idx, norm))
    return out

def _iter_pdf_files(root_dir: str) -> Iterable[str]:
    for entry in os.scandir(root_dir):
        if entry.is_file() and entry.name.lower().endswith(".pdf"):
            yield entry.path

# ---------------- Main Pipeline ----------------

def ingest_directory(tenant: TenantContext, directory: str) -> IngestionStats:
    """
    Ingest all PDF files in a directory into vector store.
    """
    t0 = time.time()
    vector_store = get_vector_store()
    embeddings = get_embedding_service()

    pdf_files = list(_iter_pdf_files(directory))
    total_pages = 0
    all_chunks: List[Tuple[str, int, int, str]] = []  # (source, page, chunk_index, text)

    for pdf in pdf_files:
        pages = _load_pdf_texts(pdf)
        total_pages += len(pages)
        page_chunks = _chunk_page_texts(pages)
        for page_num, chunk_idx, text in page_chunks:
            all_chunks.append((pdf, page_num, chunk_idx, text))

    raw_chunks = len(all_chunks)

    # Prepare docs with metadata & hashes
    docs: List[VectorDocument] = []
    hashes: List[str] = []
    for source, page, cidx, text in all_chunks:
        h = _hash_text(text)
        hashes.append(h)
        # Deterministic UUID v5 based on content hash + page + chunk index for idempotency (Qdrant requires int or UUID)
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{h}:{page}:{cidx}"))
        docs.append(
            VectorDocument(
                id=doc_id,
                text=text,
                metadata={
                    "text": text,
                    "source": os.path.basename(source),
                    "source_path": source,
                    "page": page,
                    "chunk_index": cidx,
                    "hash": h,
                    "created_at": int(time.time())
                }
            )
        )

    # Deduplicate by hash before embedding (save work)
    seen = set()
    final_docs: List[VectorDocument] = []
    for d in docs:
        h = d.metadata["hash"]
        if h in seen:
            continue
        seen.add(h)
        final_docs.append(d)

    # Embed (batch)
    batch_size = settings.embedding_batch_size
    embeddings_out: List[List[float]] = []
    for i in range(0, len(final_docs), batch_size):
        batch_texts = [d.text for d in final_docs[i:i+batch_size]]
        embeddings_out.extend(embeddings.embed_texts(batch_texts))

    # Upsert
    inserted = vector_store.upsert(
        tenant_id=tenant.tenant_id,
        docs=final_docs,
        embeddings=embeddings_out,
        skip_if_exists=True
    )

    stats = IngestionStats(
        tenant_id=tenant.tenant_id,
        pdf_files=len(pdf_files),
        pages=total_pages,
        raw_chunks=raw_chunks,
        new_chunks=inserted,
        skipped_duplicates=raw_chunks - inserted,
        elapsed_sec=round(time.time() - t0, 3)
    )
    return stats

def ingest_single_file(tenant: TenantContext, file_path: str) -> IngestionStats:
    tmp_dir = os.path.dirname(file_path)
    return ingest_directory(tenant, tmp_dir)
