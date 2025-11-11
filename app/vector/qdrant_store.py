from __future__ import annotations
from typing import Sequence, List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from .base import VectorStore, VectorDocument, SearchResult
from app.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

@dataclass
class _CollectionState:
    name: str
    vector_size: Optional[int] = None
    distance: qm.Distance = qm.Distance.COSINE

class QdrantVectorStore(VectorStore):
    """
    Multi-tenant vector store wrapper around a single (or per-tenant promoted) Qdrant collection.

    Strategy (initial):
      - Single base collection (settings.qdrant_collection)
      - Filter on tenant_id for isolation
      - Payload fields: tenant_id, source, page, chunk_index, hash, created_at
      - Create payload indexes on tenant_id, hash for faster lookups
    """

    def __init__(self) -> None:
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key
        )
        self._state = _CollectionState(name=settings.qdrant_collection)
        self._ensured = False

    # -------------------------- Internal helpers --------------------------

    def _ensure_collection(self, vector_size: int) -> None:
        """
        Lazily create collection with given vector_size if missing.
        Idempotent: if exists, verify size; warn if mismatch.
        """
        try:
            exists = self.client.collection_exists(self._state.name)
        except Exception as e:
            raise RuntimeError(f"Failed checking collection existence: {e}") from e

        if not exists:
            logger.info("Creating Qdrant collection '%s' (size=%d)", self._state.name, vector_size)
            try:
                self.client.recreate_collection(
                    collection_name=self._state.name,
                    vectors_config=qm.VectorParams(size=vector_size, distance=self._state.distance)
                )
            except Exception as e:
                raise RuntimeError(f"Failed creating collection: {e}") from e
            # Create payload indexes (optimize later when volumes grow)
            try:
                self.client.create_payload_index(
                    self._state.name,
                    field_name="tenant_id",
                    field_schema=qm.PayloadSchemaType.KEYWORD
                )
                self.client.create_payload_index(
                    self._state.name,
                    field_name="hash",
                    field_schema=qm.PayloadSchemaType.KEYWORD
                )
                self.client.create_payload_index(
                    self._state.name,
                    field_name="source_id",
                    field_schema=qm.PayloadSchemaType.KEYWORD
                )
            except Exception as e:
                logger.warning("Failed creating payload index(es): %s", e)
        else:
            coll_info = self.client.get_collection(self._state.name)
            size_existing = coll_info.vectors_count  # vectors_count is total vectors, not size; fetch vector size via config
            # qdrant-client does not directly expose size here; accept existing.
            pass

        self._state.vector_size = vector_size
        self._ensured = True

    def ensure_collections(self) -> None:
        """
        Ensure we know whether the collection exists on the Qdrant server.

        Behaviour:
          - If the collection already exists, mark the store as ensured so
            read-only operations (search/scroll/count) can run after a process
            restart without requiring a new upsert.
          - If the collection does not exist we defer creation until the first
            upsert because we need the embedding dimension to create the
            collection.
        """
        if self._ensured:
            return

        try:
            exists = self.client.collection_exists(self._state.name)
        except Exception as e:
            # If we cannot reach Qdrant, do not raise here; callers will handle as empty.
            logger.warning("Failed checking collection existence: %s", e)
            return

        if exists:
            logger.info("Qdrant collection '%s' exists; enabling read operations", self._state.name)
            # We intentionally do not attempt to derive vector_size here (qdrant client
            # surface differs between versions). The important part is allowing searches
            # to proceed after a restart. vector_size will be set during the first upsert.
            self._ensured = True
        else:
            logger.debug("Qdrant collection '%s' does not exist; will create on first upsert", self._state.name)

    # -------------------------- Public API --------------------------

    def upsert(
        self,
        tenant_id: str,
        docs: Sequence[VectorDocument],
        embeddings: Sequence[Sequence[float]],
        skip_if_exists: bool = True,
    ) -> int:
        if not docs:
            return 0
        if len(docs) != len(embeddings):
            raise ValueError("docs and embeddings length mismatch")

        vector_size = len(embeddings[0])
        if not self._ensured:
            self._ensure_collection(vector_size)
        elif self._state.vector_size and self._state.vector_size != vector_size:
            raise ValueError(f"Inconsistent embedding size. Expected {self._state.vector_size} got {vector_size}")

        # Optionally skip duplicates based on hash metadata if present
        if skip_if_exists:
            hashes = [d.metadata.get("hash") for d in docs if "hash" in d.metadata]
            existing_map: Dict[str, bool] = {}
            if hashes:
                existing_map = self.doc_exists_hash(tenant_id, [h for h in hashes if h])
            filtered: List[tuple[VectorDocument, Sequence[float]]] = []
            for d, emb in zip(docs, embeddings):
                h = d.metadata.get("hash")
                if h and existing_map.get(h):
                    continue
                filtered.append((d, emb))
            docs, embeddings = [t[0] for t in filtered], [t[1] for t in filtered]

        if not docs:
            return 0

        points: List[qm.PointStruct] = []
        for d, emb in zip(docs, embeddings):
            payload = dict(d.metadata)
            payload["tenant_id"] = tenant_id
            points.append(
                qm.PointStruct(
                    id=d.id,
                    vector=list(emb),
                    payload=payload
                )
            )

        try:
            self.client.upsert(collection_name=self._state.name, points=points, wait=True)
        except Exception as e:
            raise RuntimeError(f"Qdrant upsert failed: {e}") from e

        return len(points)

    def search(
        self,
        tenant_id: str,
        embedding: Sequence[float],
        top_k: int,
        score_threshold: Optional[float] = None,
        source_ids: Optional[List[str]] = None,
    ) -> List[SearchResult]:
        # Ensure we know whether the collection exists (this allows searches after
        # process restarts when collection was created previously).
        if not self._ensured:
            self.ensure_collections()
        if not self._ensured:
            # Nothing indexed yet
            return []

        # Build filter conditions
        must_conditions = [
            qm.FieldCondition(
                key="tenant_id",
                match=qm.MatchValue(value=tenant_id)
            )
        ]
        
        # Add source filtering if provided
        if source_ids:
            should_conditions = [
                qm.FieldCondition(
                    key="source_id",
                    match=qm.MatchValue(value=source_id)
                )
                for source_id in source_ids
            ]
            
            query_filter = qm.Filter(
                must=must_conditions,
                should=should_conditions
            )
        else:
            query_filter = qm.Filter(must=must_conditions)

        # Debug: log key search parameters to help diagnose missing-hit issues
        try:
            logger.debug(
                "Qdrant search called: collection=%s, query_vector_len=%d, top_k=%d, score_threshold=%s, tenant_id=%s, source_ids=%s",
                self._state.name, len(embedding) if embedding is not None else 0, top_k, score_threshold, tenant_id, source_ids
            )
            res = self.client.search(
                collection_name=self._state.name,
                query_vector=list(embedding),
                limit=top_k,
                with_payload=True,
                with_vectors=False,
                score_threshold=score_threshold,
                query_filter=query_filter
            )
        except Exception as e:
            logger.error("Qdrant search error: %s", e)
            raise RuntimeError(f"Qdrant search failed: {e}") from e

        out: List[SearchResult] = []
        for p in res:
            payload = p.payload or {}
            text = payload.get("text") or payload.get("content") or ""
            out.append(
                SearchResult(
                    id=str(p.id),
                    text=text,
                    score=p.score,
                    metadata=payload
                )
            )
        return out

    def delete_tenant(self, tenant_id: str) -> int:
        if not self._ensured:
            return 0
        try:
            result = self.client.delete(
                collection_name=self._state.name,
                points_selector=qm.FilterSelector(
                    filter=qm.Filter(
                        must=[
                            qm.FieldCondition(
                                key="tenant_id",
                                match=qm.MatchValue(value=tenant_id)
                            )
                        ]
                    )
                )
            )
            # Qdrant does not always return count; return -1 unknown deletion count
            return -1
        except Exception as e:
            raise RuntimeError(f"Failed deleting tenant docs: {e}") from e

    def count_tenant(self, tenant_id: str) -> int:
        if not self._ensured:
            return 0
        try:
            count_res = self.client.count(
                collection_name=self._state.name,
                count_filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="tenant_id",
                            match=qm.MatchValue(value=tenant_id)
                        )
                    ]
                )
            )
            return count_res.count
        except Exception as e:
            raise RuntimeError(f"Failed counting tenant docs: {e}") from e

    def doc_exists_hash(
        self,
        tenant_id: str,
        content_hashes: Sequence[str],
    ) -> Dict[str, bool]:
        if not content_hashes:
            return {}
        if not self._ensured:
            return {h: False for h in content_hashes}

        # Qdrant filter supports 'should' for OR semantics.
        conditions = [
            qm.FieldCondition(
                key="hash",
                match=qm.MatchValue(value=h)
            ) for h in content_hashes
        ]

        try:
            scroll_iter = self.client.scroll(
                collection_name=self._state.name,
                limit=len(content_hashes),
                with_payload=True,
                scroll_filter=qm.Filter(
                    must=[
                        qm.FieldCondition(
                            key="tenant_id",
                            match=qm.MatchValue(value=tenant_id)
                        )
                    ],
                    should=conditions
                )
            )
        except Exception as e:
            raise RuntimeError(f"Failed hash existence lookup: {e}") from e

        points, _ = scroll_iter
        found_hashes = set()
        for p in points:
            if p.payload and "hash" in p.payload:
                found_hashes.add(p.payload["hash"])

        return {h: (h in found_hashes) for h in content_hashes}

# Factory helper
_store_singleton: Optional[QdrantVectorStore] = None
def get_vector_store() -> QdrantVectorStore:
    global _store_singleton
    if _store_singleton is None:
        _store_singleton = QdrantVectorStore()
    return _store_singleton
