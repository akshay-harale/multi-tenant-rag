from __future__ import annotations
import uuid
import httpx
from typing import List, Dict, Any, Tuple
from app.core.db import fetch_one, fetch_all, execute

from app.core.config import get_settings
from app.embeddings.factory import get_embedding_service
from app.vector.qdrant_store import get_vector_store
from app.chat.model_factory import chat_complete

settings = get_settings()

# -------- Postgres Conversation Persistence --------

def load_session(tenant_id: str, session_id: str) -> List[Dict[str, str]]:
    rows = fetch_all(
        "SELECT role, content FROM chat_messages WHERE tenant_id=%s AND session_id=%s ORDER BY turn_index",
        tenant_id, session_id
    )
    return [{"role": r["role"], "content": r["content"]} for r in rows]

def ensure_session(tenant_id: str, session_id: str) -> None:
    execute(
        "INSERT INTO chat_sessions (tenant_id, session_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
        tenant_id, session_id
    )

def append_messages(tenant_id: str, session_id: str, new_messages: List[Dict[str, str]]) -> None:
    # Get current max turn_index
    row = fetch_one(
        "SELECT COALESCE(MAX(turn_index), -1) AS max_idx FROM chat_messages WHERE tenant_id=%s AND session_id=%s",
        tenant_id, session_id
    )
    start_idx = (row["max_idx"] if row else -1) + 1
    for offset, m in enumerate(new_messages):
        execute(
            "INSERT INTO chat_messages (tenant_id, session_id, turn_index, role, content) VALUES (%s, %s, %s, %s, %s)",
            tenant_id, session_id, start_idx + offset, m["role"], m["content"]
        )

SYSTEM_TEMPLATE = (
    "You are a retrieval augmented assistant. Use ONLY the provided context chunks.\n"
    "If the answer is not in the context, say you do not know.\n"
    "Cite sources by filename and chunk id if possible.\n\n"
    "Context Chunks:\n{context}\n\n"
    "Answer the user query clearly and concisely."
)

def build_context_chunks(results: List[Any], max_chars: int = 8000) -> Tuple[str, List[str], List[str]]:
    """
    results: list of SearchResult (id, text, score, metadata)
    Returns: (context_string, citation_ids, source_list)
    """
    context_parts: List[str] = []
    citations: List[str] = []
    sources: List[str] = []
    total_len = 0
    for r in results:
        source = r.metadata.get("source") or "unknown"
        cid = f"{source}#chunk{r.metadata.get('chunk_index')}"
        snippet = r.text[:1200]
        block = f"[{cid} | score={round(r.score,4)}]\n{snippet}"
        block_len = len(block)
        if total_len + block_len > max_chars:
            break
        context_parts.append(block)
        citations.append(cid)
        sources.append(source)
        total_len += block_len
    return "\n\n".join(context_parts), citations, sources


def rag_chat(
    tenant_id: str,
    user_message: str,
    session_id: str | None,
    top_k: int,
    include_history: bool
) -> Dict[str, Any]:
    """
    Main RAG chat pipeline:
      1. Create / load session
      2. Retrieve top_k chunks for user message
      3. Construct system + history + user messages
      4. Call Ollama chat model
      5. Persist conversation
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    # Ensure session exists then load history
    ensure_session(tenant_id, session_id)
    history_messages = load_session(tenant_id, session_id) if include_history else []

    embeddings = get_embedding_service()
    vector_store = get_vector_store()

    # Retrieval
    query_emb = embeddings.embed_query(user_message)
    results = vector_store.search(
        tenant_id=tenant_id,
        embedding=query_emb,
        top_k=top_k,
        score_threshold=None
    )

    context_str, citations, sources = build_context_chunks(results)

    if not results:
        answer = "No relevant context found for this query."
        # Persist user + assistant minimal
        history_messages.append({"role": "user", "content": user_message})
        history_messages.append({"role": "assistant", "content": answer})
        append_messages(tenant_id, session_id, [{"role": "user", "content": user_message},
                                                {"role": "assistant", "content": answer}])
        return {
            "session_id": session_id,
            "answer": answer,
            "citations": [],
            "used_chunks": 0,
            "sources": []
        }

    system_prompt = SYSTEM_TEMPLATE.format(context=context_str)
    # Build message list for LLM
    llm_messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # Append prior conversation (truncate if very long)
    token_budget_est = 0
    for m in history_messages[-20:]:
        token_budget_est += len(m["content"])
        if token_budget_est > 12000:
            break
        llm_messages.append({"role": m["role"], "content": m["content"]})

    # Current user turn
    llm_messages.append({"role": "user", "content": user_message})

    # Call model
    try:
        answer = chat_complete(llm_messages)
    except Exception as e:
        answer = f"LLM backend error: {e}"

    # Persist
    history_messages.append({"role": "user", "content": user_message})
    history_messages.append({"role": "assistant", "content": answer})
    append_messages(tenant_id, session_id, [{"role": "user", "content": user_message},
                                            {"role": "assistant", "content": answer}])

    return {
        "session_id": session_id,
        "answer": answer,
        "citations": citations,
        "used_chunks": len(citations),
        "sources": list(dict.fromkeys(sources))
    }
