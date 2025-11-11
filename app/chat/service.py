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
    "You are a retrieval-augmented assistant. You MUST use ONLY the provided context chunks to answer.\n"
    "Do NOT attempt to answer from outside knowledge. Cite sources by filename and chunk id when possible.\n\n"
    "Context Chunks:\n{context}\n\n"
    "Answer the user query clearly and concisely. If you must refuse, use the exact phrase above."
)

def build_context_chunks(results: List[Any], max_chars: int = 8000) -> Tuple[str, List[str], List[str]]:
    """
    results: list of SearchResult (id, text, score, metadata)
    Returns: (context_string, citation_ids, source_list)

    Notes:
    - Respect settings.max_context_docs by trimming the candidate results before assembling context.
    - This reduces prompt size and keeps highest-confidence chunks.
    """
    # Trim to configured max documents (results are expected ordered by score)
    trimmed = results[:settings.max_context_docs] if settings.max_context_docs and len(results) > settings.max_context_docs else results

    context_parts: List[str] = []
    citations: List[str] = []
    sources: List[str] = []
    total_len = 0
    for r in trimmed:
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
    include_history: bool,
    source_ids: List[str] | None = None
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
    # Use a wider recall for search then trim context; apply configured score threshold
    search_top_k = settings.max_search_k if settings.max_search_k and settings.max_search_k > top_k else top_k
    results = vector_store.search(
        tenant_id=tenant_id,
        embedding=query_emb,
        top_k=search_top_k,
        score_threshold=settings.min_score_threshold,
        source_ids=source_ids
    )

    context_str, citations, sources = build_context_chunks(results)

    # Retrieval safety checks
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

    # If a min_score_threshold is configured, ensure top hit meets it
    top_score = results[0].score if results else None
    # if settings.min_score_threshold is not None and (top_score is None or top_score < settings.min_score_threshold):
    #     answer = "No relevant context found for this query."
    #     history_messages.append({"role": "user", "content": user_message})
    #     history_messages.append({"role": "assistant", "content": answer})
    #     append_messages(tenant_id, session_id, [{"role": "user", "content": user_message},
    #                                             {"role": "assistant", "content": answer}])
    #     return {
    #         "session_id": session_id,
    #         "answer": answer,
    #         "citations": [],
    #         "used_chunks": 0,
    #         "sources": []
    #     }

    # For short queries, ensure the retrieved chunks actually contain the query keyword to avoid noisy hits
    # if settings.require_keyword_in_short_queries:
    #     tokens = [t.lower() for t in user_message.split() if t.strip()]
    #     if len(tokens) <= settings.short_query_token_threshold:
    #         found = False
    #         for r in results:
    #             text_lower = (r.text or "").lower()
    #             if any(tok in text_lower for tok in tokens):
    #                 found = True
    #                 break
    #         if not found:
    #             answer = "No relevant context found for this query."
    #             history_messages.append({"role": "user", "content": user_message})
    #             history_messages.append({"role": "assistant", "content": answer})
    #             append_messages(tenant_id, session_id, [{"role": "user", "content": user_message},
    #                                                     {"role": "assistant", "content": answer}])
    #             return {
    #                 "session_id": session_id,
    #                 "answer": answer,
    #                 "citations": [],
    #                 "used_chunks": 0,
    #                 "sources": []
    #             }

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
    #print("Context:", context_str)

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

    print("LLM message", llm_messages)
    # Call model
    try:
        answer = chat_complete(llm_messages)
    except Exception as e:
        answer = f"LLM backend error: {e}"

    print("LLM answer:", answer)
    # Normalise unknown/empty model replies to the exact phrase required by the system prompt.
    # This ensures callers get the deterministic response when the model cannot answer
    # from the provided documents.
    canonical_unknown = "I don't know based on the provided documents."
    if isinstance(answer, str):
        ans_str = answer.strip()
        low = ans_str.lower()
        # If answer is empty or contains a variant of "i don't know", replace with canonical phrase.
        if not ans_str or "i don't know" in low:
            answer = canonical_unknown

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
