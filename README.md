# RAG Tutorial v2

Small RAG (Retrieval-Augmented Generation) reference app demonstrating:
- document ingestion (PDFs)
- embeddings (local / cloud providers)
- vector store (Qdrant)
- a simple chat/query API and a web demo

Repository layout
- app/ — main application code (API, chat service, ingestion pipeline, embedding factories)
- data/ and data_2/ — sample PDFs used for ingestion
- web/index.html — minimal demo UI
- docker-compose.yml / Dockerfile — containerized run options
- populate_database.py — run ingestion to populate vector store
- query_data.py / get_embedding_function.py — helpers and examples
- test_api_flow.py / test_rag.py — test examples

Quickstart (local, Python)
1. Create virtualenv and install deps
```bash
# Bash
python -m venv .venv
.venv/Scripts/activate    # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

2. Ingest sample documents
```bash
python populate_database.py
```

3. Run the API
```bash
python app/main.py
# or use your preferred ASGI server:
# uvicorn app.main:app --reload
```

4. Query the running service (example)
- Check `app/main.py` for the exact route path used by this repo. A generic POST example:
```bash
# Bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"What is invoice INV-2023-001 about?"}'
```
- The web demo: open `web/index.html` (or point it at the running API as configured).

Docker / Docker Compose
- To run with Docker Compose:
```bash
docker compose up --build
```
- Configure provider and models via environment variables (docker-compose.yml or .env).

Important environment variables (examples)
- PROVIDER_EMBED — provider for embeddings (e.g., `openai`, `ollama`)
- PROVIDER_CHAT — provider for chat/completion
- EMBED_MODEL — embedding model name (e.g., `text-embedding-3-small`, `nomic-embed-text`)
- CHAT_MODEL — chat model name (e.g., `gpt-4o`, `llama2`)
- OLLAMA_BASE_URL — if using a local Ollama server (e.g., `http://host.docker.internal:11434`)
- OPENAI_API_KEY — OpenAI key (if using OpenAI)
- QDRANT_URL / QDRANT_API_KEY — Qdrant connection details

Example .env snippet
```
PROVIDER_EMBED=openai
PROVIDER_CHAT=openai
EMBED_MODEL=text-embedding-3-small
CHAT_MODEL=gpt-4o
OPENAI_API_KEY=sk-...
QDRANT_URL=http://localhost:6333
```

Testing
```bash
pytest -q
```

Troubleshooting
- ModuleNotFoundError: No module named 'langchain.document_loaders'
  - Cause: mismatched langchain package version vs code import paths.
  - Options:
    1) Pin to the known-working langchain version used by this repo (if the code expects 0.3.x):
```bash
pip install "langchain==0.3.14" "langchain-core==0.3.35"
```
    2) Or update imports in ingestion code to match langchain 1.0+ layout (code changes required).
  - Create and use a virtualenv to avoid system-wide package conflicts.
- If pip reports dependency resolver conflicts after upgrading, prefer creating a fresh venv and installing requirements.txt.

Notes about providers and compatibility
- Some third-party packages in the ecosystem require older langchain-core versions. If you need legacy compatibility, pin package versions in `requirements.txt`.
- If you want Ollama locally:
  - Start Ollama and set `OLLAMA_BASE_URL`.
  - Set `PROVIDER_EMBED=ollama` and `EMBED_MODEL` to the Ollama embedding model name.
- For hybrid setups (embed with local Ollama, chat with OpenAI), set PROVIDER_EMBED and PROVIDER_CHAT independently.

Contributing
- Open issues or PRs for improvements.
- Keep changes minimal and include tests where applicable.

License
- Default: MIT (update if you prefer another license)

Contact / References
- See `app/` for implementation details and `docker-compose.yml` for environment variable examples.
