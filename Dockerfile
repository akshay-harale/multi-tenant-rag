# ---------- Base Image (Slim Python) ----------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps (libmagic/poppler/etc not required here; add if future PDF tooling needs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---------- Dependency Layer ----------
FROM base AS deps

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# ---------- Runtime Image ----------
FROM base AS runtime

# Non-root user
RUN useradd -m appuser \
 && mkdir -p /data/storage/uploads \
 && chown -R appuser:appuser /data
USER appuser

WORKDIR /app

# Copy installed libs from deps layer
COPY --from=deps /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy source
COPY app ./app
COPY data ./data
COPY web ./web
COPY test_api_flow.py .
COPY README.md .
COPY requirements.txt .
COPY docker-compose.yml .

# Storage mount path (matches STORAGE_ROOT=/data/storage in compose)
VOLUME ["/data/storage"]

EXPOSE 8000

# Health check (simple)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://127.0.0.1:8000/health || exit 1

# Default command (can be overridden in compose)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
