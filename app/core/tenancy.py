from __future__ import annotations
import re
import uuid
from dataclasses import dataclass
from typing import Set
from app.core.config import get_settings
from app.core.db import fetch_one, fetch_all, execute

_settings = get_settings()
_TENANT_ID_RE = re.compile(_settings.tenant_id_pattern)

@dataclass(frozen=True)
class TenantContext:
    tenant_id: str

# ------------- Validation -------------

def validate_tenant_id(tenant_id: str) -> None:
    if not tenant_id:
        raise ValueError("tenant_id required")
    if not _TENANT_ID_RE.match(tenant_id):
        raise ValueError("Invalid tenant_id pattern")

# ------------- Persistence (Postgres) -------------

def register_tenant(tenant_id: str) -> None:
    """
    Insert tenant row if it does not exist (idempotent).
    """
    validate_tenant_id(tenant_id)
    row = fetch_one("SELECT tenant_id FROM tenants WHERE tenant_id=%s", tenant_id)
    if row:
        return
    execute("INSERT INTO tenants (tenant_id) VALUES (%s)", tenant_id)

def list_tenants() -> Set[str]:
    rows = fetch_all("SELECT tenant_id FROM tenants ORDER BY tenant_id")
    return {r["tenant_id"] for r in rows}

def ensure_tenant_exists(tenant_id: str) -> None:
    if not fetch_one("SELECT tenant_id FROM tenants WHERE tenant_id=%s", tenant_id):
        raise ValueError(f"Unknown tenant_id '{tenant_id}'. Create tenant first.")

def get_tenant_context(tenant_id: str) -> TenantContext:
    validate_tenant_id(tenant_id)
    ensure_tenant_exists(tenant_id)
    return TenantContext(tenant_id=tenant_id)

# ------------- API Key Authorization (Future) -------------

# Placeholder table not yet created; logic retained for future extension.
# For now always pass if no mapping implemented.
def authorize_api_key(api_key: str, tenant_id: str) -> None:
    # Extend with a join table (api_keys -> tenants) when needed.
    # Example:
    # row = fetch_one(\"SELECT 1 FROM api_key_tenants WHERE api_key=%s AND tenant_id=%s\", api_key, tenant_id)
    # if not row: raise PermissionError(\"API key not authorized for tenant\")
    return

# ------------- Demo / Dev Helper -------------

def bootstrap_demo_tenant(tenant_id: str = "tenant_demo"):
    register_tenant(tenant_id)
