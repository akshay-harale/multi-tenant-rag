import json, sys, time
import requests

BASE = "http://127.0.0.1:8000"

def post(path: str, payload: dict):
    r = requests.post(f"{BASE}{path}", json=payload)
    try:
        data = r.json()
    except Exception:
        print("Non-JSON response:", r.text)
        sys.exit(1)
    print(path, r.status_code, json.dumps(data, indent=2))
    return r.status_code, data

def get(path: str):
    r = requests.get(f"{BASE}{path}")
    try:
        data = r.json()
    except Exception:
        print("Non-JSON response:", r.text)
        sys.exit(1)
    print(path, r.status_code, json.dumps(data, indent=2))
    return r.status_code, data

def main():
    # 1. Create tenant
    post("/tenants", {"tenant_id": "acme"})
    # 2. List tenants
    get("/tenants")
    # 3. Ingest local 'data' directory
    post("/tenants/acme/ingest", {"directory": "data"})
    # 4. Simple search
    post("/tenants/acme/search", {"query": "invoice", "top_k": 5})

if __name__ == "__main__":
    main()
