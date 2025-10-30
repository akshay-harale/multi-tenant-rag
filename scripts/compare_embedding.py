#!/usr/bin/env python3
"""
Run inside the api container (so env / provider keys match).
1. Fetch one stored point (with vector) from Qdrant.
2. Compute embedding for a test query using the app's embedding service.
3. Print lengths and cosine similarity to check whether the API's embedding aligns with stored vectors.
"""
import json, urllib.request, sys, math
from typing import List

QDRANT_SCROLL = "http://qdrant:6333/collections/documents/points/scroll"
TEST_QUERY = "amount due"

def fetch_point():
    data = json.dumps({"limit":1, "with_vector": True}).encode("utf-8")
    req = urllib.request.Request(QDRANT_SCROLL, data=data, headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode())

def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na*nb)

def main():
    try:
        obj = fetch_point()
    except Exception as e:
        print("Failed fetching point:", e)
        sys.exit(2)
    pts = obj.get("result", {}).get("points", [])
    if not pts:
        print("No points in collection")
        print(json.dumps(obj, indent=2))
        sys.exit(0)
    p = pts[0]
    stored_vec = p.get("vector")
    print("Stored vector length:", len(stored_vec) if stored_vec else "MISSING")

    # Import embedding service from app (must run inside api container)
    try:
        from app.embeddings.factory import get_embedding_service
    except Exception as e:
        print("Failed importing embedding service:", repr(e))
        sys.exit(2)

    svc = get_embedding_service()
    emb = svc.embed_query(TEST_QUERY)
    print("Computed embedding length:", len(emb))
    sim = cosine(stored_vec, emb)
    print("Cosine similarity (stored_vec vs query embedding):", sim)
    # Show a few components
    print("Stored vec[0:8]:", stored_vec[:8])
    print("Query emb[0:8]:", emb[:8])

if __name__ == '__main__':
    main()
