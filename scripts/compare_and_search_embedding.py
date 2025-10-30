#!/usr/bin/env python3
"""
Compute embedding for a test query using the app's embedding service (inside api container),
then call Qdrant search with that embedding and print results.
"""
import json, urllib.request, sys, math
from typing import List

QDRANT_SEARCH = "http://qdrant:6333/collections/documents/points/search"
TEST_QUERY = "amount due"

def post_json(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode())

def main():
    # Import embedding service from app (must run inside api container)
    try:
        from app.embeddings.factory import get_embedding_service
    except Exception as e:
        print("Failed importing embedding service:", repr(e))
        sys.exit(2)

    svc = get_embedding_service()
    emb = svc.embed_query(TEST_QUERY)
    print("Computed embedding length:", len(emb))
    print("embedding[0:8]:", emb[:8])

    payload = {
        "limit": 5,
        "with_payload": True,
        "with_vector": False,
        "vector": emb,
        "filter": {
            "must": [
                { "key": "tenant_id", "match": { "value": "john" } }
            ]
        }
    }
    try:
        res = post_json(QDRANT_SEARCH, payload)
        print("\nQdrant search response (using computed embedding):")
        print(json.dumps(res, indent=2))
    except Exception as e:
        print("Failed Qdrant search:", repr(e))
        sys.exit(3)

if __name__ == '__main__':
    main()
