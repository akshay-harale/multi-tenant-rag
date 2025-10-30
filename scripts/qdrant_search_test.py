#!/usr/bin/env python3
"""
Fetch a stored point (with its vector) then run a Qdrant search using that same vector.
This verifies whether Qdrant's vector search returns matches when indexed_vectors_count is 0.
Improved error handling to show HTTP error body.
"""
import json, urllib.request, sys, math, urllib.error

BASE = "http://localhost:6333"
SCROLL_URL = f"{BASE}/collections/documents/points/scroll"
SEARCH_URL = f"{BASE}/collections/documents/points/search"

def post_json(url, payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type":"application/json"})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        print(f"HTTPError {e.code}: {body}")
        raise

try:
    # 1) Scroll to get one point with full vector
    sc = post_json(SCROLL_URL, {"limit": 1, "with_vector": True})
    pts = sc.get("result", {}).get("points", [])
    if not pts:
        print("No points returned from scroll. Collection empty or API error.")
        print(json.dumps(sc, indent=2))
        sys.exit(0)
    p = pts[0]
    vec = p.get("vector")
    pid = p.get("id")
    print("Using point id:", pid)
    if not vec:
        print("Point has no vector stored.")
        print(json.dumps(p, indent=2))
        sys.exit(0)
    print("Vector length:", len(vec))

    # 2) Run search with that exact vector and tenant_id filter
    payload = {
        "limit": 5,
        "with_payload": True,
        "with_vector": False,
        "vector": vec,
        "filter": {
            "must": [
                { "key": "tenant_id", "match": { "value": p.get("payload", {}).get("tenant_id") } }
            ]
        }
    }
    res = post_json(SEARCH_URL, payload)
    print("\nSearch response:")
    print(json.dumps(res, indent=2))

    # 3) If no results, print some diagnostics
    hits = res.get("result", [])
    if not hits:
        print("\nNo search hits. Diagnostics:")
        # show collection meta
        coll = post_json(f"{BASE}/collections/documents")
        print("Collection meta (trimmed):")
        print(json.dumps(coll.get("result", {}), indent=2))
        sys.exit(0)
    else:
        print(f"\nFound {len(hits)} hits (should include the point used).")
        for h in hits:
            print("hit id:", h.get("id"), "score:", h.get("score"))
except Exception as e:
    print("ERROR", repr(e))
    sys.exit(2)
