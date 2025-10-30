#!/usr/bin/env python3
import json, urllib.request, sys

url = "http://localhost:6333/collections/documents/points/scroll"
payload = {"limit": 1, "with_vector": True}
data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode()
        obj = json.loads(body)
        pts = obj.get("result", {}).get("points", [])
        if not pts:
            print("No points returned")
            print(json.dumps(obj, indent=2))
            sys.exit(0)
        p = pts[0]
        print("id:", p.get("id"))
        payload = p.get("payload", {})
        print("payload keys:", list(payload.keys()))
        vector = p.get("vector", None)
        if vector is None:
            print("vector: MISSING")
        else:
            print("vector length:", len(vector))
            print("first 8 components:", vector[:8])
        print("\nfull point (trimmed):")
        p_copy = dict(p)
        if "vector" in p_copy:
            p_copy["vector"] = p_copy["vector"][:8]
        print(json.dumps(p_copy, indent=2))
except Exception as e:
    print("ERROR", repr(e))
    sys.exit(1)
