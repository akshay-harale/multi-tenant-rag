#!/usr/bin/env python3
import json
import urllib.request
import sys

url = "http://localhost:8000/tenants/john/search"
payload = {"query": "amount due", "top_k": 5}
data = json.dumps(payload).encode("utf-8")
req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        body = resp.read().decode()
        try:
            parsed = json.loads(body)
            print(json.dumps(parsed, indent=2))
        except Exception:
            print(body)
except Exception as e:
    print("ERROR", repr(e))
    sys.exit(1)
