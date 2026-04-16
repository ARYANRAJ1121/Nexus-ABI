import httpx, json

# Test /health
r = httpx.get("http://localhost:8000/health", timeout=10)
print(f"GET /health → {r.status_code}")
print(json.dumps(r.json(), indent=2))

# Test /metrics
print("\nGET /metrics →")
r = httpx.get("http://localhost:8000/metrics", timeout=15)
data = r.json()
print(f"Status: {r.status_code} | Total metrics: {data['total_metrics']}")
for m in data["metrics"]:
    print(f"  {m['display_name']:<40} {m['value']} {m['unit']}")

# Test /ask
print("\nPOST /ask →")
r = httpx.post(
    "http://localhost:8000/ask",
    json={"question": "What is our churn rate?"},
    timeout=120,
)
data = r.json()
print(f"Status: {r.status_code} | Priority: {data['priority']} | Elapsed: {data['elapsed_seconds']}s")
print(f"Actions: {len(data['actions'])}")
for a in data["actions"]:
    print(f"  {a['index']}. {a['action'][:80]}...")
