import httpx, json

BASE = "http://localhost:8000"

# 1. KPIs
r = httpx.get(f"{BASE}/powerbi/kpis", timeout=15)
print(f"\nGET /powerbi/kpis → {r.status_code}")
for m in r.json():
    print(f"  {m['metric_name']:<42} {m['value']} {m['unit']}")

# 2. Churn by industry
r = httpx.get(f"{BASE}/powerbi/churn-by-industry", timeout=15)
print(f"\nGET /powerbi/churn-by-industry → {r.status_code} ({len(r.json())} industries)")
for row in r.json()[:4]:
    print(f"  {row['industry']:<25} churn={row['churn_rate_pct']}%  avg_spend=${row['avg_monthly_spend']}")

# 3. Churn by plan
r = httpx.get(f"{BASE}/powerbi/churn-by-plan", timeout=15)
print(f"\nGET /powerbi/churn-by-plan → {r.status_code}")
for row in r.json():
    print(f"  {row['plan_type']:<12} churn={row['churn_rate_pct']}%  customers={row['total_customers']}")

# 4. At-risk customers
r = httpx.get(f"{BASE}/powerbi/at-risk-customers", timeout=15)
data = r.json()
print(f"\nGET /powerbi/at-risk-customers → {r.status_code} ({len(data)} customers scored)")
for row in data[:5]:
    print(f"  {row['company_name']:<35} risk={row['risk_level']}  churn_prob={row['churn_probability_pct']}%")
