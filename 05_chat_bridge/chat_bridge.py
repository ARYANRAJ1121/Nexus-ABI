"""
=============================================================================
NEXUS-ABI | Layer 5: Chat Bridge
File: chat_bridge.py
=============================================================================

PURPOSE:
  A FastAPI web server that exposes the entire Nexus-ABI pipeline as an API.
  Any tool that can make an HTTP request — Power BI, a browser, Postman,
  a mobile app — can now query the AI pipeline without touching Python.

WHY FASTAPI?
  FastAPI is the modern Python web framework. It auto-generates interactive
  API documentation at /docs (Swagger UI) — so you can test every endpoint
  in the browser without writing any extra code.
  Alternative: Flask — older, slower, no automatic docs. FastAPI is the
  industry standard for ML/AI APIs.

HOW IT WORKS:
  - On startup: all three agents are initialised ONCE (warm-up).
    Loading XGBoost, ChromaDB, and the LLM config takes ~5 seconds.
    After that, each request is fast because the agents are already ready.
  - On each /ask request: run_nexus() is called with the question.
    The entire SQL + RAG + Strategist pipeline runs and the JSON result
    is returned to the caller.

ENDPOINTS:
  GET  /           → Welcome message + available endpoints
  GET  /health     → Checks Ollama + DB status before you start sending questions
  POST /ask        → Main endpoint: send question, get strategy
  GET  /metrics    → Computes all 10 KPIs live from the Semantic Layer
  GET  /schema     → Returns the database schema (useful for debugging)

HOW TO START THE SERVER:
  cd 05_chat_bridge
  uvicorn chat_bridge:app --host 0.0.0.0 --port 8000 --reload

HOW TO TEST IT:
  Browser: http://localhost:8000/docs   → interactive Swagger UI
  Curl:    curl -X POST http://localhost:8000/ask \
           -H "Content-Type: application/json" \
           -d '{"question": "What is our churn rate?"}'

HOW POWER BI CONNECTS:
  Data Source → Web → URL: http://localhost:8000/metrics
  Power BI fetches the JSON and maps columns to visuals automatically.
=============================================================================
"""

import os
import sys
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

os.environ["PYTHONUTF8"] = "1"

PROJECT_ROOT = Path(__file__).parent.parent
LAYER_4_ROOT = PROJECT_ROOT / "04_agentic_engine"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LAYER_4_ROOT))

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from loguru import logger

# ---------------------------------------------------------------------------
# WARM-UP: Agents are loaded once at startup, not per-request
# ---------------------------------------------------------------------------
from agents.sql_agent        import SQLAgent
from agents.rag_agent        import RAGAgent
from agents.strategist_agent import StrategistAgent
from main_graph              import run_nexus

sys.path.insert(0, str(PROJECT_ROOT / "03_semantic_layer"))
from metrics_defs import compute_all, METRICS

from utils.llm_config import health_check as ollama_health_check
from utils.db_config  import get_schema_info, get_engine, load_data_to_db


# ---------------------------------------------------------------------------
# GLOBAL AGENT INSTANCES (initialised once in lifespan)
# ---------------------------------------------------------------------------
_agents_ready = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Runs ONCE when the server starts (before accepting any requests).
    Initialises all agents so the first /ask request isn't slow.
    """
    global _agents_ready
    logger.info("=== NEXUS-ABI Server Starting ===")

    # Ensure DB is populated
    logger.info("Checking database...")
    try:
        load_data_to_db(force_reload=False)
        logger.success("Database ready.")
    except Exception as e:
        logger.warning("DB init warning: {}", e)

    # Ensure RAG vectorstore is populated
    logger.info("Checking RAG vectorstore...")
    try:
        _rag = RAGAgent()
        _rag.ingest(force_reload=False)
        logger.success("RAG vectorstore ready.")
    except Exception as e:
        logger.warning("RAG init warning: {}", e)

    _agents_ready = True
    logger.success("All agents warmed up. Server ready.")

    yield  # Server is now running and accepting requests

    logger.info("Server shutting down.")


# ---------------------------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------------------------
app = FastAPI(
    title        = "Nexus-ABI API",
    description  = "Agentic Business Intelligence — turn plain English questions into governed, predictive business strategies.",
    version      = "1.0.0",
    docs_url     = "/docs",
    redoc_url    = "/redoc",
    lifespan     = lifespan,
)

# CORS — allows browsers, Power BI, and any front-end to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],   # In production: restrict to your domain
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ---------------------------------------------------------------------------
# REQUEST / RESPONSE MODELS
# Pydantic models give you automatic validation + documentation
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length  = 5,
        max_length  = 500,
        description = "Your business question in plain English.",
        examples    = [
            "What is our current churn rate?",
            "Which customers are at highest risk and what are they complaining about?",
            "What should we do to reduce enterprise churn this quarter?",
        ],
    )


class ActionItem(BaseModel):
    index:  int
    action: str


class AskResponse(BaseModel):
    question:        str
    priority:        str
    recommendation:  str
    actions:         list[ActionItem]
    sql_summary:     Optional[str]
    rag_insight:     Optional[str]
    evidence:        dict
    elapsed_seconds: float


class MetricValue(BaseModel):
    name:         str
    display_name: str
    value:        Optional[float]
    unit:         str
    owner:        str


class MetricsResponse(BaseModel):
    computed_at: str
    total_metrics: int
    metrics: list[MetricValue]


class HealthResponse(BaseModel):
    status:        str
    ollama:        str
    database:      str
    agents_warmed: bool
    model:         str


# ---------------------------------------------------------------------------
# ENDPOINTS
# ---------------------------------------------------------------------------

@app.get("/", tags=["Info"])
async def root():
    """Welcome endpoint — lists all available routes."""
    return {
        "service":     "Nexus-ABI Agentic BI API",
        "version":     "1.0.0",
        "description": "Turn plain English into governed business strategy.",
        "endpoints": {
            "GET  /health":  "Check Ollama + DB status",
            "POST /ask":     "Ask any business question",
            "GET  /metrics": "Compute all 10 live KPIs",
            "GET  /schema":  "View database schema",
            "GET  /docs":    "Interactive API documentation (Swagger UI)",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health():
    """
    Checks the health of all system dependencies.
    Call this first to verify everything is ready before sending questions.
    """
    # Check Ollama
    ollama_result = ollama_health_check()
    ollama_status = ollama_result["status"]

    # Check DB
    try:
        engine = get_engine()
        with engine.connect() as conn:
            from sqlalchemy import text
            count = conn.execute(text("SELECT COUNT(*) FROM customers")).scalar()
        db_status = f"ok ({count:,} customers)"
    except Exception as e:
        db_status = f"error: {e}"

    overall = "ok" if ollama_status == "ok" and "ok" in db_status else "degraded"

    return HealthResponse(
        status        = overall,
        ollama        = ollama_status,
        database      = db_status,
        agents_warmed = _agents_ready,
        model         = "llama3",
    )


@app.post("/ask", response_model=AskResponse, tags=["Intelligence"])
async def ask(request: AskRequest):
    """
    **The main endpoint.** Send a business question in plain English.
    Returns a complete strategy with priority level, recommendations,
    and specific action items — grounded in real data and support tickets.

    **Example questions:**
    - "What is our current churn rate?"
    - "Which Enterprise customers are at risk and why?"
    - "What should we do to prevent churn this quarter?"
    """
    if not _agents_ready:
        raise HTTPException(
            status_code = 503,
            detail      = "Agents are still warming up. Try again in a few seconds.",
        )

    logger.info("POST /ask → '{}'", request.question)
    start_time = time.time()

    try:
        result   = run_nexus(request.question)
        strategy = result.get("strategy", {})

        # Extract action items
        raw_actions = strategy.get("actions", [])
        actions = [
            ActionItem(index=i + 1, action=a)
            for i, a in enumerate(raw_actions)
        ]

        # SQL summary
        sql_result  = result.get("sql_result") or {}
        sql_summary = sql_result.get("summary", "No SQL data available.")

        # RAG insight
        rag_result  = result.get("rag_result") or {}
        rag_insight = rag_result.get("insight", "No support ticket evidence available.")

        elapsed = round(time.time() - start_time, 2)
        logger.success("POST /ask completed in {}s. Priority: {}", elapsed, strategy.get("priority"))

        return AskResponse(
            question        = request.question,
            priority        = strategy.get("priority", "UNKNOWN"),
            recommendation  = strategy.get("recommendation", ""),
            actions         = actions,
            sql_summary     = sql_summary,
            rag_insight     = rag_insight,
            evidence        = strategy.get("evidence", {}),
            elapsed_seconds = elapsed,
        )

    except Exception as e:
        logger.error("POST /ask FAILED: {}", e)
        raise HTTPException(
            status_code = 500,
            detail      = f"Pipeline error: {str(e)}. Check server logs for details.",
        )


@app.get("/metrics", response_model=MetricsResponse, tags=["Intelligence"])
async def metrics():
    """
    Computes all 10 KPIs from the Semantic Layer in real time.
    Returns governed, consistent metric values — same formulas every time.

    **Power BI integration:** Set this URL as a Web data source.
    The returned JSON maps directly to dashboard visuals.
    """
    import pandas as pd
    from utils.db_config import run_query

    logger.info("GET /metrics → computing all KPIs")

    try:
        df = run_query("SELECT * FROM customers")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB query failed: {e}")

    scalar_metrics = [
        "churn_rate", "mrr", "revenue_at_risk", "arpu",
        "avg_clv", "support_ticket_rate", "enterprise_churn_rate",
        "avg_tenure_churned", "inactive_rate",
    ]

    results = []
    for name in scalar_metrics:
        metric = METRICS[name]
        try:
            value = metric.compute(df)
        except Exception:
            value = None

        results.append(MetricValue(
            name         = name,
            display_name = metric.display_name,
            value        = value,
            unit         = metric.unit,
            owner        = metric.owner,
        ))

    from datetime import datetime, timezone
    return MetricsResponse(
        computed_at    = datetime.now(timezone.utc).isoformat(),
        total_metrics  = len(results),
        metrics        = results,
    )


@app.get("/schema", tags=["Debug"])
async def schema():
    """
    Returns the database schema — all tables and column names.
    Useful for understanding what questions the SQL Agent can answer.
    """
    schema_str = get_schema_info()
    tables = {}
    current_table = None

    for line in schema_str.split("\n"):
        line = line.strip()
        if line.startswith("TABLE:"):
            current_table = line.replace("TABLE:", "").strip()
            tables[current_table] = []
        elif line and current_table:
            parts = line.split()
            if parts:
                tables[current_table].append({
                    "column": parts[0],
                    "type":   parts[1] if len(parts) > 1 else "TEXT",
                })

    return {
        "tables":      list(tables.keys()),
        "schema":      tables,
        "total_tables": len(tables),
    }


# ---------------------------------------------------------------------------
# POWER BI ENDPOINTS
# All GET requests returning flat JSON tables — directly consumable by
# Power BI's Web connector without any transformation needed.
#
# In Power BI Desktop:
#   Home → Get Data → Web → paste the URL → Load
# ---------------------------------------------------------------------------

@app.get("/powerbi/kpis", tags=["Power BI"])
async def powerbi_kpis():
    """
    **Power BI Ready — KPI Summary Card Data**

    Returns all 9 KPIs as a flat list. Each row has: name, value, unit.
    Use this to populate KPI cards at the top of your dashboard.

    Power BI: Get Data → Web → `http://localhost:8000/powerbi/kpis`
    Then: Convert to Table → Expand columns → use value for card visuals.
    """
    from utils.db_config import run_query
    df = run_query("SELECT * FROM customers")

    scalar_metrics = [
        "churn_rate", "mrr", "revenue_at_risk", "arpu",
        "avg_clv", "support_ticket_rate", "enterprise_churn_rate",
        "avg_tenure_churned", "inactive_rate",
    ]

    rows = []
    for name in scalar_metrics:
        metric = METRICS[name]
        try:
            value = metric.compute(df)
        except Exception:
            value = None
        rows.append({
            "metric_name":   metric.display_name,
            "value":         round(value, 2) if value is not None else None,
            "unit":          metric.unit,
            "owner":         metric.owner,
            "metric_id":     name,
        })
    return rows


@app.get("/powerbi/churn-by-industry", tags=["Power BI"])
async def powerbi_churn_by_industry():
    """
    **Power BI Ready — Churn Rate by Industry (Bar Chart)**

    Returns churn rate, customer count, and average spend per industry.
    Use for a clustered bar chart: industry on X-axis, churn % on Y-axis.

    Power BI: Get Data → Web → `http://localhost:8000/powerbi/churn-by-industry`
    """
    from utils.db_config import run_query
    df = run_query("""
        SELECT
            industry,
            COUNT(*)                                                      AS total_customers,
            SUM(CASE WHEN churned = 1 THEN 1 ELSE 0 END)                 AS churned_count,
            ROUND(100.0 * SUM(CASE WHEN churned = 1 THEN 1 ELSE 0 END)
                  / COUNT(*), 2)                                          AS churn_rate_pct,
            ROUND(AVG(monthly_spend), 2)                                  AS avg_monthly_spend,
            ROUND(AVG(clv), 2)                                            AS avg_clv
        FROM customers
        GROUP BY industry
        ORDER BY churn_rate_pct DESC
    """)
    return df.to_dict(orient="records")


@app.get("/powerbi/churn-by-plan", tags=["Power BI"])
async def powerbi_churn_by_plan():
    """
    **Power BI Ready — Churn Rate by Plan Type (Donut / Bar Chart)**

    Returns churn breakdown by subscription plan — Starter, Growth,
    Enterprise, Legacy. Use with a donut chart or stacked bar.

    Power BI: Get Data → Web → `http://localhost:8000/powerbi/churn-by-plan`
    """
    from utils.db_config import run_query
    df = run_query("""
        SELECT
            plan_type,
            COUNT(*)                                                      AS total_customers,
            SUM(CASE WHEN churned = 1 THEN 1 ELSE 0 END)                 AS churned_count,
            ROUND(100.0 * SUM(CASE WHEN churned = 1 THEN 1 ELSE 0 END)
                  / COUNT(*), 2)                                          AS churn_rate_pct,
            ROUND(SUM(monthly_spend), 2)                                  AS total_mrr,
            ROUND(AVG(clv), 2)                                            AS avg_clv
        FROM customers
        GROUP BY plan_type
        ORDER BY churn_rate_pct DESC
    """)
    return df.to_dict(orient="records")


@app.get("/powerbi/at-risk-customers", tags=["Power BI"])
async def powerbi_at_risk_customers():
    """
    **Power BI Ready — Top 50 At-Risk Active Customers (Table Visual)**

    Returns active customers most likely to churn, ranked by:
    - Days inactive (highest = most at risk)
    - Support ticket count (highest = most frustrated)
    - CLV (highest = most valuable to save)

    Use as a drill-through table: click a segment → see the exact accounts.

    Power BI: Get Data → Web → `http://localhost:8000/powerbi/at-risk-customers`
    """
    from utils.db_config import run_query
    import joblib, json
    import pandas as pd

    # Load model for live churn scoring
    MODEL_DIR = PROJECT_ROOT / "02_predictive_core" / "models"
    try:
        churn_model   = joblib.load(MODEL_DIR / "churn_model.pkl")
        with open(MODEL_DIR / "feature_names.json") as f:
            feature_names = json.load(f)
        model_loaded = True
    except Exception:
        model_loaded = False

    df = run_query("""
        SELECT
            customer_id, company_name, contact_name, email,
            industry, region, plan_type,
            monthly_spend, clv, tenure_months,
            support_tickets_count, last_login_days_ago, num_users,
            churned
        FROM customers
        WHERE churned = 0
        ORDER BY last_login_days_ago DESC, support_tickets_count DESC
        LIMIT 50
    """)

    # Score with XGBoost if model is available
    if model_loaded and not df.empty:
        try:
            fe = pd.DataFrame()
            fe["tenure_months"]         = df["tenure_months"].clip(lower=1)
            fe["monthly_spend"]         = df["monthly_spend"].clip(lower=0)
            fe["support_tickets_count"] = df["support_tickets_count"].clip(lower=0)
            fe["last_login_days_ago"]   = df["last_login_days_ago"].clip(lower=0)
            fe["num_users"]             = df["num_users"].clip(lower=1)
            fe["spend_per_user"]        = (fe["monthly_spend"] / fe["num_users"]).round(2)
            fe["ticket_rate"]           = (fe["support_tickets_count"] / fe["tenure_months"]).round(4)
            fe["is_new_customer"]       = (fe["tenure_months"] < 6).astype(int)
            fe["is_inactive"]           = (fe["last_login_days_ago"] > 30).astype(int)
            fe["is_legacy_plan"]        = (df["plan_type"] == "Legacy").astype(int)
            fe["revenue_risk"]          = (
                fe["monthly_spend"] * (fe["is_inactive"] + fe["ticket_rate"] + fe["is_new_customer"])
            ).round(2)
            plan_map = {"Starter": 0, "Growth": 1, "Enterprise": 2, "Legacy": 3}
            fe["plan_encoded"]     = df["plan_type"].map(plan_map).fillna(0).astype(int)
            fe["industry_encoded"] = 0
            fe["region_encoded"]   = 0
            fe = fe[feature_names]

            probs = churn_model.predict_proba(fe)[:, 1]
            df["churn_probability_pct"] = (probs * 100).round(1)
        except Exception:
            df["churn_probability_pct"] = None
    else:
        df["churn_probability_pct"] = None

    # Risk label
    def risk_label(prob):
        if prob is None: return "Unknown"
        if prob >= 70:   return "🔴 Critical"
        if prob >= 40:   return "🟡 High"
        return                  "🟢 Medium"

    df["risk_level"] = df["churn_probability_pct"].apply(risk_label)
    df = df.sort_values("churn_probability_pct", ascending=False)

    return df.to_dict(orient="records")


@app.get("/powerbi/churn-by-region", tags=["Power BI"])
async def powerbi_churn_by_region():
    """**Power BI Ready — Churn Rate by Region (Bar Chart)**"""
    from utils.db_config import run_query
    df = run_query("""
        SELECT
            region,
            COUNT(*)                                                      AS total_customers,
            SUM(CASE WHEN churned = 1 THEN 1 ELSE 0 END)                 AS churned_count,
            ROUND(100.0 * SUM(CASE WHEN churned = 1 THEN 1 ELSE 0 END)
                  / COUNT(*), 2)                                          AS churn_rate_pct,
            ROUND(AVG(monthly_spend), 2)                                  AS avg_monthly_spend,
            ROUND(SUM(monthly_spend), 2)                                  AS total_mrr
        FROM customers
        GROUP BY region
        ORDER BY churn_rate_pct DESC
    """)
    return df.to_dict(orient="records")


@app.get("/powerbi/support-issues", tags=["Power BI"])
async def powerbi_support_issues():
    """**Power BI Ready — Support Ticket Issue Type Breakdown**"""
    from utils.db_config import run_query
    df = run_query("""
        SELECT
            sl.issue_type,
            COUNT(*)                                        AS ticket_count,
            COUNT(DISTINCT sl.customer_id)                  AS unique_customers,
            SUM(CASE WHEN sl.sentiment = 'Negative' THEN 1 ELSE 0 END) AS negative_sentiment,
            SUM(CASE WHEN c.churned = 1 THEN 1 ELSE 0 END) AS churned_customers
        FROM support_logs sl
        JOIN customers c ON sl.customer_id = c.customer_id
        GROUP BY sl.issue_type
        ORDER BY ticket_count DESC
    """)
    return df.to_dict(orient="records")


@app.get("/powerbi/churned-customers", tags=["Power BI"])
async def powerbi_churned_customers():
    """**Power BI Ready — Lost Accounts (Churned Customers)**"""
    from utils.db_config import run_query
    df = run_query("""
        SELECT
            c.customer_id, c.company_name, c.contact_name, c.email,
            c.industry, c.region, c.plan_type,
            c.monthly_spend, c.clv, c.tenure_months,
            c.support_tickets_count, c.last_login_days_ago,
            c.num_users,
            ROUND(c.monthly_spend * 12, 2) AS annual_revenue_lost
        FROM customers c
        WHERE c.churned = 1
        ORDER BY c.monthly_spend DESC
        LIMIT 100
    """)
    return df.to_dict(orient="records")


@app.get("/powerbi/health-distribution", tags=["Power BI"])
async def powerbi_health_distribution():
    """**Power BI Ready — Customer Health Score Distribution (Histogram buckets)**"""
    from utils.db_config import run_query
    import joblib, json, pandas as pd

    MODEL_DIR = PROJECT_ROOT / "02_predictive_core" / "models"
    try:
        churn_model   = joblib.load(MODEL_DIR / "churn_model.pkl")
        with open(MODEL_DIR / "feature_names.json") as f:
            feature_names = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model load failed: {e}")

    df = run_query("""
        SELECT tenure_months, monthly_spend, support_tickets_count,
               last_login_days_ago, num_users, plan_type, churned
        FROM customers
        WHERE churned = 0
        LIMIT 2000
    """)

    fe = pd.DataFrame()
    fe["tenure_months"]         = df["tenure_months"].clip(lower=1)
    fe["monthly_spend"]         = df["monthly_spend"].clip(lower=0)
    fe["support_tickets_count"] = df["support_tickets_count"].clip(lower=0)
    fe["last_login_days_ago"]   = df["last_login_days_ago"].clip(lower=0)
    fe["num_users"]             = df["num_users"].clip(lower=1)
    fe["spend_per_user"]        = (fe["monthly_spend"] / fe["num_users"]).round(2)
    fe["ticket_rate"]           = (fe["support_tickets_count"] / fe["tenure_months"]).round(4)
    fe["is_new_customer"]       = (fe["tenure_months"] < 6).astype(int)
    fe["is_inactive"]           = (fe["last_login_days_ago"] > 30).astype(int)
    fe["is_legacy_plan"]        = (df["plan_type"] == "Legacy").astype(int)
    fe["revenue_risk"]          = (fe["monthly_spend"] * (fe["is_inactive"] + fe["ticket_rate"] + fe["is_new_customer"])).round(2)
    plan_map = {"Starter": 0, "Growth": 1, "Enterprise": 2, "Legacy": 3}
    fe["plan_encoded"]     = df["plan_type"].map(plan_map).fillna(0).astype(int)
    fe["industry_encoded"] = 0
    fe["region_encoded"]   = 0
    fe = fe[feature_names]

    probs = (churn_model.predict_proba(fe)[:, 1] * 100).round(1)

    # Return histogram buckets
    import numpy as np
    bins   = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    labels = ["0-10%","10-20%","20-30%","30-40%","40-50%","50-60%","60-70%","70-80%","80-90%","90-100%"]
    counts, _ = np.histogram(probs, bins=bins)
    return [{"bucket": labels[i], "count": int(counts[i]), "risk": "Critical" if i >= 6 else "High" if i >= 3 else "Safe"}
            for i in range(len(labels))]


# ---------------------------------------------------------------------------
# ERROR HANDLERS
# ---------------------------------------------------------------------------

@app.exception_handler(404)
async def not_found(request: Request, exc):
    return JSONResponse(
        status_code=404,
        content={
            "error":   "Endpoint not found",
            "message": f"'{request.url.path}' does not exist. Visit /docs to see all endpoints.",
        },
    )



# ---------------------------------------------------------------------------
# MAIN — Run with: python chat_bridge.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Nexus-ABI Chat Bridge on http://localhost:8000")
    logger.info("API Docs: http://localhost:8000/docs")
    uvicorn.run(
        "chat_bridge:app",
        host        = "0.0.0.0",
        port        = 8000,
        reload      = True,     # Auto-restart on code changes
        log_level   = "warning",  # Suppress uvicorn's own logs (we use loguru)
    )
