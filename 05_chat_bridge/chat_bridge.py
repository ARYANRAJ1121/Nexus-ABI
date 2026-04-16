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
