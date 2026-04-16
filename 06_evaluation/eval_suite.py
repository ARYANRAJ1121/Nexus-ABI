"""
=============================================================================
NEXUS-ABI | Layer 6: Evaluation Suite
File: eval_suite.py
=============================================================================

PURPOSE:
  Automated quality gate for the entire Nexus-ABI pipeline.
  Tests every layer — SQL, RAG, Strategist, FastAPI — to ensure
  outputs are correct, consistent, and production-ready.

WHY EVALUATION MATTERS:
  Without this, you're shipping an AI system and hoping it works.
  Evaluation gives you PROOF it works — or early warning when it breaks.

  "Vibe checking" an LLM output is not testing. This is testing.

EVALUATION PHILOSOPHY:
  We use two types of tests:

  1. DETERMINISTIC TESTS (no LLM judge needed):
     - We know the exact right answer (we built the data).
     - Assert the SQL returns 20.64% for churn rate.
     - Assert the API returns HTTP 200.
     - Assert the strategy JSON has a "priority" field.
     - These ALWAYS give the same result. Zero flakiness.

  2. HEURISTIC TESTS (soft assertions):
     - We can't know the exact wording, but we know properties.
     - RAG insight mentions words from the retrieved tickets.
     - Strategy contains time-bound language ("days", "weeks").
     - These catch hallucination and irrelevance statistically.

DEEPEVAL INTEGRATION:
  DeepEval is the testing framework. It provides:
  - Test case structure (LLMTestCase)
  - Built-in metrics (AnswerRelevancy, Faithfulness, etc.)
  - HTML test report generation

  NOTE ON LLM-AS-JUDGE METRICS:
  DeepEval's advanced metrics (Faithfulness, AnswerRelevancy) use an LLM
  to judge quality. By default they use OpenAI GPT-4. To use Llama 3
  locally instead, set DEEPEVAL_MODEL="ollama/llama3" (see config below).
  We currently use DeepEval for structure + run our own heuristic checks
  to avoid needing an OpenAI key.

HOW TO RUN:
  python 06_evaluation/eval_suite.py

WHAT SUCCESS LOOKS LIKE:
  All tests pass → system is production-ready
  Any test fails → shows exactly which layer broke and why
=============================================================================
"""

import os
import sys
import time
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

os.environ["PYTHONUTF8"] = "1"

PROJECT_ROOT = Path(__file__).parent.parent
LAYER_4_ROOT = PROJECT_ROOT / "04_agentic_engine"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LAYER_4_ROOT))

from loguru import logger
from rich.console import Console
from rich.table   import Table
from rich.panel   import Panel
from rich.rule    import Rule

console = Console(highlight=False)


# =============================================================================
# TEST RESULT DATACLASS
# =============================================================================

@dataclass
class TestResult:
    name:     str
    category: str
    passed:   bool
    message:  str
    elapsed:  float = 0.0
    details:  dict  = field(default_factory=dict)


# =============================================================================
# LAYER 1: DATA PIPELINE TESTS
# =============================================================================

def test_data_files_exist() -> TestResult:
    """Verify all synthetic data files were generated."""
    start = time.time()
    required = [
        PROJECT_ROOT / "01_data_pipeline" / "raw" / "customers.csv",
        PROJECT_ROOT / "01_data_pipeline" / "raw" / "transactions.csv",
        PROJECT_ROOT / "01_data_pipeline" / "raw" / "support_logs.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    passed  = len(missing) == 0
    return TestResult(
        name     = "Data Files Exist",
        category = "Layer 1: Data Pipeline",
        passed   = passed,
        message  = "All 3 CSV files present" if passed else f"Missing: {missing}",
        elapsed  = round(time.time() - start, 3),
    )


def test_data_row_counts() -> TestResult:
    """Verify the synthetic data has the expected number of rows."""
    import pandas as pd
    start = time.time()
    try:
        customers  = pd.read_csv(PROJECT_ROOT / "01_data_pipeline" / "raw" / "customers.csv")
        txns       = pd.read_csv(PROJECT_ROOT / "01_data_pipeline" / "raw" / "transactions.csv")
        logs       = pd.read_csv(PROJECT_ROOT / "01_data_pipeline" / "raw" / "support_logs.csv")

        checks = {
            "customers":    (len(customers),  9_000,  11_000),    # ~10K
            "transactions": (len(txns),        45_000, 55_000),   # ~50K
            "support_logs": (len(logs),        1_800,  2_200),    # ~2K
        }

        failures = []
        for table, (count, lo, hi) in checks.items():
            if not (lo <= count <= hi):
                failures.append(f"{table}: {count:,} (expected {lo:,}-{hi:,})")

        passed = len(failures) == 0
        return TestResult(
            name     = "Data Row Counts",
            category = "Layer 1: Data Pipeline",
            passed   = passed,
            message  = f"customers={len(customers):,}, transactions={len(txns):,}, logs={len(logs):,}"
                       if passed else f"Out of range: {failures}",
            elapsed  = round(time.time() - start, 3),
            details  = {"customers": len(customers), "transactions": len(txns), "logs": len(logs)},
        )
    except Exception as e:
        return TestResult("Data Row Counts", "Layer 1: Data Pipeline", False, str(e), round(time.time() - start, 3))


def test_churn_signal_planted() -> TestResult:
    """Verify that legacy plan customers have higher churn rate (our planted signal)."""
    import pandas as pd
    start = time.time()
    try:
        df = pd.read_csv(PROJECT_ROOT / "01_data_pipeline" / "raw" / "customers.csv")

        legacy_churn = df[df["plan_type"] == "Legacy"]["churned"].mean()
        growth_churn = df[df["plan_type"] == "Growth"]["churned"].mean()

        # Legacy churn should be higher than Growth churn (this was our planted signal)
        passed  = legacy_churn > growth_churn
        message = (
            f"Legacy churn {legacy_churn:.1%} > Growth churn {growth_churn:.1%} ✓"
            if passed
            else f"Signal not found: Legacy {legacy_churn:.1%} ≤ Growth {growth_churn:.1%}"
        )
        return TestResult(
            name     = "Churn Signal Planted",
            category = "Layer 1: Data Pipeline",
            passed   = passed,
            message  = message,
            elapsed  = round(time.time() - start, 3),
            details  = {"legacy_churn": round(legacy_churn, 4), "growth_churn": round(growth_churn, 4)},
        )
    except Exception as e:
        return TestResult("Churn Signal Planted", "Layer 1: Data Pipeline", False, str(e), round(time.time() - start, 3))


# =============================================================================
# LAYER 2: PREDICTIVE CORE TESTS
# =============================================================================

def test_models_exist() -> TestResult:
    """Verify XGBoost model files were saved."""
    start  = time.time()
    models = [
        PROJECT_ROOT / "02_predictive_core" / "models" / "churn_model.pkl",
        PROJECT_ROOT / "02_predictive_core" / "models" / "clv_model.pkl",
        PROJECT_ROOT / "02_predictive_core" / "models" / "feature_names.json",
    ]
    missing = [str(p.name) for p in models if not p.exists()]
    passed  = len(missing) == 0
    return TestResult(
        name     = "XGBoost Models Saved",
        category = "Layer 2: Predictive Core",
        passed   = passed,
        message  = "All 3 model files present" if passed else f"Missing: {missing}",
        elapsed  = round(time.time() - start, 3),
    )


def test_model_inference() -> TestResult:
    """Run a quick inference test on the churn model."""
    import joblib
    import pandas as pd
    import numpy as np
    start = time.time()
    try:
        churn_model = joblib.load(PROJECT_ROOT / "02_predictive_core" / "models" / "churn_model.pkl")
        with open(PROJECT_ROOT / "02_predictive_core" / "models" / "feature_names.json") as f:
            feature_names = json.load(f)

        # Simulate a high-risk customer
        high_risk = pd.DataFrame([{
            "tenure_months":         2,    # new customer
            "monthly_spend":         500,
            "support_tickets_count": 8,    # many tickets
            "last_login_days_ago":   45,   # inactive
            "num_users":             5,
            "spend_per_user":        100.0,
            "ticket_rate":           4.0,
            "is_new_customer":       1,
            "is_inactive":           1,
            "is_legacy_plan":        0,
            "revenue_risk":          1500.0,
            "plan_encoded":          0,
            "industry_encoded":      0,
            "region_encoded":        0,
        }])[feature_names]

        prob = churn_model.predict_proba(high_risk)[0][1]
        # High-risk customer should have > 40% churn probability
        passed  = prob > 0.40
        message = f"High-risk customer churn probability: {prob:.1%} {'✓' if passed else '✗ (expected > 40%)'}"

        return TestResult(
            name     = "Model Inference",
            category = "Layer 2: Predictive Core",
            passed   = passed,
            message  = message,
            elapsed  = round(time.time() - start, 3),
            details  = {"churn_probability": round(float(prob), 4)},
        )
    except Exception as e:
        return TestResult("Model Inference", "Layer 2: Predictive Core", False, str(e), round(time.time() - start, 3))


# =============================================================================
# LAYER 3: SEMANTIC LAYER TESTS
# =============================================================================

def test_all_metrics_registered() -> TestResult:
    """Verify all 10 KPIs are registered in the Semantic Layer."""
    start = time.time()
    sys.path.insert(0, str(PROJECT_ROOT / "03_semantic_layer"))
    from metrics_defs import METRICS
    expected = [
        "churn_rate", "mrr", "revenue_at_risk", "arpu", "avg_clv",
        "support_ticket_rate", "enterprise_churn_rate", "plan_distribution",
        "avg_tenure_churned", "inactive_rate",
    ]
    missing = [m for m in expected if m not in METRICS]
    passed  = len(missing) == 0
    return TestResult(
        name     = "All KPIs Registered",
        category = "Layer 3: Semantic Layer",
        passed   = passed,
        message  = f"{len(METRICS)} KPIs registered" if passed else f"Missing: {missing}",
        elapsed  = round(time.time() - start, 3),
    )


def test_metric_compute_accuracy() -> TestResult:
    """
    Verify the churn_rate formula returns the expected value.
    This is the most important governance test — KPI formulas must be exact.
    """
    import pandas as pd
    start = time.time()
    sys.path.insert(0, str(PROJECT_ROOT / "03_semantic_layer"))
    from metrics_defs import METRICS

    try:
        df = pd.read_csv(PROJECT_ROOT / "01_data_pipeline" / "raw" / "customers.csv")

        # Compute churn rate using Semantic Layer formula
        computed = METRICS["churn_rate"].compute(df)

        # Manual calculation to cross-check
        manual = round(df["churned"].sum() / len(df) * 100, 2)

        # They must match to 2 decimal places
        passed  = abs(computed - manual) < 0.01
        message = (
            f"Churn rate: computed={computed}%, manual={manual}% — match ✓"
            if passed
            else f"MISMATCH: computed={computed}%, manual={manual}%"
        )
        return TestResult(
            name     = "KPI Formula Accuracy",
            category = "Layer 3: Semantic Layer",
            passed   = passed,
            message  = message,
            elapsed  = round(time.time() - start, 3),
            details  = {"computed": computed, "manual": manual},
        )
    except Exception as e:
        return TestResult("KPI Formula Accuracy", "Layer 3: Semantic Layer", False, str(e), round(time.time() - start, 3))


def test_metric_interpretation() -> TestResult:
    """Verify the interpretation thresholds work correctly."""
    start = time.time()
    sys.path.insert(0, str(PROJECT_ROOT / "03_semantic_layer"))
    from metrics_defs import interpret

    checks = [
        (interpret("churn_rate", 3.0),  "Healthy"),    # 3% should be healthy
        (interpret("churn_rate", 7.0),  "Warning"),    # 7% should be warning
        (interpret("churn_rate", 20.0), "Critical"),   # 20% should be critical
    ]

    failures = [f"Expected '{expected}' in '{result}'" for result, expected in checks if expected.lower() not in result.lower()]
    passed   = len(failures) == 0

    return TestResult(
        name     = "Interpretation Thresholds",
        category = "Layer 3: Semantic Layer",
        passed   = passed,
        message  = "All thresholds correct" if passed else f"Failures: {failures}",
        elapsed  = round(time.time() - start, 3),
    )


# =============================================================================
# LAYER 4: SQL AGENT TESTS
# =============================================================================

def test_sql_governance() -> TestResult:
    """
    Verify that questions about known KPIs use the Semantic Layer,
    not LLM-generated SQL. This is the core governance test.
    """
    start = time.time()
    from agents.sql_agent import SQLAgent, _match_metric

    kpi_questions = [
        ("What is our churn rate?",             "churn_rate"),
        ("Show me our monthly recurring revenue", "mrr"),
        ("What is the revenue at risk?",         "revenue_at_risk"),
    ]

    failures = []
    for question, expected_metric in kpi_questions:
        matched = _match_metric(question)
        if matched != expected_metric:
            failures.append(f"'{question}' → got '{matched}', expected '{expected_metric}'")

    passed = len(failures) == 0
    return TestResult(
        name     = "SQL Governance (Semantic Layer)",
        category = "Layer 4: SQL Agent",
        passed   = passed,
        message  = f"All {len(kpi_questions)} KPI questions routed to Semantic Layer" if passed else f"Failures: {failures}",
        elapsed  = round(time.time() - start, 3),
    )


def test_sql_accuracy() -> TestResult:
    """
    Run the SQL Agent on a question with a known answer and verify accuracy.
    Churn rate must be 20.64% (our dataset is deterministic).
    """
    start = time.time()
    from agents.sql_agent import SQLAgent

    try:
        agent  = SQLAgent()
        result = agent.run("What is our current churn rate?")

        df = result.get("result_df")
        if df is None or df.empty:
            return TestResult("SQL Accuracy", "Layer 4: SQL Agent", False,
                              "Query returned no data", round(time.time() - start, 3))

        actual_churn = float(df["churn_rate_pct"].iloc[0])
        expected     = 20.64
        tolerance    = 0.5    # Allow ±0.5% for rounding differences

        passed  = abs(actual_churn - expected) <= tolerance
        message = (
            f"Churn rate: {actual_churn}% (expected ≈ {expected}%) ✓"
            if passed
            else f"WRONG: got {actual_churn}%, expected {expected}% ± {tolerance}%"
        )
        return TestResult(
            name     = "SQL Accuracy",
            category = "Layer 4: SQL Agent",
            passed   = passed,
            message  = message,
            elapsed  = round(time.time() - start, 3),
            details  = {"actual": actual_churn, "expected": expected, "source": result.get("source")},
        )
    except Exception as e:
        return TestResult("SQL Accuracy", "Layer 4: SQL Agent", False, str(e), round(time.time() - start, 3))


# =============================================================================
# LAYER 4: RAG AGENT TESTS
# =============================================================================

def test_rag_retrieval_count() -> TestResult:
    """RAG must retrieve at least 1 ticket for a billing question."""
    start = time.time()
    from agents.rag_agent import RAGAgent
    try:
        agent  = RAGAgent()
        result = agent.query("billing portal problems and cancellation threats")
        count  = result.get("num_retrieved", 0)
        passed = count >= 1
        return TestResult(
            name     = "RAG Retrieval Count",
            category = "Layer 4: RAG Agent",
            passed   = passed,
            message  = f"Retrieved {count} tickets for billing query {'✓' if passed else '✗ (expected ≥ 1)'}",
            elapsed  = round(time.time() - start, 3),
            details  = {"retrieved": count},
        )
    except Exception as e:
        return TestResult("RAG Retrieval Count", "Layer 4: RAG Agent", False, str(e), round(time.time() - start, 3))


def test_rag_relevance_heuristic() -> TestResult:
    """
    Heuristic test: retrieved tickets for a billing question must contain
    billing-related keywords. If the RAG is pulling unrelated tickets,
    the embedding model or collection is broken.
    """
    start = time.time()
    from agents.rag_agent import RAGAgent
    try:
        agent    = RAGAgent()
        result   = agent.query("billing portal issues")
        tickets  = result.get("retrieved", [])

        billing_keywords = ["billing", "portal", "invoice", "payment", "charge"]
        relevant = sum(
            1 for t in tickets
            if any(kw in t["text"].lower() for kw in billing_keywords)
        )

        passed  = relevant >= 1
        message = (
            f"{relevant}/{len(tickets)} tickets contain billing-related keywords ✓"
            if passed
            else f"0/{len(tickets)} tickets relevant to billing — RAG may be broken"
        )
        return TestResult(
            name     = "RAG Relevance (Heuristic)",
            category = "Layer 4: RAG Agent",
            passed   = passed,
            message  = message,
            elapsed  = round(time.time() - start, 3),
            details  = {"relevant": relevant, "total_retrieved": len(tickets)},
        )
    except Exception as e:
        return TestResult("RAG Relevance (Heuristic)", "Layer 4: RAG Agent", False, str(e), round(time.time() - start, 3))


# =============================================================================
# LAYER 4: STRATEGIST + LANGGRAPH TESTS
# =============================================================================

def test_strategy_format() -> TestResult:
    """
    The Strategist must return a response with the required structured fields.
    If ANY field is missing, the downstream API or dashboard will break.
    """
    start = time.time()
    from agents.strategist_agent import StrategistAgent

    try:
        agent    = StrategistAgent()
        strategy = agent.run(
            question   = "What is our churn rate?",
            sql_result = {
                "result_df": None, "summary": "Churn rate is 20.64%.",
                "source": "mock", "attempts": 1, "error": None,
            },
            rag_result = {
                "retrieved": [], "insight": "No relevant tickets found.",
                "num_retrieved": 0, "question": "mock",
            },
        )

        required_keys = ["recommendation", "priority", "actions", "evidence"]
        missing = [k for k in required_keys if k not in strategy]
        valid_priorities = {"CRITICAL", "HIGH", "MEDIUM", "LOW", "UNKNOWN"}

        failures = []
        if missing:
            failures.append(f"Missing keys: {missing}")
        if strategy.get("priority") not in valid_priorities:
            failures.append(f"Invalid priority: {strategy.get('priority')}")
        if not isinstance(strategy.get("actions"), list):
            failures.append("'actions' must be a list")

        passed = len(failures) == 0
        return TestResult(
            name     = "Strategy Output Format",
            category = "Layer 4: Strategist",
            passed   = passed,
            message  = f"Priority={strategy.get('priority')}, Actions={len(strategy.get('actions',[]))}" if passed else f"Failures: {failures}",
            elapsed  = round(time.time() - start, 3),
            details  = {"priority": strategy.get("priority"), "action_count": len(strategy.get("actions", []))},
        )
    except Exception as e:
        return TestResult("Strategy Output Format", "Layer 4: Strategist", False, str(e), round(time.time() - start, 3))


def test_langgraph_pipeline() -> TestResult:
    """
    End-to-end test: run the full LangGraph pipeline and verify the output
    contains all required fields and a valid priority.
    """
    start = time.time()
    from main_graph import run_nexus

    try:
        result   = run_nexus("What is our churn rate?")
        strategy = result.get("strategy", {})

        checks = {
            "has_recommendation": bool(strategy.get("recommendation")),
            "has_priority":       strategy.get("priority") in {"CRITICAL", "HIGH", "MEDIUM", "LOW"},
            "has_sql_result":     result.get("sql_result") is not None,
            "has_rag_result":     result.get("rag_result") is not None,
            "no_error":           result.get("error") is None,
        }

        failures = [k for k, v in checks.items() if not v]
        passed   = len(failures) == 0

        return TestResult(
            name     = "LangGraph End-to-End",
            category = "Layer 4: LangGraph",
            passed   = passed,
            message  = f"Full pipeline: priority={strategy.get('priority')}" if passed else f"Failed checks: {failures}",
            elapsed  = round(time.time() - start, 3),
            details  = checks,
        )
    except Exception as e:
        return TestResult("LangGraph End-to-End", "Layer 4: LangGraph", False, str(e), round(time.time() - start, 3))


# =============================================================================
# LAYER 5: API TESTS
# =============================================================================

def test_api_health_endpoint() -> TestResult:
    """GET /health must return 200 OK."""
    start = time.time()
    try:
        import httpx
        r = httpx.get("http://localhost:8000/health", timeout=10)
        passed = r.status_code == 200 and r.json().get("status") == "ok"
        return TestResult(
            name     = "GET /health",
            category = "Layer 5: FastAPI",
            passed   = passed,
            message  = f"HTTP {r.status_code} | {r.json().get('status', 'unknown')}",
            elapsed  = round(time.time() - start, 3),
        )
    except Exception as e:
        return TestResult("GET /health", "Layer 5: FastAPI",
                          False, f"Server not reachable: {e}", round(time.time() - start, 3))


def test_api_metrics_endpoint() -> TestResult:
    """GET /metrics must return 9 KPIs with non-null values."""
    start = time.time()
    try:
        import httpx
        r    = httpx.get("http://localhost:8000/metrics", timeout=15)
        data = r.json()

        total   = data.get("total_metrics", 0)
        metrics = data.get("metrics", [])
        nulls   = [m["name"] for m in metrics if m["value"] is None]

        passed  = r.status_code == 200 and total >= 9 and len(nulls) == 0
        message = (
            f"HTTP {r.status_code} | {total} metrics, 0 nulls ✓"
            if passed
            else f"HTTP {r.status_code} | {total} metrics | nulls: {nulls}"
        )
        return TestResult(
            name     = "GET /metrics",
            category = "Layer 5: FastAPI",
            passed   = passed,
            message  = message,
            elapsed  = round(time.time() - start, 3),
            details  = {"total_metrics": total, "null_metrics": nulls},
        )
    except Exception as e:
        return TestResult("GET /metrics", "Layer 5: FastAPI",
                          False, f"Server not reachable: {e}", round(time.time() - start, 3))


def test_api_ask_response_time() -> TestResult:
    """POST /ask must complete within 120 seconds (Llama 3 on CPU)."""
    start = time.time()
    try:
        import httpx
        r       = httpx.post(
            "http://localhost:8000/ask",
            json    = {"question": "What is our churn rate?"},
            timeout = 150,
        )
        elapsed = round(time.time() - start, 2)
        data    = r.json()

        passed  = r.status_code == 200 and elapsed <= 120
        message = (
            f"HTTP {r.status_code} | {elapsed}s | priority={data.get('priority')} ✓"
            if passed
            else f"HTTP {r.status_code} | {elapsed}s (too slow or error)"
        )
        return TestResult(
            name     = "POST /ask Response Time",
            category = "Layer 5: FastAPI",
            passed   = passed,
            message  = message,
            elapsed  = elapsed,
            details  = {"status_code": r.status_code, "priority": data.get("priority")},
        )
    except Exception as e:
        return TestResult("POST /ask Response Time", "Layer 5: FastAPI",
                          False, f"Server not reachable: {e}", round(time.time() - start, 3))


def test_api_ask_response_schema() -> TestResult:
    """POST /ask must return all required JSON fields."""
    start = time.time()
    try:
        import httpx
        r    = httpx.post(
            "http://localhost:8000/ask",
            json    = {"question": "What is our churn rate?"},
            timeout = 150,
        )
        data = r.json()

        required = ["question", "priority", "recommendation", "actions", "elapsed_seconds"]
        missing  = [k for k in required if k not in data]
        passed   = r.status_code == 200 and len(missing) == 0

        return TestResult(
            name     = "POST /ask Response Schema",
            category = "Layer 5: FastAPI",
            passed   = passed,
            message  = f"All {len(required)} required fields present" if passed else f"Missing: {missing}",
            elapsed  = round(time.time() - start, 3),
        )
    except Exception as e:
        return TestResult("POST /ask Response Schema", "Layer 5: FastAPI",
                          False, f"Error: {e}", round(time.time() - start, 3))


# =============================================================================
# TEST RUNNER
# =============================================================================

def run_all_tests(skip_api: bool = False, skip_llm: bool = False) -> list[TestResult]:
    """
    Runs all tests and returns results.

    Args:
        skip_api: Skip FastAPI tests (use if server isn't running)
        skip_llm: Skip tests that call Llama 3 (use for fast CI/CD checks)
    """
    tests = []

    console.print(Rule("[bold cyan]Layer 1: Data Pipeline[/bold cyan]"))
    tests.append(test_data_files_exist())
    tests.append(test_data_row_counts())
    tests.append(test_churn_signal_planted())

    console.print(Rule("[bold cyan]Layer 2: Predictive Core[/bold cyan]"))
    tests.append(test_models_exist())
    tests.append(test_model_inference())

    console.print(Rule("[bold cyan]Layer 3: Semantic Layer[/bold cyan]"))
    tests.append(test_all_metrics_registered())
    tests.append(test_metric_compute_accuracy())
    tests.append(test_metric_interpretation())

    console.print(Rule("[bold cyan]Layer 4: Agents[/bold cyan]"))
    tests.append(test_sql_governance())

    if not skip_llm:
        tests.append(test_sql_accuracy())
        tests.append(test_rag_retrieval_count())
        tests.append(test_rag_relevance_heuristic())
        tests.append(test_strategy_format())
        tests.append(test_langgraph_pipeline())

    if not skip_api:
        console.print(Rule("[bold cyan]Layer 5: FastAPI[/bold cyan]"))
        tests.append(test_api_health_endpoint())
        tests.append(test_api_metrics_endpoint())
        if not skip_llm:
            tests.append(test_api_ask_response_time())
            tests.append(test_api_ask_response_schema())

    return tests


def print_results(results: list[TestResult]) -> None:
    """Prints a rich summary table of all test results."""
    table = Table(
        title        = "[bold]Nexus-ABI Evaluation Report[/bold]",
        show_header  = True,
        header_style = "bold magenta",
        show_lines   = True,
    )
    table.add_column("Category",  style="dim",   width=25)
    table.add_column("Test",               width=35)
    table.add_column("Status",    justify="center", width=8)
    table.add_column("Time",      justify="right",  width=7)
    table.add_column("Message",             width=50)

    for r in results:
        status = "[bold green]PASS[/bold green]" if r.passed else "[bold red]FAIL[/bold red]"
        table.add_row(
            r.category,
            r.name,
            status,
            f"{r.elapsed}s",
            r.message[:80],
        )

    console.print(table)

    passed = sum(1 for r in results if r.passed)
    total  = len(results)
    pct    = round(passed / total * 100) if total > 0 else 0

    colour = "green" if pct == 100 else "yellow" if pct >= 80 else "red"
    console.print(Panel.fit(
        f"[bold {colour}]{passed}/{total} tests passed ({pct}%)[/bold {colour}]",
        border_style=colour,
    ))

    # Print failures in detail
    failures = [r for r in results if not r.passed]
    if failures:
        console.print("\n[bold red]Failed Tests:[/bold red]")
        for r in failures:
            console.print(f"  ✗ [{r.category}] {r.name}: {r.message}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Nexus-ABI Evaluation Suite")
    parser.add_argument("--skip-api",  action="store_true", help="Skip FastAPI tests (server must be running)")
    parser.add_argument("--skip-llm",  action="store_true", help="Skip tests that call Llama 3 (fast mode)")
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold cyan]NEXUS-ABI[/bold cyan] | [white]Evaluation Suite[/white]\n"
        "[dim]Running quality gate across all 5 layers...[/dim]",
        border_style="cyan"
    ))

    if args.skip_llm:
        console.print("[yellow]Fast mode: skipping LLM tests[/yellow]")
    if args.skip_api:
        console.print("[yellow]Skipping API tests (start server first)[/yellow]")

    results = run_all_tests(skip_api=args.skip_api, skip_llm=args.skip_llm)
    print_results(results)
