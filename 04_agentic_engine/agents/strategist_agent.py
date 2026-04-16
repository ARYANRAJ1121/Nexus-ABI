"""
=============================================================================
NEXUS-ABI | Layer 4: Agentic Engine -> Agents
File: strategist_agent.py
=============================================================================

PURPOSE:
  The Strategist Agent. The "CEO advisor" of the system.
  It receives outputs from the SQL Agent and RAG Agent, combines them
  with XGBoost churn predictions and Semantic Layer interpretations,
  and generates a structured, actionable business recommendation.

WHY THIS IS THE "ELITE" LAYER:
  Without this agent, you have two isolated outputs:
    SQL Agent:  "Churn rate is 20.64%"
    RAG Agent:  "Billing portal issues in Media & Entertainment"

  The Strategist synthesises:
    "Your 20.64% churn rate is critical (Semantic Layer threshold: >10%).
     The root cause is billing portal outages in Media & Entertainment
     (evidence: 3 cited tickets from Kimberly Lin, Evan Hobbs, Lisa Reilly).
     XGBoost flags 2 of them as >70% churn probability with CLV > $8,000.
     Recommended action: emergency patch billing portal + CSM outreach to
     these 3 accounts within 48 hours."

  That is a complete, evidence-backed strategy — not a KPI dashboard.

INPUTS:
  - question      (str)  → the original business question
  - sql_result    (dict) → output from SQLAgent.run()
  - rag_result    (dict) → output from RAGAgent.query()
  - customer_df   (DataFrame, optional) → for XGBoost inference on at-risk accounts

OUTPUT: dict with keys:
  recommendation  → the full strategic recommendation (str)
  priority        → "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
  actions         → list of specific recommended actions
  evidence        → sources used (sql, rag, model, semantic_layer)
=============================================================================
"""

import os
import sys
import json
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"

PROJECT_ROOT = Path(__file__).parent.parent.parent
LAYER_4_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LAYER_4_ROOT))

import joblib
import pandas as pd
from loguru import logger
from langchain_core.prompts import ChatPromptTemplate

from utils.llm_config import get_llm

# Semantic Layer
sys.path.insert(0, str(PROJECT_ROOT / "03_semantic_layer"))
from metrics_defs import interpret, METRICS

# ---------------------------------------------------------------------------
# MODEL PATHS
# ---------------------------------------------------------------------------
MODEL_DIR          = PROJECT_ROOT / "02_predictive_core" / "models"
CHURN_MODEL_PATH   = MODEL_DIR / "churn_model.pkl"
CLV_MODEL_PATH     = MODEL_DIR / "clv_model.pkl"
FEATURE_NAMES_PATH = MODEL_DIR / "feature_names.json"


# ---------------------------------------------------------------------------
# PROMPT TEMPLATE
# ---------------------------------------------------------------------------

STRATEGY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are the Chief Strategy Officer of NexaCorp, a B2B SaaS company.
You have access to data from three sources:
  1. SQL data (precise KPIs and numbers)
  2. Support ticket evidence (real customer voices)
  3. ML model predictions (churn probability and CLV)
  4. Semantic Layer interpretations (business thresholds)

Your job is to synthesise all sources into ONE clear strategic recommendation.

FORMAT YOUR RESPONSE EXACTLY AS:

PRIORITY: [CRITICAL / HIGH / MEDIUM / LOW]

SITUATION:
[2-3 sentences: what the data tells us, citing specific numbers]

ROOT CAUSE:
[1-2 sentences: what the support evidence reveals as the underlying reason]

RECOMMENDED ACTIONS:
1. [Specific, time-bound action]
2. [Specific, time-bound action]
3. [Specific, time-bound action]

FINANCIAL IMPACT:
[1 sentence: revenue at risk or opportunity, using the numbers provided]

Be direct. No hedging. Every claim must be grounded in the data provided."""),

    ("human", """BUSINESS QUESTION: {question}

--- SQL DATA ---
{sql_summary}
Raw numbers: {sql_data}

--- SUPPORT TICKET EVIDENCE (RAG) ---
{rag_insight}
Tickets retrieved: {num_tickets}

--- XGBOOST MODEL PREDICTIONS ---
{model_predictions}

--- SEMANTIC LAYER INTERPRETATION ---
{semantic_interpretation}

Strategic Recommendation:"""),
])


# =============================================================================
# STRATEGIST AGENT CLASS
# =============================================================================

class StrategistAgent:
    """
    The Strategist Agent — synthesises SQL + RAG + ML + Semantic Layer
    into one executive-grade recommendation.

    Usage:
        sql_agent  = SQLAgent()
        rag_agent  = RAGAgent()
        strategist = StrategistAgent()

        sql_result = sql_agent.run("What is our churn situation?")
        rag_result = rag_agent.query("Why are customers leaving?")
        strategy   = strategist.run(
            question   = "What should we do about churn?",
            sql_result = sql_result,
            rag_result = rag_result,
        )
        print(strategy["recommendation"])
    """

    def __init__(self):
        self.llm          = get_llm(temperature=0.3)  # Slight creativity for strategy
        self.churn_model  = None
        self.clv_model    = None
        self.feature_names = None
        self._load_models()
        logger.info("StrategistAgent initialised.")

    def _load_models(self):
        """Loads the XGBoost models trained in Layer 2."""
        try:
            self.churn_model   = joblib.load(CHURN_MODEL_PATH)
            self.clv_model     = joblib.load(CLV_MODEL_PATH)
            with open(FEATURE_NAMES_PATH) as f:
                self.feature_names = json.load(f)
            logger.success("XGBoost models loaded from {}", MODEL_DIR)
        except FileNotFoundError:
            logger.warning(
                "XGBoost models not found. Run: python 02_predictive_core/train_churn.py\n"
                "Continuing without model predictions."
            )

    def _get_model_predictions(self, customer_df: pd.DataFrame | None) -> str:
        """
        Runs XGBoost on a sample of customers and returns a formatted
        prediction summary that the Strategist uses as evidence.

        If no customer_df is provided, loads a sample from the database.
        """
        if self.churn_model is None:
            return "XGBoost models not loaded. Run train_churn.py to enable predictions."

        if customer_df is None:
            # Load a sample of high-risk customers from the DB
            try:
                sys.path.insert(0, str(LAYER_4_ROOT))
                from utils.db_config import run_query
                customer_df = run_query("""
                    SELECT * FROM customers
                    WHERE churned = 0
                    ORDER BY support_tickets_count DESC, last_login_days_ago DESC
                    LIMIT 20
                """)
            except Exception as e:
                logger.warning("Could not load customer data for predictions: {}", e)
                return "Could not compute real-time churn predictions."

        if customer_df.empty:
            return "No active customers to score."

        try:
            # Engineer the same features as training
            fe = pd.DataFrame()
            fe["tenure_months"]         = customer_df["tenure_months"].clip(lower=1)
            fe["monthly_spend"]         = customer_df["monthly_spend"].clip(lower=0)
            fe["support_tickets_count"] = customer_df["support_tickets_count"].clip(lower=0)
            fe["last_login_days_ago"]   = customer_df["last_login_days_ago"].clip(lower=0)
            fe["num_users"]             = customer_df["num_users"].clip(lower=1)
            fe["spend_per_user"]        = (fe["monthly_spend"] / fe["num_users"]).round(2)
            fe["ticket_rate"]           = (fe["support_tickets_count"] / fe["tenure_months"]).round(4)
            fe["is_new_customer"]       = (fe["tenure_months"] < 6).astype(int)
            fe["is_inactive"]           = (fe["last_login_days_ago"] > 30).astype(int)
            fe["is_legacy_plan"]        = (customer_df["plan_type"] == "Legacy").astype(int)
            fe["revenue_risk"]          = (
                fe["monthly_spend"] * (fe["is_inactive"] + fe["ticket_rate"] + fe["is_new_customer"])
            ).round(2)

            plan_map = {"Starter": 0, "Growth": 1, "Enterprise": 2, "Legacy": 3}
            fe["plan_encoded"]      = customer_df["plan_type"].map(plan_map).fillna(0).astype(int)
            fe["industry_encoded"]  = 0   # Simplified for runtime scoring
            fe["region_encoded"]    = 0

            # Align to training feature order
            fe = fe[self.feature_names]

            # Predict
            churn_probs = self.churn_model.predict_proba(fe)[:, 1]
            clv_preds   = self.clv_model.predict(fe)

            # Build result
            results = customer_df[["company_name", "plan_type", "monthly_spend"]].copy()
            results["churn_probability"] = (churn_probs * 100).round(1)
            results["predicted_clv"]     = clv_preds.round(0)
            results = results.sort_values("churn_probability", ascending=False).head(5)

            # Format for the LLM prompt
            lines = ["Top 5 accounts by churn risk:"]
            for _, r in results.iterrows():
                lines.append(
                    f"  - {r['company_name']} ({r['plan_type']}): "
                    f"{r['churn_probability']}% churn risk | "
                    f"${r['monthly_spend']:,.0f}/mo | "
                    f"CLV ${r['predicted_clv']:,.0f}"
                )
            return "\n".join(lines)

        except Exception as e:
            logger.warning("Model prediction failed: {}", e)
            return f"Prediction error: {e}"

    def _get_semantic_interpretation(self, sql_result: dict) -> str:
        """
        Looks at the SQL result and tries to match it to a Semantic Layer metric
        to get the business interpretation (e.g., "Critical — escalate to leadership").
        """
        interpretations = []

        # Try to extract churn rate from result
        if sql_result.get("result_df") is not None:
            df = sql_result["result_df"]
            if "churn_rate_pct" in df.columns and not df.empty:
                val = float(df["churn_rate_pct"].iloc[0])
                interpretations.append(f"Churn Rate ({val}%): {interpret('churn_rate', val)}")

            if "mrr" in df.columns and not df.empty:
                val = float(df["mrr"].iloc[0])
                interpretations.append(f"MRR (${val:,.0f}): {interpret('mrr', val)}")

        if not interpretations:
            # Generic interpretation based on SQL source
            source = sql_result.get("source", "unknown")
            interpretations.append(
                f"SQL source: {source}. "
                "Consult metrics_defs.py for full threshold definitions."
            )

        return " | ".join(interpretations)

    def run(
        self,
        question:    str,
        sql_result:  dict,
        rag_result:  dict,
        customer_df: pd.DataFrame | None = None,
    ) -> dict:
        """
        Main entry point. Synthesises all agent outputs into a strategy.

        Args:
            question:    The original business question
            sql_result:  Output from SQLAgent.run()
            rag_result:  Output from RAGAgent.query()
            customer_df: Optional customer DataFrame for live churn scoring

        Returns:
            dict with recommendation, priority, actions, evidence
        """
        logger.info("StrategistAgent synthesising strategy for: '{}'", question)

        # Prepare all context pieces
        sql_summary  = sql_result.get("summary", "No SQL summary available.")
        sql_data_str = (
            sql_result["result_df"].head(5).to_string(index=False)
            if sql_result.get("result_df") is not None and not sql_result["result_df"].empty
            else "No data returned."
        )
        rag_insight  = rag_result.get("insight", "No RAG insight available.")
        num_tickets  = rag_result.get("num_retrieved", 0)
        model_preds  = self._get_model_predictions(customer_df)
        semantic_int = self._get_semantic_interpretation(sql_result)

        # Call Llama 3 to synthesise
        chain    = STRATEGY_PROMPT | self.llm
        response = chain.invoke({
            "question":               question,
            "sql_summary":            sql_summary,
            "sql_data":               sql_data_str,
            "rag_insight":            rag_insight,
            "num_tickets":            num_tickets,
            "model_predictions":      model_preds,
            "semantic_interpretation": semantic_int,
        })

        recommendation = response.content.strip()

        # Extract priority from the structured response
        priority = "MEDIUM"
        for p in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            if p in recommendation.upper()[:100]:
                priority = p
                break

        # Extract action items (lines starting with 1., 2., 3.)
        actions = []
        for line in recommendation.split("\n"):
            stripped = line.strip()
            if len(stripped) > 3 and stripped[0].isdigit() and stripped[1] == ".":
                actions.append(stripped[3:].strip())

        result = {
            "question":       question,
            "recommendation": recommendation,
            "priority":       priority,
            "actions":        actions,
            "evidence": {
                "sql_source":    sql_result.get("source"),
                "sql_attempts":  sql_result.get("attempts"),
                "rag_tickets":   num_tickets,
                "model_used":    self.churn_model is not None,
            },
        }

        logger.success(
            "Strategy generated. Priority: {} | Actions: {}",
            priority, len(actions)
        )
        return result


# =============================================================================
# MAIN — Full pipeline test: SQL + RAG + Strategist together
# =============================================================================

if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule

    console = Console(highlight=False)
    console.print(Panel.fit(
        "[bold cyan]NEXUS-ABI[/bold cyan] | [white]Strategist Agent Test[/white]\n"
        "[dim]Running full SQL + RAG + Strategy pipeline...[/dim]",
        border_style="cyan"
    ))

    # Import other agents
    from agents.sql_agent import SQLAgent
    from agents.rag_agent import RAGAgent

    sql_agent   = SQLAgent()
    rag_agent   = RAGAgent()
    strategist  = StrategistAgent()

    # Ensure RAG vector store is populated
    rag_agent.ingest(force_reload=False)

    # --- Test question ---
    question = "Our churn is high. What is the root cause and what should we do?"

    console.print(Rule(f"[bold yellow]{question}[/bold yellow]"))

    # Step 1: SQL
    console.print("\n[dim]Step 1: SQL Agent querying...[/dim]")
    sql_result = sql_agent.run("What is our current churn rate?")
    console.print(f"[cyan]SQL:[/cyan] {sql_result['summary'][:120]}...")

    # Step 2: RAG
    console.print("[dim]Step 2: RAG Agent searching support logs...[/dim]")
    rag_result = rag_agent.query("Why are customers threatening to cancel or expressing dissatisfaction?")
    console.print(f"[cyan]RAG:[/cyan] {rag_result['insight'][:120]}...")

    # Step 3: Strategist
    console.print("[dim]Step 3: Strategist synthesising...[/dim]\n")
    strategy = strategist.run(
        question   = question,
        sql_result = sql_result,
        rag_result = rag_result,
    )

    console.print(Panel(
        strategy["recommendation"],
        title=f"[bold red]PRIORITY: {strategy['priority']}[/bold red]",
        border_style="red" if strategy["priority"] == "CRITICAL" else "yellow",
    ))

    if strategy["actions"]:
        console.print("\n[bold green]Extracted Action Items:[/bold green]")
        for i, action in enumerate(strategy["actions"], 1):
            console.print(f"  {i}. {action}")

    console.print(f"\n[dim]Evidence: {strategy['evidence']}[/dim]\n")
