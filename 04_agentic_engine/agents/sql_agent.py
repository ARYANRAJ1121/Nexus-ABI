"""
=============================================================================
NEXUS-ABI | Layer 4: Agentic Engine → Agents
File: sql_agent.py
=============================================================================

PURPOSE:
  The SQL Expert Agent. Translates natural language business questions into
  SQL queries, executes them against the SQLite database, and returns
  structured results with a plain-English summary.

THE SELF-CORRECTION LOOP (Why this is "elite"):
  If the generated SQL fails, the agent doesn't crash. It:
    1. Catches the SQL error
    2. Sends the original question + broken SQL + error message back to Llama 3
    3. Llama 3 writes a corrected SQL
    4. Retries (up to MAX_RETRIES times)
  This is the same pattern used by production Text-to-SQL systems at Meta,
  Uber, and LinkedIn.

GOVERNANCE (Anti-Hallucination):
  Before generating SQL from scratch, the agent checks the Semantic Layer
  (metrics_defs.py). If the question maps to a known KPI like "churn rate",
  it uses the pre-defined SQL directly — the LLM never touches the formula.
  Only for questions outside the Semantic Layer does it generate SQL freely.

INPUT:  A natural language question (str)
OUTPUT: dict with keys:
          question     → original question
          sql          → the SQL that was executed
          result_df    → pandas DataFrame of results
          summary      → plain-English description of results
          source       → "semantic_layer" | "llm_generated"
          attempts     → how many tries it took
=============================================================================
"""

import sys
import io
import os
from pathlib import Path

# Path setup — allows running this file directly OR importing it
PROJECT_ROOT  = Path(__file__).parent.parent.parent
LAYER_4_ROOT  = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LAYER_4_ROOT))

os.environ["PYTHONUTF8"] = "1"

import pandas as pd
from loguru import logger
from langchain_core.prompts import ChatPromptTemplate

from utils.llm_config import get_llm
from utils.db_config import run_query, get_schema_info

# Semantic Layer import
sys.path.insert(0, str(PROJECT_ROOT / "03_semantic_layer"))
from metrics_defs import METRICS

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
MAX_RETRIES = 3   # Max SQL self-correction attempts before giving up

# ---------------------------------------------------------------------------
# KEYWORD → METRIC MAPPING
# The agent checks if the user's question mentions known KPI keywords.
# If yes, it uses the pre-defined SQL from the Semantic Layer (governance).
# ---------------------------------------------------------------------------
METRIC_KEYWORDS: dict[str, list[str]] = {
    "churn_rate":           ["churn rate", "churn %", "churning percentage"],
    "mrr":                  ["mrr", "monthly recurring revenue", "recurring revenue"],
    "revenue_at_risk":      ["revenue at risk", "risk revenue", "lost revenue"],
    "arpu":                 ["arpu", "average revenue per user", "revenue per customer"],
    "avg_clv":              ["clv", "lifetime value", "customer lifetime value"],
    "support_ticket_rate":  ["ticket rate", "support tickets", "tickets per customer"],
    "enterprise_churn_rate":["enterprise churn", "enterprise retention"],
    "avg_tenure_churned":   ["tenure churned", "how long before churn", "churn tenure"],
    "inactive_rate":        ["inactive", "not logged in", "disengaged customers"],
    "plan_distribution":    ["plan distribution", "plan breakdown", "plan mix"],
}


def _match_metric(question: str) -> str | None:
    """
    Checks if the question maps to a known Semantic Layer metric.
    Returns metric name if found, else None.
    """
    q_lower = question.lower()
    for metric_name, keywords in METRIC_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            logger.info("Question matched Semantic Layer metric: '{}'", metric_name)
            return metric_name
    return None


# ---------------------------------------------------------------------------
# PROMPT TEMPLATES
# ---------------------------------------------------------------------------

SQL_GENERATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert SQL analyst for a B2B SaaS company called NexaCorp.
Your job is to write a single, correct SQLite SQL query to answer the user's question.

DATABASE SCHEMA:
{schema}

RULES:
- Write ONLY the SQL query. No explanation. No markdown. No backticks.
- Use only columns that exist in the schema above.
- SQLite does not support FULL OUTER JOIN or RIGHT JOIN. Use LEFT JOIN or subqueries.
- Always use LIMIT 20 for queries that could return many rows, unless a specific number is asked.
- For percentage calculations, use: ROUND(100.0 * numerator / NULLIF(denominator, 0), 2)
- churned = 1 means the customer has left. churned = 0 means active.
- Return useful column aliases (e.g., AS churn_rate_pct, AS total_revenue).
"""),
    ("human", "Question: {question}\n\nSQL Query:"),
])

SQL_CORRECTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert SQL analyst. A previous SQL query failed.
Fix it so it works in SQLite.

DATABASE SCHEMA:
{schema}

RULES:
- Write ONLY the corrected SQL. No explanation. No markdown. No backticks.
- Analyse the error message carefully to fix the exact issue.
- Ensure all column names exist in the schema.
"""),
    ("human", """Original question: {question}

Broken SQL:
{broken_sql}

Error message:
{error}

Corrected SQL:"""),
])

SUMMARY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a concise business analyst. Summarise SQL query results in 2-3 sentences.
Focus on the business implication, not the technical details.
No bullet points. Write flowing prose. Be specific about numbers."""),
    ("human", """Question: {question}

SQL Results:
{results}

Business Summary (2-3 sentences):"""),
])


# =============================================================================
# SQL AGENT CLASS
# =============================================================================

class SQLAgent:
    """
    The SQL Expert Agent.

    Usage:
        agent = SQLAgent()
        result = agent.run("What is our current churn rate by plan type?")
        print(result["summary"])
        print(result["result_df"])
    """

    def __init__(self):
        self.llm    = get_llm(temperature=0.0)   # Temperature 0 = deterministic SQL
        self.schema = get_schema_info()
        logger.info("SQLAgent initialised with schema:\n{}", self.schema[:300] + "...")

    def run(self, question: str) -> dict:
        """
        Main entry point. Takes a question, returns a structured result dict.
        """
        logger.info("SQLAgent received question: '{}'", question)

        result = {
            "question": question,
            "sql":       None,
            "result_df": None,
            "summary":   None,
            "source":    None,
            "attempts":  0,
            "error":     None,
        }

        # ---- STEP 1: Check Semantic Layer first ----
        metric_name = _match_metric(question)
        if metric_name and metric_name in METRICS:
            metric = METRICS[metric_name]
            sql    = metric.sql.strip()
            result["source"] = "semantic_layer"
            result["sql"]    = sql
            logger.info("Using governed SQL from Semantic Layer for metric: {}", metric_name)
        else:
            # ---- STEP 2: Generate SQL with LLM ----
            result["source"] = "llm_generated"
            sql = self._generate_sql(question)
            result["sql"] = sql

        # ---- STEP 3: Execute SQL with self-correction loop ----
        for attempt in range(1, MAX_RETRIES + 1):
            result["attempts"] = attempt
            try:
                df = run_query(sql)
                result["result_df"] = df
                logger.success("SQL executed successfully on attempt {}", attempt)
                break

            except Exception as e:
                error_msg = str(e)
                logger.warning("Attempt {} failed: {}", attempt, error_msg)
                result["error"] = error_msg

                if attempt < MAX_RETRIES:
                    logger.info("Self-correcting SQL... (attempt {}/{})", attempt + 1, MAX_RETRIES)
                    sql = self._correct_sql(question, sql, error_msg)
                    result["sql"] = sql
                else:
                    logger.error("All {} attempts failed. Returning error state.", MAX_RETRIES)
                    result["summary"] = f"Could not answer after {MAX_RETRIES} attempts. Last error: {error_msg}"
                    return result

        # ---- STEP 4: Generate plain-English summary ----
        result["summary"] = self._summarise(question, result["result_df"])
        return result

    def _generate_sql(self, question: str) -> str:
        """Asks Llama 3 to write SQL for the question."""
        chain    = SQL_GENERATION_PROMPT | self.llm
        response = chain.invoke({"question": question, "schema": self.schema})
        sql      = response.content.strip()
        # Clean up in case the LLM wraps in markdown despite instructions
        sql = sql.replace("```sql", "").replace("```", "").strip()
        logger.info("Generated SQL:\n{}", sql)
        return sql

    def _correct_sql(self, question: str, broken_sql: str, error: str) -> str:
        """Asks Llama 3 to fix a broken SQL query."""
        chain    = SQL_CORRECTION_PROMPT | self.llm
        response = chain.invoke({
            "question":   question,
            "broken_sql": broken_sql,
            "error":      error,
            "schema":     self.schema,
        })
        corrected = response.content.strip()
        corrected = corrected.replace("```sql", "").replace("```", "").strip()
        logger.info("Corrected SQL:\n{}", corrected)
        return corrected

    def _summarise(self, question: str, df: pd.DataFrame) -> str:
        """Generates a plain-English business summary of the SQL results."""
        if df is None or df.empty:
            return "The query returned no results."

        # Truncate large result tables to avoid overflowing context window
        results_str = df.head(10).to_string(index=False)

        chain    = SUMMARY_PROMPT | self.llm
        response = chain.invoke({"question": question, "results": results_str})
        return response.content.strip()


# =============================================================================
# MAIN — Run this file directly to test the SQL Agent
# =============================================================================

if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule

    console = Console(highlight=False)
    console.print(Panel.fit(
        "[bold cyan]NEXUS-ABI[/bold cyan] | [white]SQL Agent Test[/white]\n"
        "[dim]Testing natural language to SQL pipeline...[/dim]",
        border_style="cyan"
    ))

    agent = SQLAgent()

    test_questions = [
        "What is our current churn rate?",                                    # → Semantic Layer
        "Which industry has the highest average monthly spend?",              # → LLM Generated
        "Show me the top 5 customers by CLV who are at risk of churning.",    # → LLM Generated
    ]

    for q in test_questions:
        console.print(Rule(f"[bold yellow]{q}[/bold yellow]"))
        result = agent.run(q)

        console.print(f"[dim]Source: {result['source']} | Attempts: {result['attempts']}[/dim]")
        console.print(f"[cyan]SQL:[/cyan]\n[dim]{result['sql']}[/dim]\n")

        if result["result_df"] is not None and not result["result_df"].empty:
            console.print("[cyan]Results:[/cyan]")
            console.print(result["result_df"].head(5).to_string(index=False))

        console.print(f"\n[bold green]Summary:[/bold green] {result['summary']}\n")
