"""
=============================================================================
NEXUS-ABI | Layer 3: Semantic Layer
File: metrics_defs.py
=============================================================================

PURPOSE:
  Single source of truth for every KPI in the NexaCorp business.
  Every metric is defined exactly once — as a Python function AND as a SQL
  string — and can be called by any agent in Layer 4.

THE PROBLEM THIS SOLVES (Metric Drift / Hallucination):
  Without this file, an LLM might define "Churn Rate" differently on each
  query. One response uses total customers as denominator. The next uses
  active customers. A third uses customers-at-start-of-month.

  Result: your board gets three different "Churn Rate" numbers from the same
  system in the same week. That is a career-ending data governance failure.

  This file makes that impossible. The formula is code. Code is law.

HOW AGENTS USE THIS:
  - SQL Agent  → uses metric.sql to generate the final SELECT statement
  - Strategist → uses metric.interpretation to frame the answer in context
  - Evaluator  → uses metric.compute() to verify the SQL output is correct

METRICS DEFINED:
  1.  churn_rate              → % customers who left this period
  2.  monthly_recurring_revenue (MRR) → predictable monthly revenue
  3.  net_revenue_retention   (NRR)   → expansion vs contraction revenue
  4.  customer_lifetime_value (CLV)   → expected total revenue per customer
  5.  arpu                    → average revenue per user
  6.  revenue_at_risk         → MRR from customers likely to churn
  7.  support_ticket_rate     → avg tickets per customer per month
  8.  plan_distribution       → breakdown of customers by plan type
  9.  avg_tenure_churned       → how long churned customers stayed
  10. high_value_churn_rate   → churn rate among Enterprise customers only

=============================================================================
"""

from dataclasses import dataclass, field
from typing import Callable, Optional
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
import sys, io

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

console = Console()


# =============================================================================
# METRIC DEFINITION SCHEMA
# =============================================================================

@dataclass
class MetricDefinition:
    """
    A single, canonical definition of a business KPI.

    Every agent that needs this metric gets it from here.
    No improvisation. No guessing. No drift.

    Attributes:
        name         : Machine-readable identifier (snake_case)
        display_name : Human-readable label for the dashboard
        description  : What this metric measures and why it matters
        unit         : The unit of measurement (%, $, count, ratio)
        sql          : The SQL query that computes this metric from PostgreSQL.
                       Uses {table} placeholder so agents can inject table names.
        compute      : Python function that calculates the metric from a DataFrame.
                       Agents use this for in-memory validation.
        interpretation: Dict of thresholds and what they mean for the business.
                        The Strategist Agent uses this to frame its recommendations.
        owner        : Team responsible for this metric in a real organisation.
    """
    name:           str
    display_name:   str
    description:    str
    unit:           str
    sql:            str
    compute:        Callable[[pd.DataFrame], float]
    interpretation: dict = field(default_factory=dict)
    owner:          str  = "Analytics"
    higher_is_better: bool = True


# =============================================================================
# METRIC DEFINITIONS — The Governance Layer
# =============================================================================

METRICS: dict[str, MetricDefinition] = {}

def register(metric: MetricDefinition) -> MetricDefinition:
    """Registers a metric into the global METRICS registry."""
    METRICS[metric.name] = metric
    return metric


# -----------------------------------------------------------------------------
# 1. CHURN RATE
# -----------------------------------------------------------------------------
METRICS["churn_rate"] = MetricDefinition(
    name         = "churn_rate",
    display_name = "Monthly Churn Rate",
    description  = (
        "The percentage of active customers at the start of the period "
        "who cancelled their subscription during that period. "
        "Formula: churned_customers / total_customers * 100. "
        "Industry benchmark for B2B SaaS: < 5% monthly is healthy."
    ),
    unit         = "%",
    sql          = """
        SELECT
            ROUND(
                100.0 * SUM(CASE WHEN churned = 1 THEN 1 ELSE 0 END)
                / NULLIF(COUNT(*), 0),
                2
            ) AS churn_rate_pct
        FROM customers
    """,
    compute      = lambda df: round(
        df["churned"].sum() / len(df) * 100, 2
    ),
    interpretation = {
        "healthy":  {"range": [0, 5],     "message": "Healthy. Customer retention is strong."},
        "warning":  {"range": [5, 10],    "message": "Warning. Investigate top churn reasons immediately."},
        "critical": {"range": [10, 100],  "message": "Critical. Retention is failing. Escalate to leadership."},
    },
    higher_is_better = False,
    owner        = "Customer Success",
)


# -----------------------------------------------------------------------------
# 2. MONTHLY RECURRING REVENUE (MRR)
# -----------------------------------------------------------------------------
METRICS["mrr"] = MetricDefinition(
    name         = "mrr",
    display_name = "Monthly Recurring Revenue (MRR)",
    description  = (
        "Total predictable monthly subscription revenue from active customers. "
        "Formula: SUM(monthly_spend) WHERE churned = 0. "
        "This is the single most important metric for a SaaS business — "
        "it represents committed, repeatable revenue."
    ),
    unit         = "$",
    sql          = """
        SELECT
            ROUND(SUM(monthly_spend), 2) AS mrr
        FROM customers
        WHERE churned = 0
    """,
    compute      = lambda df: round(
        df.loc[df["churned"] == 0, "monthly_spend"].sum(), 2
    ),
    interpretation = {
        "growing":  {"message": "MRR is growing. Expansion revenue exceeding churn."},
        "flat":     {"message": "MRR is flat. New revenue is offsetting churned revenue exactly."},
        "shrinking":{"message": "MRR is shrinking. Churned revenue exceeds new and expansion revenue."},
    },
    higher_is_better = True,
    owner        = "Finance",
)


# -----------------------------------------------------------------------------
# 3. REVENUE AT RISK
# -----------------------------------------------------------------------------
METRICS["revenue_at_risk"] = MetricDefinition(
    name         = "revenue_at_risk",
    display_name = "Revenue At Risk",
    description  = (
        "The total monthly spend from customers who have churned. "
        "This is the MRR that has already been lost. "
        "Formula: SUM(monthly_spend) WHERE churned = 1. "
        "Used to prioritise retention — the agents will flag the top "
        "revenue-at-risk customers for immediate outreach."
    ),
    unit         = "$",
    sql          = """
        SELECT
            ROUND(SUM(monthly_spend), 2) AS revenue_at_risk,
            COUNT(*)                      AS churned_count
        FROM customers
        WHERE churned = 1
    """,
    compute      = lambda df: round(
        df.loc[df["churned"] == 1, "monthly_spend"].sum(), 2
    ),
    interpretation = {
        "low":      {"threshold": 10000,  "message": "Revenue at risk is manageable. Focus on prevention."},
        "medium":   {"threshold": 50000,  "message": "Significant revenue exposure. Launch retention campaign."},
        "high":     {"threshold": 999999, "message": "Critical revenue loss. Emergency escalation required."},
    },
    higher_is_better = False,
    owner        = "Revenue Operations",
)


# -----------------------------------------------------------------------------
# 4. AVERAGE REVENUE PER USER (ARPU)
# -----------------------------------------------------------------------------
METRICS["arpu"] = MetricDefinition(
    name         = "arpu",
    display_name = "Average Revenue Per User (ARPU)",
    description  = (
        "Average monthly spend per active customer. "
        "Formula: SUM(monthly_spend) / COUNT(active_customers). "
        "A rising ARPU means customers are upgrading or expanding. "
        "A falling ARPU means downgrades or plan mix shift toward cheaper tiers."
    ),
    unit         = "$",
    sql          = """
        SELECT
            ROUND(AVG(monthly_spend), 2) AS arpu
        FROM customers
        WHERE churned = 0
    """,
    compute      = lambda df: round(
        df.loc[df["churned"] == 0, "monthly_spend"].mean(), 2
    ),
    interpretation = {
        "strong":  {"message": "ARPU indicates healthy plan mix with enterprise tiers dominant."},
        "average": {"message": "ARPU is in the mid-range. Consider upsell motions."},
        "weak":    {"message": "ARPU is low. Product is commoditised or customers are under-utilising."},
    },
    higher_is_better = True,
    owner        = "Product",
)


# -----------------------------------------------------------------------------
# 5. CUSTOMER LIFETIME VALUE (CLV)
# -----------------------------------------------------------------------------
METRICS["avg_clv"] = MetricDefinition(
    name         = "avg_clv",
    display_name = "Average Customer Lifetime Value",
    description  = (
        "The average total revenue expected from a customer over their lifetime. "
        "Formula: AVG(clv) across all customers. "
        "CLV is pre-computed by the XGBoost model in Layer 2. "
        "The SQL Agent never recalculates this — it reads the model output. "
        "This is where the Predictive Core connects to the Semantic Layer."
    ),
    unit         = "$",
    sql          = """
        SELECT
            ROUND(AVG(clv), 2)  AS avg_clv,
            ROUND(MIN(clv), 2)  AS min_clv,
            ROUND(MAX(clv), 2)  AS max_clv
        FROM customers
    """,
    compute      = lambda df: round(df["clv"].mean(), 2),
    interpretation = {
        "note": "CLV values come from the XGBoost model trained in Layer 2. "
                "Do not recompute using a manual formula — use the model output."
    },
    higher_is_better = True,
    owner        = "Data Science",
)


# -----------------------------------------------------------------------------
# 6. SUPPORT TICKET RATE
# -----------------------------------------------------------------------------
METRICS["support_ticket_rate"] = MetricDefinition(
    name         = "support_ticket_rate",
    display_name = "Avg Support Tickets Per Customer",
    description  = (
        "Average number of support tickets per customer. "
        "Formula: AVG(support_tickets_count). "
        "A high ticket rate signals product quality issues or poor onboarding. "
        "The RAG Agent searches support_logs for the root cause context."
    ),
    unit         = "tickets",
    sql          = """
        SELECT
            ROUND(AVG(support_tickets_count), 2)                    AS avg_tickets,
            ROUND(AVG(CASE WHEN churned = 1 THEN support_tickets_count END), 2) AS avg_tickets_churned,
            ROUND(AVG(CASE WHEN churned = 0 THEN support_tickets_count END), 2) AS avg_tickets_retained
        FROM customers
    """,
    compute      = lambda df: round(df["support_tickets_count"].mean(), 2),
    interpretation = {
        "healthy":  {"range": [0, 2],   "message": "Low ticket volume. Product is self-serve and reliable."},
        "elevated": {"range": [2, 5],   "message": "Elevated tickets. Review top issue types with RAG Agent."},
        "critical": {"range": [5, 999], "message": "High ticket burden. Likely causing customer frustration and churn."},
    },
    higher_is_better = False,
    owner        = "Support",
)


# -----------------------------------------------------------------------------
# 7. HIGH-VALUE CUSTOMER CHURN RATE
# -----------------------------------------------------------------------------
METRICS["enterprise_churn_rate"] = MetricDefinition(
    name         = "enterprise_churn_rate",
    display_name = "Enterprise Customer Churn Rate",
    description  = (
        "Churn rate calculated only among Enterprise-plan customers. "
        "Formula: churned_enterprise / total_enterprise * 100. "
        "This matters more than overall churn — one Enterprise customer "
        "losing is worth more than 20 Starter customers churning."
    ),
    unit         = "%",
    sql          = """
        SELECT
            ROUND(
                100.0 * SUM(CASE WHEN churned = 1 THEN 1 ELSE 0 END)
                / NULLIF(COUNT(*), 0),
                2
            ) AS enterprise_churn_rate_pct
        FROM customers
        WHERE plan_type = 'Enterprise'
    """,
    compute      = lambda df: round(
        df.loc[df["plan_type"] == "Enterprise", "churned"].mean() * 100, 2
    ),
    interpretation = {
        "healthy":  {"range": [0, 3],   "message": "Enterprise retention is excellent."},
        "warning":  {"range": [3, 7],   "message": "Enterprise churn is elevated. Assign dedicated CSM."},
        "critical": {"range": [7, 100], "message": "Critical. Enterprise churn threatens company survival."},
    },
    higher_is_better = False,
    owner        = "Enterprise Sales",
)


# -----------------------------------------------------------------------------
# 8. PLAN DISTRIBUTION
# -----------------------------------------------------------------------------
METRICS["plan_distribution"] = MetricDefinition(
    name         = "plan_distribution",
    display_name = "Customer Plan Distribution",
    description  = (
        "Breakdown of active customers by subscription plan type. "
        "Formula: COUNT(*) GROUP BY plan_type WHERE churned = 0. "
        "A healthy SaaS product should see customers moving UP the plan "
        "tiers over time (Starter -> Growth -> Enterprise). "
        "Legacy customers should be actively migrated."
    ),
    unit         = "count",
    sql          = """
        SELECT
            plan_type,
            COUNT(*)                              AS customer_count,
            ROUND(SUM(monthly_spend), 2)          AS plan_mrr,
            ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct_of_total
        FROM customers
        WHERE churned = 0
        GROUP BY plan_type
        ORDER BY plan_mrr DESC
    """,
    compute      = lambda df: (
        df.loc[df["churned"] == 0]
          .groupby("plan_type")
          .size()
          .to_dict()
    ),
    interpretation = {
        "note": "High Legacy % = retention risk. High Enterprise % = healthy. "
                "Compare plan_mrr to customer_count to find disproportionate value tiers."
    },
    higher_is_better = True,
    owner        = "Product Marketing",
)


# -----------------------------------------------------------------------------
# 9. AVERAGE TENURE OF CHURNED CUSTOMERS
# -----------------------------------------------------------------------------
METRICS["avg_tenure_churned"] = MetricDefinition(
    name         = "avg_tenure_churned",
    display_name = "Avg Tenure of Churned Customers (months)",
    description  = (
        "The average number of months a customer stayed before churning. "
        "Formula: AVG(tenure_months) WHERE churned = 1. "
        "If this is < 6 months: your onboarding is failing. "
        "If this is > 18 months: long-tenured customers are leaving — "
        "suggests a product quality or pricing issue, not onboarding."
    ),
    unit         = "months",
    sql          = """
        SELECT
            ROUND(AVG(tenure_months), 1) AS avg_tenure_churned_months
        FROM customers
        WHERE churned = 1
    """,
    compute      = lambda df: round(
        df.loc[df["churned"] == 1, "tenure_months"].mean(), 1
    ),
    interpretation = {
        "onboarding_failure": {
            "range": [0, 6],
            "message": "Customers churning early. Root cause: failed onboarding or value gap."
        },
        "healthy_midpoint": {
            "range": [6, 18],
            "message": "Mid-tenure churn. Investigate competitive displacement."
        },
        "long_term_issue": {
            "range": [18, 999],
            "message": "Long-tenured customers leaving. Likely pricing or product erosion."
        },
    },
    higher_is_better = True,
    owner        = "Customer Success",
)


# -----------------------------------------------------------------------------
# 10. INACTIVE CUSTOMER RATE
# -----------------------------------------------------------------------------
METRICS["inactive_rate"] = MetricDefinition(
    name         = "inactive_rate",
    display_name = "Inactive Customer Rate (30+ days)",
    description  = (
        "Percentage of active customers who have not logged in for 30+ days. "
        "Formula: COUNT(last_login_days_ago > 30) / total_active * 100. "
        "This is a leading indicator of churn — inactivity precedes cancellation. "
        "The XGBoost model uses this as one of its top features."
    ),
    unit         = "%",
    sql          = """
        SELECT
            ROUND(
                100.0 * SUM(CASE WHEN last_login_days_ago > 30 THEN 1 ELSE 0 END)
                / NULLIF(COUNT(*), 0),
                2
            ) AS inactive_rate_pct
        FROM customers
        WHERE churned = 0
    """,
    compute      = lambda df: round(
        (df.loc[df["churned"] == 0, "last_login_days_ago"] > 30).mean() * 100, 2
    ),
    interpretation = {
        "healthy":  {"range": [0, 10],  "message": "Low inactivity. Engagement programmes are working."},
        "warning":  {"range": [10, 25], "message": "Rising inactivity. Launch re-engagement campaign."},
        "critical": {"range": [25, 100],"message": "High inactivity. Significant churn wave incoming."},
    },
    higher_is_better = False,
    owner        = "Customer Success",
)


# =============================================================================
# PUBLIC API — What agents call
# =============================================================================

def get_metric(name: str) -> MetricDefinition:
    """
    Returns a MetricDefinition by name.
    Raises a clear error if the metric doesn't exist — no silent failures.
    """
    if name not in METRICS:
        available = list(METRICS.keys())
        raise KeyError(
            f"Metric '{name}' is not defined in the Semantic Layer.\n"
            f"Available metrics: {available}\n"
            f"To add a new metric, define it in 03_semantic_layer/metrics_defs.py."
        )
    return METRICS[name]


def compute_all(df: pd.DataFrame) -> dict[str, float]:
    """
    Runs all scalar metrics against a customer DataFrame.
    Returns a flat dict of {metric_name: value}.
    Skips non-scalar metrics (like plan_distribution).
    """
    results = {}
    scalar_metrics = [
        "churn_rate", "mrr", "revenue_at_risk", "arpu",
        "avg_clv", "support_ticket_rate", "enterprise_churn_rate",
        "avg_tenure_churned", "inactive_rate",
    ]
    for name in scalar_metrics:
        try:
            results[name] = METRICS[name].compute(df)
        except Exception as e:
            results[name] = None
            logger.warning("Could not compute {}: {}", name, e)
    return results


def interpret(metric_name: str, value: float) -> str:
    """
    Returns the business interpretation string for a given metric value.
    Used by the Strategist Agent to frame its answer.

    Example:
      interpret("churn_rate", 12.5)
      → "Critical. Retention is failing. Escalate to leadership."
    """
    metric = get_metric(metric_name)
    interp = metric.interpretation

    for level, config in interp.items():
        if isinstance(config, dict) and "range" in config:
            lo, hi = config["range"]
            if lo <= value < hi:
                return config["message"]

    # Fallback if no range matched
    return interp.get("note", f"Value: {value} {metric.unit}")


# =============================================================================
# MAIN — Demonstrate the Semantic Layer
# =============================================================================

def main():
    from pathlib import Path

    console.print(Panel.fit(
        "[bold cyan]NEXUS-ABI[/bold cyan] | [white]Semantic Layer Audit[/white]\n"
        "[dim]Computing all KPIs from the canonical definitions...[/dim]",
        border_style="cyan"
    ))

    # Load the customer data
    data_path = Path("01_data_pipeline/raw/customers.csv")
    if not data_path.exists():
        console.print("[red]Run synthetic_gen.py first.[/red]")
        return

    df = pd.read_csv(data_path)

    # Compute all metrics
    results = compute_all(df)

    # Display results table
    table = Table(
        title="[bold cyan]NexaCorp Business KPI Dashboard[/bold cyan]",
        show_header=True, header_style="bold magenta"
    )
    table.add_column("Metric",       style="dim",  width=30)
    table.add_column("Value",        justify="right", width=15)
    table.add_column("Unit",         width=8)
    table.add_column("Interpretation", width=50)

    for name, value in results.items():
        metric = METRICS[name]
        if value is None:
            continue
        interp_text = interpret(name, value) if metric.interpretation else "—"
        table.add_row(
            metric.display_name,
            str(value),
            metric.unit,
            interp_text,
        )

    console.print(table)

    # Show available SQL for one metric as demonstration
    console.print("\n[bold yellow]Example: SQL for 'churn_rate' (what the SQL Agent runs):[/bold yellow]")
    console.print(f"[dim]{METRICS['churn_rate'].sql.strip()}[/dim]")

    console.print(
        f"\n[bold green]Semantic Layer loaded: {len(METRICS)} metrics registered.[/bold green]"
    )
    console.print("[dim]Agents will import from this file. No metric can be hallucinated.[/dim]\n")


if __name__ == "__main__":
    main()
