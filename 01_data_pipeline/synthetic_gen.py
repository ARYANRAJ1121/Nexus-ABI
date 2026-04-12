"""
=============================================================================
NEXUS-ABI | Layer 1: Data Engine
File: synthetic_gen.py
=============================================================================

PURPOSE:
  Generates a realistic synthetic dataset for a fictional B2B SaaS company
  called "NexaCorp". The data mimics 3 years of business operations and is
  designed to have *real, learnable patterns* for the XGBoost churn model.

WHY SYNTHETIC DATA?
  - No privacy risk (no real customer PII)
  - We control the signal: we PLANT churn reasons so the ML model trains well
  - Can scale to any size (10K, 1M rows) by changing NUM_CUSTOMERS

OUTPUTS (saved to 01_data_pipeline/raw/):
  - customers.csv    → Customer profiles + churn label (target variable)
  - transactions.csv → Purchase history per customer
  - support_logs.csv → Free-text support tickets (used by RAG agent later)

RUN:
  python 01_data_pipeline/synthetic_gen.py
=============================================================================
"""

import random
import uuid
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker
from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

# --------------------------------------------------------------------------
# CONFIGURATION — Change these to scale the dataset up or down
# --------------------------------------------------------------------------
NUM_CUSTOMERS    = 10_000   # Total unique B2B customers
NUM_TRANSACTIONS = 50_000   # Total transaction records
NUM_SUPPORT_LOGS = 2_000    # Total support tickets (for RAG)
RANDOM_SEED      = 42       # Fixed seed so results are reproducible

OUTPUT_DIR = Path("01_data_pipeline/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# SETUP
# --------------------------------------------------------------------------
fake = Faker("en_US")
Faker.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

console = Console()

# --------------------------------------------------------------------------
# BUSINESS CONSTANTS — These mirror a real SaaS pricing model
# --------------------------------------------------------------------------
PLAN_TYPES = {
    # plan_name : (monthly_price_min, monthly_price_max, weight)
    "Starter":    (99,   499,   0.40),   # 40% of customers on cheapest plan
    "Growth":     (500,  1999,  0.35),   # 35% mid-tier
    "Enterprise": (2000, 9999,  0.20),   # 20% high-value
    "Legacy":     (50,   150,   0.05),   # 5%  old plan — highest churn risk
}

INDUSTRIES = [
    "FinTech", "HealthTech", "E-Commerce", "Logistics",
    "Manufacturing", "Education", "Retail", "Real Estate",
    "Media & Entertainment", "Cybersecurity",
]

REGIONS = ["North America", "Europe", "Asia-Pacific", "Latin America", "Middle East"]

ISSUE_TYPES = [
    "Billing dispute",
    "Performance degradation",
    "Feature request not delivered",
    "API integration failure",
    "Account access issue",
    "Data export problem",
    "Onboarding support",
    "Contract renewal query",
    "Security concern",
    "SLA breach complaint",
]

# --------------------------------------------------------------------------
# SUPPORT LOG TEMPLATES
# These give the RAG agent rich, searchable context.
# Each template represents a real business scenario.
# --------------------------------------------------------------------------
SUPPORT_TEMPLATES = [
    "Customer {name} from {industry} reported that their {feature} has been unresponsive for {days} days. They are threatening to cancel unless resolved by {date}.",
    "Ticket raised by {name}: Our team cannot export data from the dashboard. This is blocking our monthly board report. Plan: {plan}.",
    "High-priority complaint from {name} ({industry}): We experienced a 4-hour outage during peak hours on {date}. This directly impacted our revenue. Requesting SLA credit.",
    "{name} called in to dispute an invoice of ${amount}. They believe they were charged for seats that were never provisioned. Account is on {plan} plan.",
    "Customer {name} has submitted their 3rd ticket this month about slow query performance. Their usage has been growing rapidly. May be a scaling issue.",
    "Churn risk flagged for {name}: They have not logged in for {days} days and their last 2 support tickets were marked unresolved.",
    "{name} from {industry} is requesting a feature that was promised in Q{quarter}. Non-delivery is causing frustration. Risk of downgrade.",
    "Positive feedback from {name}: The new analytics module has significantly improved their weekly reporting. They are considering upgrading to Enterprise.",
    "Account {name} is expanding their team and needs 15 new user licenses. Please prioritize provisioning to avoid delay in their onboarding.",
    "{name} is requesting an urgent call with the solution architect. They are evaluating a competitor ({competitor}) and need a strong retention pitch.",
]

COMPETITORS = ["Salesforce", "HubSpot", "Tableau", "Looker", "Domo", "Qlik", "SAP Analytics"]
FEATURES = ["reporting dashboard", "API gateway", "data export module", "user management panel", "billing portal"]


# =============================================================================
# GENERATOR FUNCTIONS
# =============================================================================

def generate_customers() -> pd.DataFrame:
    """
    Generates the master customer table.

    THE CHURN LOGIC (This is why this file is not just random noise):
    We simulate REAL reasons companies churn:
      1. High support ticket count + low login frequency → likely leaving
      2. Legacy plan customers → low satisfaction, high price sensitivity
      3. Short tenure (<6 months) → haven't seen value yet → higher risk
      4. Low monthly spend relative to plan → underutilizing → might cancel

    This planted logic means XGBoost will find REAL signal in the data.
    """
    logger.info("Generating {} customer profiles...", NUM_CUSTOMERS)
    records = []

    plan_names    = list(PLAN_TYPES.keys())
    plan_weights  = [v[2] for v in PLAN_TYPES.values()]

    for _ in range(NUM_CUSTOMERS):
        customer_id   = str(uuid.uuid4())
        signup_date   = fake.date_between(start_date="-3y", end_date="-1m")
        tenure_months = max(1, (datetime.today().date() - signup_date).days // 30)

        plan          = random.choices(plan_names, weights=plan_weights, k=1)[0]
        price_min, price_max, _ = PLAN_TYPES[plan]
        monthly_spend = round(random.uniform(price_min, price_max), 2)

        support_tickets_count = max(0, int(np.random.exponential(scale=2.5)))
        last_login_days_ago   = max(0, int(np.random.exponential(scale=20)))
        num_users             = random.randint(1, 200)
        industry              = random.choice(INDUSTRIES)
        region                = random.choice(REGIONS)

        # ---- CHURN PROBABILITY ENGINE ----
        # We assign a churn probability based on business signals.
        # This is the "ground truth" that the ML model will learn to replicate.
        churn_prob = 0.10  # base 10% churn rate (realistic for SaaS)

        if plan == "Legacy":
            churn_prob += 0.25       # Legacy customers are unhappy
        if tenure_months < 6:
            churn_prob += 0.15       # New customers haven't found value yet
        if support_tickets_count > 5:
            churn_prob += 0.20       # Too many problems = frustration
        if last_login_days_ago > 30:
            churn_prob += 0.20       # Not using = not renewing
        if monthly_spend < price_min * 1.1:
            churn_prob += 0.10       # Underutilizing their plan

        churn_prob = min(churn_prob, 0.95)  # Cap at 95% (nobody is certain)
        churned    = int(random.random() < churn_prob)

        # ---- CLV CALCULATION ----
        # Customer Lifetime Value = Monthly Spend × Expected Remaining Months
        # Expected months = inverse of churn probability (simple estimate)
        expected_months = max(1, int(1 / max(churn_prob, 0.05)))
        clv = round(monthly_spend * expected_months, 2)

        records.append({
            "customer_id":           customer_id,
            "company_name":          fake.company(),
            "contact_name":          fake.name(),
            "email":                 fake.company_email(),
            "industry":              industry,
            "region":                region,
            "plan_type":             plan,
            "monthly_spend":         monthly_spend,
            "num_users":             num_users,
            "signup_date":           signup_date,
            "tenure_months":         tenure_months,
            "support_tickets_count": support_tickets_count,
            "last_login_days_ago":   last_login_days_ago,
            "clv":                   clv,
            "churned":               churned,   # ← TARGET VARIABLE for XGBoost
        })

    df = pd.DataFrame(records)
    logger.success("Customers generated. Churn rate: {:.1f}%",
                   df["churned"].mean() * 100)
    return df


def generate_transactions(customers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates transaction history for each customer.

    WHY THIS MATTERS:
    Transactions reveal SPENDING PATTERNS — a customer who suddenly reduces
    their spend month-over-month is a churn signal. The SQL Agent will query
    this table to find revenue trends.
    """
    logger.info("Generating {} transactions...", NUM_TRANSACTIONS)
    records = []

    customer_ids = customers_df["customer_id"].tolist()
    categories   = ["Software License", "Add-on Module", "Professional Services",
                     "Training", "Support Tier Upgrade", "API Overage"]

    for _ in range(NUM_TRANSACTIONS):
        customer_id  = random.choice(customer_ids)
        txn_date     = fake.date_between(start_date="-3y", end_date="today")
        amount       = round(random.uniform(50, 15000), 2)
        category     = random.choice(categories)
        status       = random.choices(
                           ["Completed", "Refunded", "Disputed"],
                           weights=[0.88, 0.08, 0.04]
                       )[0]

        records.append({
            "transaction_id": str(uuid.uuid4()),
            "customer_id":    customer_id,
            "date":           txn_date,
            "amount":         amount,
            "category":       category,
            "status":         status,
            "payment_method": random.choice(["Credit Card", "Wire Transfer", "ACH", "Invoice"]),
        })

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    logger.success("Transactions generated. Total revenue: ${:,.2f}",
                   df[df["status"] == "Completed"]["amount"].sum())
    return df


def generate_support_logs(customers_df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates unstructured support ticket text.

    WHY THIS MATTERS FOR RAG:
    The RAG (Retrieval-Augmented Generation) agent will embed these logs
    into ChromaDB as vectors. When someone asks "Why are FinTech customers
    churning?", the RAG agent will SEARCH these logs and find real evidence,
    instead of hallucinating an answer.

    The templates above produce varied, realistic B2B support language.
    """
    logger.info("Generating {} support logs...", NUM_SUPPORT_LOGS)
    records = []

    for i in range(NUM_SUPPORT_LOGS):
        customer_row  = customers_df.sample(1).iloc[0]
        template      = random.choice(SUPPORT_TEMPLATES)
        issue_type    = random.choice(ISSUE_TYPES)
        log_date      = fake.date_between(start_date="-2y", end_date="today")

        # Fill in template placeholders
        text = template.format(
            name       = customer_row["contact_name"],
            industry   = customer_row["industry"],
            plan       = customer_row["plan_type"],
            feature    = random.choice(FEATURES),
            days       = random.randint(1, 45),
            date       = fake.date_between(start_date="-30d", end_date="+30d"),
            amount     = f"{random.randint(500, 25000):,}",
            quarter    = random.randint(1, 4),
            competitor = random.choice(COMPETITORS),
        )

        # Sentiment — churned customers have more negative tickets
        is_churned = customer_row["churned"]
        sentiment  = random.choices(
            ["Negative", "Neutral", "Positive"],
            weights=[0.6 if is_churned else 0.2,
                     0.3,
                     0.1 if is_churned else 0.5]
        )[0]

        records.append({
            "log_id":      f"LOG-{i+1:05d}",
            "customer_id": customer_row["customer_id"],
            "date":        log_date,
            "issue_type":  issue_type,
            "sentiment":   sentiment,
            "text":        text,         # ← This raw text goes into ChromaDB
        })

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    logger.success("Support logs generated. Negative sentiment: {:.1f}%",
                   (df["sentiment"] == "Negative").mean() * 100)
    return df


# =============================================================================
# MAIN RUNNER
# =============================================================================

def main():
    console.print(Panel.fit(
        "[bold cyan]NEXUS-ABI[/bold cyan] | [white]Synthetic Data Generator[/white]\n"
        "[dim]Generating NexaCorp's 3-year business dataset...[/dim]",
        border_style="cyan"
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        transient=True,
    ) as progress:

        # Step 1: Customers
        task = progress.add_task("[cyan]Building customer profiles...", total=3)
        customers_df = generate_customers()
        customers_df.to_csv(OUTPUT_DIR / "customers.csv", index=False)
        logger.info("Saved → {}", OUTPUT_DIR / "customers.csv")
        progress.advance(task)

        # Step 2: Transactions
        progress.update(task, description="[yellow]Generating transactions...")
        transactions_df = generate_transactions(customers_df)
        transactions_df.to_csv(OUTPUT_DIR / "transactions.csv", index=False)
        logger.info("Saved → {}", OUTPUT_DIR / "transactions.csv")
        progress.advance(task)

        # Step 3: Support Logs
        progress.update(task, description="[magenta]Writing support logs...")
        support_df = generate_support_logs(customers_df)
        support_df.to_csv(OUTPUT_DIR / "support_logs.csv", index=False)
        logger.info("Saved → {}", OUTPUT_DIR / "support_logs.csv")
        progress.advance(task)

    # Final summary
    console.print("\n[bold green]✓ Dataset generation complete![/bold green]")
    console.print(f"  [cyan]customers.csv[/cyan]    → {len(customers_df):,} rows")
    console.print(f"  [cyan]transactions.csv[/cyan] → {len(transactions_df):,} rows")
    console.print(f"  [cyan]support_logs.csv[/cyan] → {len(support_df):,} rows")
    console.print(f"\n  [dim]Output directory: {OUTPUT_DIR.resolve()}[/dim]")
    console.print(f"  [dim]Churn rate in dataset: {customers_df['churned'].mean()*100:.1f}%[/dim]")
    console.print(f"  [dim]Total synthetic revenue: ${transactions_df[transactions_df['status']=='Completed']['amount'].sum():,.0f}[/dim]")


if __name__ == "__main__":
    main()
