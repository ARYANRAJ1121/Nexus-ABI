"""
=============================================================================
NEXUS-ABI | Layer 1: Data Engine
File: spark_cleaner.py
=============================================================================

PURPOSE:
  Reads the raw CSVs produced by synthetic_gen.py and applies a series of
  data quality checks, corrections, and standardisations using PySpark.
  The cleaned output is saved as Parquet — a compressed, columnar format
  that is ~10x faster to read than CSV for ML workloads.

WHY PYSPARK AND NOT PANDAS?
  Pandas is fine for < 1 million rows. The moment NexaCorp grows — say,
  100 million transactions — pandas would crash your machine.
  PySpark distributes the work across all CPU cores automatically.
  The EXACT same code runs locally and on a cloud cluster. That is the
  enterprise-grade differentiator.

WHAT GETS CLEANED:
  [customers.csv]
    ✓ Remove duplicate customer_id rows
    ✓ Drop rows with null customer_id or email
    ✓ Flag and cap negative monthly_spend values
    ✓ Ensure tenure_months is positive
    ✓ Standardise plan_type casing
    ✓ Cast churned column to integer (0 or 1)

  [transactions.csv]
    ✓ Remove duplicate transaction_id rows
    ✓ Drop transactions with null customer_id
    ✓ Remove transactions with amount <= 0 (data entry errors)
    ✓ Flag future-dated transactions (date > today) as anomalies
    ✓ Standardise status column values

  [support_logs.csv]
    ✓ Remove rows with empty text (useless for RAG)
    ✓ Trim whitespace from all text fields
    ✓ Standardise sentiment column to title case

OUTPUT:
  01_data_pipeline/cleaned/
    ├── customers.parquet
    ├── transactions.parquet
    └── support_logs.parquet

RUN:
  python 01_data_pipeline/spark_cleaner.py
=============================================================================
"""

import os
from pathlib import Path
from datetime import date

from loguru import logger
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# ---------------------------------------------------------------------------
# PySpark imports
# ---------------------------------------------------------------------------
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, DoubleType, DateType

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------
RAW_DIR     = Path("01_data_pipeline/raw")
CLEANED_DIR = Path("01_data_pipeline/cleaned")
CLEANED_DIR.mkdir(parents=True, exist_ok=True)

console = Console()

# ---------------------------------------------------------------------------
# SPARK SESSION
# ---------------------------------------------------------------------------
def create_spark_session() -> SparkSession:
    """
    Creates a local PySpark session.

    master("local[*]") means: use ALL available CPU cores on this machine.
    In production, you'd change this to a cluster URL like:
      "spark://your-cluster-master:7077"
    The rest of the code stays identical — that's the power of Spark.
    """
    logger.info("Initialising Spark session (local mode, all cores)...")

    spark = (
        SparkSession.builder
        .appName("Nexus-ABI Data Cleaner")
        .master("local[*]")
        # Suppress verbose Spark INFO logs — we use loguru for our own logs
        .config("spark.sql.shuffle.partitions", "4")    # Right-size for local dev
        .config("spark.driver.memory", "2g")            # 2 GB for the driver
        .config("spark.ui.showConsoleProgress", "false") # No progress bars in Spark
        .getOrCreate()
    )

    # Suppress Spark's own verbose logging
    spark.sparkContext.setLogLevel("ERROR")
    logger.success("Spark session ready. Version: {}", spark.version)
    return spark


# =============================================================================
# CLEANING FUNCTIONS — One per dataset
# =============================================================================

def clean_customers(spark: SparkSession) -> tuple[DataFrame, dict]:
    """
    Cleans the customers table and returns the cleaned DataFrame
    plus a report dictionary of all anomalies found.

    The 'report' dict is what we'll display in the terminal summary.
    """
    logger.info("Loading customers.csv...")
    raw = spark.read.csv(
        str(RAW_DIR / "customers.csv"),
        header=True,
        inferSchema=True
    )

    original_count = raw.count()
    report = {"original": original_count}

    # ---- CHECK 1: Duplicate customer_id ----
    # A duplicate PK means the same customer was inserted twice — breaks JOINs
    raw = raw.dropDuplicates(["customer_id"])
    report["duplicates_removed"] = original_count - raw.count()

    # ---- CHECK 2: Null critical fields ----
    # A customer without an ID or email is unusable — drop them
    before = raw.count()
    raw = raw.dropna(subset=["customer_id", "email"])
    report["nulls_removed"] = before - raw.count()

    # ---- CHECK 3: Negative monthly_spend ----
    # Can't have a customer paying negative money. Cap at 0.
    negative_spend = raw.filter(F.col("monthly_spend") < 0).count()
    raw = raw.withColumn(
        "monthly_spend",
        F.when(F.col("monthly_spend") < 0, F.lit(0.0)).otherwise(F.col("monthly_spend"))
    )
    report["negative_spend_fixed"] = negative_spend

    # ---- CHECK 4: Invalid tenure ----
    # tenure_months < 1 is nonsensical — a customer must exist for at least 1 month
    raw = raw.withColumn(
        "tenure_months",
        F.when(F.col("tenure_months") < 1, F.lit(1)).otherwise(F.col("tenure_months"))
    )

    # ---- CHECK 5: Standardise plan_type casing ----
    # "starter", "Starter", "STARTER" all mean the same thing — normalise to title case
    raw = raw.withColumn("plan_type", F.initcap(F.col("plan_type")))

    # ---- CHECK 6: Ensure churned is 0 or 1 only ----
    raw = raw.withColumn("churned", F.col("churned").cast(IntegerType()))
    raw = raw.withColumn(
        "churned",
        F.when(F.col("churned").isin([0, 1]), F.col("churned")).otherwise(F.lit(0))
    )

    # ---- CHECK 7: Add a data quality timestamp ----
    raw = raw.withColumn("cleaned_at", F.current_timestamp())

    report["final_count"] = raw.count()
    return raw, report


def clean_transactions(spark: SparkSession) -> tuple[DataFrame, dict]:
    """
    Cleans the transactions table.

    Key rule: a transaction with amount <= 0 or a future date is
    almost certainly a data entry error or a test record. We remove
    them before they distort the revenue analysis.
    """
    logger.info("Loading transactions.csv...")
    raw = spark.read.csv(
        str(RAW_DIR / "transactions.csv"),
        header=True,
        inferSchema=True
    )

    original_count = raw.count()
    report = {"original": original_count}

    # ---- CHECK 1: Duplicate transaction_id ----
    raw = raw.dropDuplicates(["transaction_id"])
    report["duplicates_removed"] = original_count - raw.count()

    # ---- CHECK 2: Missing customer_id (orphaned transactions) ----
    before = raw.count()
    raw = raw.dropna(subset=["customer_id"])
    report["orphaned_removed"] = before - raw.count()

    # ---- CHECK 3: Zero or negative amounts ----
    # These are invalid. A refund should have status="Refunded", not amount < 0.
    before = raw.count()
    raw = raw.filter(F.col("amount") > 0)
    report["invalid_amount_removed"] = before - raw.count()

    # ---- CHECK 4: Future-dated transactions ----
    # A transaction_date in the future = someone entered wrong year. Flag it.
    today = str(date.today())
    future_txns = raw.filter(F.col("date") > F.lit(today))
    report["future_dates_flagged"] = future_txns.count()
    # We keep them but add a flag column instead of deleting
    raw = raw.withColumn(
        "is_anomaly",
        F.when(F.col("date") > F.lit(today), F.lit(True)).otherwise(F.lit(False))
    )

    # ---- CHECK 5: Standardise status column ----
    valid_statuses = ["Completed", "Refunded", "Disputed"]
    raw = raw.withColumn(
        "status",
        F.when(F.col("status").isin(valid_statuses), F.col("status"))
         .otherwise(F.lit("Unknown"))
    )

    raw = raw.withColumn("cleaned_at", F.current_timestamp())
    report["final_count"] = raw.count()
    return raw, report


def clean_support_logs(spark: SparkSession) -> tuple[DataFrame, dict]:
    """
    Cleans support logs — the unstructured text that feeds the RAG agent.

    Key rule: an empty text column means ChromaDB has nothing to embed.
    Those rows are useless for the RAG pipeline and must be removed.
    """
    logger.info("Loading support_logs.csv...")
    raw = spark.read.csv(
        str(RAW_DIR / "support_logs.csv"),
        header=True,
        inferSchema=True
    )

    original_count = raw.count()
    report = {"original": original_count}

    # ---- CHECK 1: Remove rows with empty text ----
    before = raw.count()
    raw = raw.filter(
        F.col("text").isNotNull() & (F.trim(F.col("text")) != "")
    )
    report["empty_text_removed"] = before - raw.count()

    # ---- CHECK 2: Trim all text fields ----
    # Trailing spaces in issue_type break GROUP BY queries downstream
    for col_name in ["issue_type", "sentiment", "text"]:
        raw = raw.withColumn(col_name, F.trim(F.col(col_name)))

    # ---- CHECK 3: Standardise sentiment to title case ----
    raw = raw.withColumn("sentiment", F.initcap(F.col("sentiment")))

    # ---- CHECK 4: Add word count column ----
    # Useful for RAG — very short tickets may not have enough context
    raw = raw.withColumn(
        "word_count",
        F.size(F.split(F.col("text"), " "))
    )

    raw = raw.withColumn("cleaned_at", F.current_timestamp())
    report["final_count"] = raw.count()
    return raw, report


# =============================================================================
# SAVE TO PARQUET
# =============================================================================

def save_as_parquet(df: DataFrame, name: str) -> None:
    """
    Saves a Spark DataFrame as Parquet.

    WHY PARQUET OVER CSV?
    ┌─────────────────┬────────────────┬──────────────────┐
    │  Property       │  CSV           │  Parquet         │
    ├─────────────────┼────────────────┼──────────────────┤
    │  Format         │  Text (rows)   │  Binary (cols)   │
    │  Read speed     │  Slow          │  ~10x faster     │
    │  File size      │  Large         │  ~5x smaller     │
    │  Schema         │  Must infer    │  Embedded        │
    │  Partitioning   │  Not supported │  Native support  │
    └─────────────────┴────────────────┴──────────────────┘

    coalesce(1) merges all Spark partitions into a single file.
    In production you'd remove this for parallel writes.
    """
    output_path = str(CLEANED_DIR / f"{name}.parquet")
    logger.info("Writing {} → {}", name, output_path)

    df.coalesce(1).write.mode("overwrite").parquet(output_path)
    logger.success("Saved: {}", output_path)


# =============================================================================
# REPORTING
# =============================================================================

def print_report(name: str, report: dict) -> None:
    """Prints a clean anomaly report table to the terminal using Rich."""
    table = Table(title=f"[bold cyan]{name}[/bold cyan] — Cleaning Report",
                  show_header=True, header_style="bold magenta")
    table.add_column("Check", style="dim", width=35)
    table.add_column("Count", justify="right")

    table.add_row("Original rows",          str(report.get("original", "—")))
    table.add_row("Duplicates removed",     str(report.get("duplicates_removed", 0)))
    table.add_row("Null rows removed",      str(report.get("nulls_removed", 0)))
    table.add_row("Orphaned rows removed",  str(report.get("orphaned_removed", 0)))
    table.add_row("Negative spend fixed",   str(report.get("negative_spend_fixed", 0)))
    table.add_row("Invalid amounts removed",str(report.get("invalid_amount_removed", 0)))
    table.add_row("Future dates flagged",   str(report.get("future_dates_flagged", 0)))
    table.add_row("Empty text removed",     str(report.get("empty_text_removed", 0)))
    table.add_row("[bold green]Final clean rows[/bold green]",
                  f"[bold green]{report.get('final_count', '—')}[/bold green]")

    console.print(table)
    console.print()


# =============================================================================
# MAIN RUNNER
# =============================================================================

def main():
    console.print(Panel.fit(
        "[bold cyan]NEXUS-ABI[/bold cyan] | [white]PySpark Data Cleaner[/white]\n"
        "[dim]Cleaning raw CSVs → validated Parquet files...[/dim]",
        border_style="cyan"
    ))

    spark = create_spark_session()

    try:
        # --- Clean Customers ---
        customers_clean, customers_report = clean_customers(spark)
        save_as_parquet(customers_clean, "customers")
        print_report("customers.csv", customers_report)

        # --- Clean Transactions ---
        transactions_clean, txn_report = clean_transactions(spark)
        save_as_parquet(transactions_clean, "transactions")
        print_report("transactions.csv", txn_report)

        # --- Clean Support Logs ---
        logs_clean, logs_report = clean_support_logs(spark)
        save_as_parquet(logs_clean, "support_logs")
        print_report("support_logs.csv", logs_report)

        console.print("[bold green]✓ All datasets cleaned and saved as Parquet.[/bold green]")
        console.print(f"[dim]Output: {CLEANED_DIR.resolve()}[/dim]\n")

    finally:
        spark.stop()
        logger.info("Spark session stopped cleanly.")


if __name__ == "__main__":
    main()
