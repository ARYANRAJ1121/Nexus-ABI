"""
=============================================================================
NEXUS-ABI | Layer 4: Agentic Engine → Utils
File: db_config.py
=============================================================================

PURPOSE:
  Manages the database connection and data loading for the SQL Agent.
  Loads the customer and transaction CSVs into a local SQLite database
  so the SQL Agent can run real SQL queries against them.

WHY SQLITE LOCALLY (NOT POSTGRESQL)?
  PostgreSQL requires a running server, credentials, and network setup.
  SQLite is a single file — zero infrastructure, same SQL-92 syntax.
  The SQL Agent writes standard SELECT queries that work on both.

  In production deployment:
    Change ENGINE_URL below from sqlite:/// to postgresql://...
    Nothing else changes. That's the value of SQLAlchemy abstraction.

WHAT GETS LOADED INTO THE DB:
  Table: customers        ← from 01_data_pipeline/raw/customers.csv
  Table: transactions     ← from 01_data_pipeline/raw/transactions.csv
  Table: support_logs     ← from 01_data_pipeline/raw/support_logs.csv

WHAT THIS FILE PROVIDES:
  get_engine()            → SQLAlchemy engine (importable by SQL Agent)
  load_data_to_db()       → One-time CSV → SQLite loader
  run_query(sql)          → Execute a SQL string and return a DataFrame
  get_schema_info()       → Returns table schemas for the SQL Agent's prompt
=============================================================================
"""

import sys
import io
from pathlib import Path

import pandas as pd
from loguru import logger
from sqlalchemy import create_engine, text, inspect

import os
os.environ["PYTHONUTF8"] = "1"

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
DB_PATH    = Path("04_agentic_engine/nexus_abi.db")    # SQLite file location
ENGINE_URL = f"sqlite:///{DB_PATH}"                     # SQLAlchemy connection string

# Data source paths
RAW_DIR  = Path("01_data_pipeline/raw")
TABLES   = {
    "customers":    RAW_DIR / "customers.csv",
    "transactions": RAW_DIR / "transactions.csv",
    "support_logs": RAW_DIR / "support_logs.csv",
}


# ---------------------------------------------------------------------------
# ENGINE FACTORY
# ---------------------------------------------------------------------------

def get_engine():
    """
    Returns a SQLAlchemy engine connected to the local SQLite database.

    Called by:
      - SQL Agent  → to run SELECT queries
      - db loader  → to write CSVs into tables

    The engine is lazy — it doesn't open a connection until a query runs.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(ENGINE_URL, echo=False)
    return engine


# ---------------------------------------------------------------------------
# DATA LOADER
# ---------------------------------------------------------------------------

def load_data_to_db(force_reload: bool = False) -> None:
    """
    Loads all raw CSVs into the SQLite database as tables.

    Args:
        force_reload: If True, drops and recreates all tables.
                      If False, skips tables that already exist.

    This is called ONCE during setup (or when data changes).
    Agents don't call this — they call run_query() directly.
    """
    engine = get_engine()

    # Check which tables already exist
    inspector      = inspect(engine)
    existing_tables = inspector.get_table_names()

    for table_name, csv_path in TABLES.items():

        if table_name in existing_tables and not force_reload:
            logger.info("Table '{}' already exists. Skipping. (use force_reload=True to refresh)", table_name)
            continue

        if not csv_path.exists():
            logger.warning("CSV not found: {}. Run synthetic_gen.py first.", csv_path)
            continue

        logger.info("Loading {} → table '{}'...", csv_path.name, table_name)
        df = pd.read_csv(csv_path)

        # Write to SQLite
        # if_exists="replace" drops and recreates the table — safe for local dev
        df.to_sql(
            name       = table_name,
            con        = engine,
            if_exists  = "replace",
            index      = False,
            chunksize  = 5000,   # Write in chunks to avoid memory spikes
        )

        # Verify
        with engine.connect() as conn:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}")).scalar()

        logger.success("Loaded {:,} rows into table '{}'", count, table_name)

    logger.success("Database ready at: {}", DB_PATH.resolve())


# ---------------------------------------------------------------------------
# QUERY RUNNER — What the SQL Agent calls
# ---------------------------------------------------------------------------

def run_query(sql: str) -> pd.DataFrame:
    """
    Executes a SQL string against the SQLite database and returns a DataFrame.

    The SQL Agent generates a query, passes it here, and gets back a DataFrame.
    If the query fails, the error is returned as a string so LangGraph can
    send it back to the agent for self-correction.

    Args:
        sql: A SQL SELECT string. May be generated by the LLM.

    Returns:
        pd.DataFrame on success, or raises Exception on failure.
        The calling agent (main_graph.py) catches the exception for retry.
    """
    engine = get_engine()

    logger.info("Running SQL:\n{}", sql.strip())

    try:
        with engine.connect() as conn:
            df = pd.read_sql_query(text(sql), conn)
        logger.success("Query returned {:,} rows", len(df))
        return df

    except Exception as e:
        logger.error("SQL execution failed: {}", e)
        raise  # Let the LangGraph error handler catch this for retry


# ---------------------------------------------------------------------------
# SCHEMA INFO — Injected into the SQL Agent's system prompt
# ---------------------------------------------------------------------------

def get_schema_info() -> str:
    """
    Returns a formatted string describing all table schemas.

    WHY THIS MATTERS:
      The SQL Agent needs to know what columns exist before it can write SQL.
      We inject this into its system prompt so it never guesses column names.
      Guessing column names = hallucinated SQL = runtime errors.

    Example output:
      TABLE: customers
        customer_id       TEXT
        company_name      TEXT
        plan_type         TEXT
        monthly_spend     REAL
        churned           INTEGER
        ...
    """
    engine = get_engine()

    # If DB doesn't exist yet, generate schema from CSV headers instead
    if not DB_PATH.exists():
        schema_lines = []
        for table_name, csv_path in TABLES.items():
            if csv_path.exists():
                df = pd.read_csv(csv_path, nrows=0)  # Headers only
                schema_lines.append(f"TABLE: {table_name}")
                for col in df.columns:
                    schema_lines.append(f"  {col}")
                schema_lines.append("")
        return "\n".join(schema_lines)

    inspector = inspect(engine)
    schema_lines = []

    for table_name in inspector.get_table_names():
        schema_lines.append(f"TABLE: {table_name}")
        columns = inspector.get_columns(table_name)
        for col in columns:
            schema_lines.append(f"  {col['name']:<30} {str(col['type'])}")
        schema_lines.append("")  # blank line between tables

    return "\n".join(schema_lines)


# =============================================================================
# MAIN — Run this file to initialise and verify the database
# =============================================================================

if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table as RichTable

    console = Console()
    console.print(Panel.fit(
        "[bold cyan]NEXUS-ABI[/bold cyan] | [white]Database Initialiser[/white]\n"
        "[dim]Loading CSVs into local SQLite database...[/dim]",
        border_style="cyan"
    ))

    # --- Load all CSVs into DB ---
    load_data_to_db(force_reload=True)

    # --- Show schema ---
    console.print("\n[bold yellow]Database Schema:[/bold yellow]")
    schema = get_schema_info()
    console.print(f"[dim]{schema}[/dim]")

    # --- Run a test query ---
    console.print("[bold yellow]Test Query — Top 5 churned customers by monthly spend:[/bold yellow]")
    try:
        result_df = run_query("""
            SELECT
                company_name,
                plan_type,
                monthly_spend,
                support_tickets_count,
                last_login_days_ago
            FROM customers
            WHERE churned = 1
            ORDER BY monthly_spend DESC
            LIMIT 5
        """)
        console.print(result_df.to_string(index=False))
    except Exception as e:
        console.print(f"[red]Query failed: {e}[/red]")

    # --- Show table row counts ---
    console.print("\n[bold yellow]Table Row Counts:[/bold yellow]")
    engine = get_engine()
    count_table = RichTable(show_header=True, header_style="bold magenta")
    count_table.add_column("Table",     style="cyan", width=20)
    count_table.add_column("Rows",      justify="right", width=10)

    with engine.connect() as conn:
        for tbl in ["customers", "transactions", "support_logs"]:
            try:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar()
                count_table.add_row(tbl, f"{count:,}")
            except Exception:
                count_table.add_row(tbl, "[red]not loaded[/red]")

    console.print(count_table)
    console.print(f"\n[bold green]✓ Database ready at: {DB_PATH.resolve()}[/bold green]\n")
