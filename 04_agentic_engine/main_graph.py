"""
=============================================================================
NEXUS-ABI | Layer 4: Agentic Engine
File: main_graph.py
=============================================================================

PURPOSE:
  Wires the three agents (SQL, RAG, Strategist) into a single LangGraph
  stateful workflow. This is the "conductor" of the multi-agent system.

WHY LANGGRAPH AND NOT JUST CALLING AGENTS IN SEQUENCE?
  You could do: sql_result = sql.run(); rag_result = rag.run(); strat.run()
  That works. But it breaks as soon as anything goes wrong.

  LangGraph gives you:
    1. STATE — every agent reads from and writes to a shared state dict.
               You always know exactly what each agent saw and produced.
    2. CONDITIONAL ROUTING — if SQL fails, route back for retry.
                             If it succeeds, move on automatically.
    3. CHECKPOINTING — the graph can be paused and resumed.
                       If your laptop crashes mid-run, it picks up where it left.
    4. OBSERVABILITY — every node run is logged with its input/output.
                       In production, this feeds into monitoring dashboards.

THE GRAPH FLOW:
  ┌─────────┐
  │  START  │
  └────┬────┘
       │
       ▼
  ┌──────────┐   SQL fails (< 3 retries)    ┌─────────────┐
  │ SQL NODE ├──────────────────────────────▶│ SQL RETRY * │
  └────┬─────┘                               └──────┬──────┘
       │ SQL succeeds                               │ (loops back)
       ▼                                            │
  ┌──────────┐ ◀──────────────────────────────────-┘
  │ RAG NODE │
  └────┬─────┘
       │
       ▼
  ┌──────────────────┐
  │ STRATEGIST NODE  │
  └────────┬─────────┘
           │
           ▼
        ┌─────┐
        │ END │
        └─────┘

  * Retry is handled inside SQLAgent's own self-correction loop.
    LangGraph's role is higher-level: if ALL retries are exhausted,
    LangGraph routes to a graceful error state instead of crashing.

STATE:
  Every node shares a single NexusState dict (TypedDict):
    question     → the user's original question (set at START, never changes)
    sql_result   → populated by sql_node
    rag_result   → populated by rag_node
    strategy     → populated by strategist_node
    error        → set if any node fails
    retry_count  → tracks how many SQL retries have happened

RUN:
  python 04_agentic_engine/main_graph.py
  OR import and call:
    from main_graph import run_nexus
    result = run_nexus("Why is our churn rate high?")
=============================================================================
"""

import os
import sys
from pathlib import Path
from typing import TypedDict, Optional

os.environ["PYTHONUTF8"] = "1"

PROJECT_ROOT = Path(__file__).parent.parent
LAYER_4_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LAYER_4_ROOT))

from loguru import logger

# ---------------------------------------------------------------------------
# IMPORTS — Agents and LangGraph
# ---------------------------------------------------------------------------
from agents.sql_agent        import SQLAgent
from agents.rag_agent        import RAGAgent
from agents.strategist_agent import StrategistAgent

from langgraph.graph import StateGraph, END

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
MAX_SQL_RETRIES = 3   # How many times LangGraph will re-route to SQL node


# =============================================================================
# STEP 1: DEFINE THE STATE
# This is the "memory" shared between all nodes.
# TypedDict enforces what keys the state can have — no sloppy dicts.
# =============================================================================

class NexusState(TypedDict):
    """
    The shared state that flows through the entire graph.

    Every node receives this dict and returns updated values.
    LangGraph merges the returned dict into the state automatically.
    """
    question:    str              # Original user question — never changes
    sql_result:  Optional[dict]   # Output from SQLAgent.run()
    rag_result:  Optional[dict]   # Output from RAGAgent.query()
    strategy:    Optional[dict]   # Output from StrategistAgent.run()
    error:       Optional[str]    # Error message if something went wrong
    retry_count: int              # Tracks SQL retry attempts at graph level


# =============================================================================
# STEP 2: INITIALISE AGENTS
# Agents are created once and reused across all graph runs.
# Creating them once is important because:
#   - Loading XGBoost models takes ~1 second
#   - Loading ChromaDB collection takes ~2 seconds
#   - Subsequent runs are nearly instant
# =============================================================================

logger.info("Initialising agents (one-time setup)...")
_sql_agent        = SQLAgent()
_rag_agent        = RAGAgent()
_strategist_agent = StrategistAgent()

# Ensure RAG vector store is populated
logger.info("Checking RAG vectorstore...")
_rag_agent.ingest(force_reload=False)


# =============================================================================
# STEP 3: DEFINE NODES
# Each node is a Python function that:
#   - Receives the current state (dict)
#   - Does its work
#   - Returns a dict of ONLY the keys it changed
# LangGraph merges the returned dict into the full state.
# =============================================================================

def sql_node(state: NexusState) -> dict:
    """
    Node 1: SQL Agent
    Translates the question into SQL, runs it, returns results.
    """
    logger.info("[LangGraph] Entering SQL Node. Retry count: {}", state["retry_count"])

    try:
        result = _sql_agent.run(state["question"])

        if result.get("result_df") is None or result.get("error"):
            # SQL failed all internal retries
            return {
                "sql_result":  result,
                "error":       result.get("error", "SQL returned no data"),
                "retry_count": state["retry_count"] + 1,
            }

        logger.success("[LangGraph] SQL Node succeeded. Source: {}", result.get("source"))
        return {
            "sql_result":  result,
            "error":       None,
            "retry_count": state["retry_count"],
        }

    except Exception as e:
        logger.error("[LangGraph] SQL Node crashed: {}", e)
        return {
            "sql_result":  None,
            "error":       str(e),
            "retry_count": state["retry_count"] + 1,
        }


def rag_node(state: NexusState) -> dict:
    """
    Node 2: RAG Agent
    Searches support tickets for evidence related to the question.
    """
    logger.info("[LangGraph] Entering RAG Node.")

    try:
        result = _rag_agent.query(state["question"])
        logger.success("[LangGraph] RAG Node succeeded. Tickets retrieved: {}",
                       result.get("num_retrieved", 0))
        return {"rag_result": result, "error": None}

    except Exception as e:
        logger.error("[LangGraph] RAG Node crashed: {}", e)
        # RAG failure is non-fatal — Strategist can still work with SQL alone
        return {
            "rag_result": {
                "question":      state["question"],
                "retrieved":     [],
                "insight":       f"RAG unavailable: {e}",
                "num_retrieved": 0,
            },
            "error": None,   # Don't propagate — graceful degradation
        }


def strategist_node(state: NexusState) -> dict:
    """
    Node 3: Strategist Agent
    Synthesises SQL + RAG + XGBoost into a business recommendation.
    """
    logger.info("[LangGraph] Entering Strategist Node.")

    try:
        strategy = _strategist_agent.run(
            question   = state["question"],
            sql_result = state["sql_result"] or {},
            rag_result = state["rag_result"] or {},
        )
        logger.success("[LangGraph] Strategy generated. Priority: {}",
                       strategy.get("priority"))
        return {"strategy": strategy, "error": None}

    except Exception as e:
        logger.error("[LangGraph] Strategist Node crashed: {}", e)
        return {
            "strategy": {
                "recommendation": f"Strategy generation failed: {e}",
                "priority":       "UNKNOWN",
                "actions":        [],
                "evidence":       {},
            },
            "error": str(e),
        }


def error_node(state: NexusState) -> dict:
    """
    Fallback Node: Reached when SQL exhausts all retries.
    Returns a graceful error message instead of crashing.
    """
    logger.error("[LangGraph] Error Node reached. All SQL retries exhausted.")
    return {
        "strategy": {
            "recommendation": (
                f"Unable to answer '{state['question']}' after "
                f"{state['retry_count']} attempts.\n"
                f"Last error: {state.get('error', 'Unknown')}\n"
                "Please rephrase the question or check the database connection."
            ),
            "priority": "UNKNOWN",
            "actions":  ["Check database connection", "Rephrase the question"],
            "evidence": {},
        }
    }


# =============================================================================
# STEP 4: ROUTING FUNCTIONS (Conditional Edges)
# These functions look at the current state and decide which node to go to next.
# =============================================================================

def route_after_sql(state: NexusState) -> str:
    """
    After SQL Node: decide where to go next.

    Returns:
        "rag"   → SQL succeeded, continue to RAG
        "sql"   → SQL failed but retries remain, go back to SQL
        "error" → SQL failed and all retries exhausted, go to error node
    """
    has_error   = bool(state.get("error"))
    retry_count = state.get("retry_count", 0)

    if not has_error:
        logger.info("[LangGraph] Routing: SQL → RAG")
        return "rag"
    elif retry_count < MAX_SQL_RETRIES:
        logger.warning("[LangGraph] Routing: SQL failed, retrying ({}/{})",
                       retry_count, MAX_SQL_RETRIES)
        return "sql"    # Loop back for retry
    else:
        logger.error("[LangGraph] Routing: SQL → Error (max retries reached)")
        return "error"


# =============================================================================
# STEP 5: BUILD THE GRAPH
# =============================================================================

def build_graph() -> StateGraph:
    """
    Constructs and compiles the LangGraph workflow.

    Returns a compiled graph that can be invoked with:
        graph.invoke({"question": "...", ...})
    """
    # Create the graph with our state schema
    graph = StateGraph(NexusState)

    # --- Add all nodes ---
    graph.add_node("sql",        sql_node)
    graph.add_node("rag",        rag_node)
    graph.add_node("strategist", strategist_node)
    graph.add_node("error",      error_node)

    # --- Set the entry point ---
    graph.set_entry_point("sql")

    # --- Add edges ---

    # After SQL: conditional routing (succeed → rag, fail → retry/error)
    graph.add_conditional_edges(
        "sql",             # From this node
        route_after_sql,   # Use this function to decide
        {
            "rag":   "rag",        # If returns "rag", go to rag node
            "sql":   "sql",        # If returns "sql", loop back (retry)
            "error": "error",      # If returns "error", go to error node
        }
    )

    # After RAG: always go to Strategist
    graph.add_edge("rag", "strategist")

    # After Strategist: always end
    graph.add_edge("strategist", END)

    # After Error: always end
    graph.add_edge("error", END)

    return graph.compile()


# Create the compiled graph (module-level, built once)
nexus_graph = build_graph()
logger.success("[LangGraph] Graph compiled successfully.")


# =============================================================================
# PUBLIC API — The one function everything else calls
# =============================================================================

def run_nexus(question: str) -> dict:
    """
    The single entry point for the entire Nexus-ABI system.

    Takes a plain English business question, runs the full multi-agent
    pipeline, and returns the final strategy.

    Args:
        question: Any business question in plain English.
                  Examples:
                    "What is our current churn rate?"
                    "Why are Enterprise customers leaving?"
                    "Which customers should we call today to prevent churn?"

    Returns:
        dict with:
          question     → original question
          sql_result   → data from SQL Agent
          rag_result   → evidence from RAG Agent
          strategy     → full strategic recommendation from Strategist
          error        → any error that occurred (None if all went well)
    """
    logger.info("[Nexus] Received question: '{}'", question)

    # Initialise state with defaults
    initial_state: NexusState = {
        "question":    question,
        "sql_result":  None,
        "rag_result":  None,
        "strategy":    None,
        "error":       None,
        "retry_count": 0,
    }

    # Run the graph
    final_state = nexus_graph.invoke(initial_state)

    logger.success("[Nexus] Pipeline complete. Priority: {}",
                   final_state.get("strategy", {}).get("priority", "UNKNOWN"))

    return final_state


# =============================================================================
# MAIN — Interactive demo of the full pipeline
# =============================================================================

if __name__ == "__main__":
    from rich.console import Console
    from rich.panel   import Panel
    from rich.rule    import Rule

    console = Console(highlight=False)

    console.print(Panel.fit(
        "[bold cyan]NEXUS-ABI[/bold cyan] | [white]LangGraph Orchestrator[/white]\n"
        "[dim]Full multi-agent pipeline: SQL + RAG + Strategist[/dim]",
        border_style="cyan"
    ))

    # --- Test Questions ---
    questions = [
        "What is our churn rate and what should we do about it?",
        "Which customers are at highest risk and what are they complaining about?",
    ]

    for q in questions:
        console.print(Rule(f"[bold yellow]{q}[/bold yellow]"))

        result = run_nexus(q)
        strategy = result.get("strategy", {})

        # Print the recommendation
        priority = strategy.get("priority", "UNKNOWN")
        border   = "red" if priority == "CRITICAL" else "yellow" if priority == "HIGH" else "green"

        console.print(Panel(
            strategy.get("recommendation", "No recommendation generated."),
            title=f"[bold]PRIORITY: {priority}[/bold]",
            border_style=border,
        ))

        # Print action items
        actions = strategy.get("actions", [])
        if actions:
            console.print("\n[bold green]Action Items:[/bold green]")
            for i, action in enumerate(actions, 1):
                console.print(f"  {i}. {action[:100]}...")

        # Evidence trail (audit log)
        evidence = strategy.get("evidence", {})
        console.print(f"\n[dim]Evidence trail: {evidence}[/dim]\n")
