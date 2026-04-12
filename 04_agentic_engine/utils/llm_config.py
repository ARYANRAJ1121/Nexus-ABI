"""
=============================================================================
NEXUS-ABI | Layer 4: Agentic Engine → Utils
File: llm_config.py
=============================================================================

PURPOSE:
  Single configuration point for the local LLM (Llama 3 via Ollama).
  Every agent imports get_llm() from here. Nobody hardcodes model names
  or URLs inside agent code.

WHY OLLAMA?
  Ollama lets you run open-source LLMs (Llama 3, Mistral, Gemma) locally
  with a one-line terminal command. No API key. No cost. No data leaving
  your machine. On your Ryzen machine, Llama 3 8B runs comfortably.

  Ollama exposes a REST API on localhost:11434 — LangChain connects to it
  exactly like it would connect to OpenAI, just pointing at a different URL.

HOW TO START OLLAMA (one-time setup):
  1. Install from: https://ollama.com
  2. Pull the model: ollama pull llama3
  3. It starts automatically as a background service.
     Verify with: ollama list

WHAT THIS FILE PROVIDES:
  get_llm()          → Returns a LangChain-compatible ChatOllama object
  get_embeddings()   → Returns a local embedding model for ChromaDB (RAG)
  health_check()     → Tests if Ollama is reachable before agents start
  OLLAMA_CONFIG      → Dict of all settings (importable by other modules)

SWITCHING MODELS:
  Change DEFAULT_MODEL below from "llama3" to "mistral" or "gemma:7b"
  and ALL agents automatically use the new model. Zero other changes.
=============================================================================
"""

import sys
import io
import httpx
from loguru import logger

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# CONFIGURATION — Change these to swap models or hosts
# ---------------------------------------------------------------------------
OLLAMA_CONFIG = {
    "base_url":     "http://localhost:11434",   # Ollama's default REST API port
    "model":        "llama3",                   # Model name (must be pulled via 'ollama pull llama3')
    "temperature":  0.1,                        # Low temp = more factual, less creative
                                                # For a BI system we want precision, not poetry
    "timeout":      120,                        # Seconds before giving up on a slow response
    "num_ctx":      4096,                       # Context window size (tokens)
                                                # Llama 3 8B supports up to 8192
    "embedding_model": "nomic-embed-text",      # Local embedding model for ChromaDB
                                                # Pull with: ollama pull nomic-embed-text
}


# ---------------------------------------------------------------------------
# HEALTH CHECK — Always run before starting agents
# ---------------------------------------------------------------------------

def health_check() -> dict:
    """
    Pings the Ollama API to verify it's running and the model is available.

    Returns a dict with:
      status: "ok" | "ollama_not_running" | "model_not_found"
      message: Human-readable description
      models: List of available models (if Ollama is running)

    WHY THIS MATTERS:
      If an agent starts without this check and Ollama is down, it will
      fail with a cryptic connection error deep inside LangChain.
      This gives a clear, actionable error message upfront.
    """
    result = {"status": "ok", "message": "", "models": []}

    try:
        response = httpx.get(
            f"{OLLAMA_CONFIG['base_url']}/api/tags",
            timeout=5.0
        )
        response.raise_for_status()

        data   = response.json()
        models = [m["name"] for m in data.get("models", [])]
        result["models"] = models

        configured_model = OLLAMA_CONFIG["model"]
        # Check if our model is available (name can be "llama3" or "llama3:latest")
        model_available = any(
            configured_model in m for m in models
        )

        if model_available:
            result["status"]  = "ok"
            result["message"] = f"Ollama running. Model '{configured_model}' is ready."
            logger.success("Ollama health check passed. Model: {}", configured_model)
        else:
            result["status"]  = "model_not_found"
            result["message"] = (
                f"Ollama is running but model '{configured_model}' is not pulled.\n"
                f"Fix: run  ollama pull {configured_model}\n"
                f"Available models: {models}"
            )
            logger.warning(result["message"])

    except httpx.ConnectError:
        result["status"]  = "ollama_not_running"
        result["message"] = (
            "Ollama is not running on localhost:11434.\n"
            "Fix: Install Ollama from https://ollama.com and start it.\n"
            "Then run: ollama pull llama3"
        )
        logger.error(result["message"])

    except Exception as e:
        result["status"]  = "error"
        result["message"] = f"Unexpected error during health check: {e}"
        logger.error(result["message"])

    return result


# ---------------------------------------------------------------------------
# LLM FACTORY — What every agent calls
# ---------------------------------------------------------------------------

def get_llm(temperature: float = None, model: str = None):
    """
    Returns a LangChain-compatible ChatOllama LLM object.

    Usage in agents:
        from utils.llm_config import get_llm
        llm = get_llm()
        response = llm.invoke("Analyse our churn rate.")

    Args:
        temperature: Override default temperature (0.0-1.0).
                     Use 0.0 for SQL generation (need exact syntax).
                     Use 0.3 for strategic analysis (need some nuance).
        model:       Override default model name.

    Returns:
        ChatOllama instance ready for LangChain chains and LangGraph nodes.
    """
    try:
        from langchain_ollama import ChatOllama
    except ImportError:
        logger.error("langchain-ollama not installed. Run: pip install langchain-ollama")
        raise

    llm = ChatOllama(
        base_url    = OLLAMA_CONFIG["base_url"],
        model       = model or OLLAMA_CONFIG["model"],
        temperature = temperature if temperature is not None else OLLAMA_CONFIG["temperature"],
        num_ctx     = OLLAMA_CONFIG["num_ctx"],
        timeout     = OLLAMA_CONFIG["timeout"],
    )

    logger.info(
        "LLM initialised: model={}, temperature={}",
        model or OLLAMA_CONFIG["model"],
        temperature if temperature is not None else OLLAMA_CONFIG["temperature"]
    )
    return llm


# ---------------------------------------------------------------------------
# EMBEDDINGS FACTORY — Used by the RAG agent to embed support logs
# ---------------------------------------------------------------------------

def get_embeddings():
    """
    Returns a local embedding model for ChromaDB vector storage.

    The RAG agent uses this to:
      1. Embed all support_logs.csv text at ingestion time
      2. Embed user questions at query time
      3. Find semantically similar support tickets

    Uses nomic-embed-text (via Ollama) — a high-quality open-source
    embedding model that outperforms older models like all-MiniLM.

    Pull it once with: ollama pull nomic-embed-text
    """
    try:
        from langchain_ollama import OllamaEmbeddings
        embeddings = OllamaEmbeddings(
            base_url = OLLAMA_CONFIG["base_url"],
            model    = OLLAMA_CONFIG["embedding_model"],
        )
        logger.info("Embeddings model loaded: {}", OLLAMA_CONFIG["embedding_model"])
        return embeddings
    except ImportError:
        logger.warning(
            "langchain-ollama not available. Falling back to sentence-transformers."
        )
        # Fallback: use HuggingFace sentence-transformers (no Ollama needed)
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name = "all-MiniLM-L6-v2"
        )
        logger.info("Embeddings model loaded (fallback): all-MiniLM-L6-v2")
        return embeddings


# =============================================================================
# MAIN — Run this file directly to test your Ollama setup
# =============================================================================

if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    console.print(Panel.fit(
        "[bold cyan]NEXUS-ABI[/bold cyan] | [white]Ollama Health Check[/white]\n"
        "[dim]Verifying local LLM infrastructure...[/dim]",
        border_style="cyan"
    ))

    # --- Run health check ---
    result = health_check()

    status_colour = {
        "ok":                "bold green",
        "model_not_found":   "bold yellow",
        "ollama_not_running":"bold red",
        "error":             "bold red",
    }.get(result["status"], "white")

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key",   style="dim",    width=20)
    table.add_column("Value", style="white",  width=60)

    table.add_row("Status",  f"[{status_colour}]{result['status'].upper()}[/{status_colour}]")
    table.add_row("Message", result["message"])
    table.add_row("Config",  f"URL: {OLLAMA_CONFIG['base_url']} | Model: {OLLAMA_CONFIG['model']}")

    if result["models"]:
        table.add_row("Available Models", ", ".join(result["models"]))

    console.print(table)

    # --- If Ollama is running, test a real inference call ---
    if result["status"] == "ok":
        console.print("\n[bold yellow]Testing live inference...[/bold yellow]")
        try:
            llm      = get_llm(temperature=0.0)
            response = llm.invoke(
                "In one sentence, what is Customer Churn Rate in SaaS?"
            )
            console.print(f"\n[bold green]LLM Response:[/bold green]")
            console.print(f"  [dim]{response.content}[/dim]")
            console.print("\n[bold green]✓ Ollama is fully operational.[/bold green]")
        except Exception as e:
            console.print(f"[red]Inference failed: {e}[/red]")
    else:
        console.print(
            "\n[bold yellow]Ollama is not running — agents will use fallback mode.[/bold yellow]\n"
            "[dim]This is expected if you haven't installed Ollama yet.[/dim]\n"
            "[dim]Run the agents after: ollama pull llama3[/dim]"
        )
