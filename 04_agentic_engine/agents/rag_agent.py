"""
=============================================================================
NEXUS-ABI | Layer 4: Agentic Engine -> Agents
File: rag_agent.py
=============================================================================

PURPOSE:
  The RAG (Retrieval-Augmented Generation) Research Agent.
  Searches the 2,000 support ticket logs using semantic similarity to find
  real evidence for business questions — then asks Llama 3 to synthesise
  the findings into an insight.

WHY RAG AND NOT JUST SQL?
  Support tickets are UNSTRUCTURED text. You can't SQL-query:
    "Find tickets where customers sound frustrated about billing."
  But RAG can. It embeds every ticket as a vector, then finds tickets
  that are semantically similar to your question — even if they use
  completely different words.

  SQL Agent answers: "Churn rate is 20.64%"  (the number)
  RAG Agent answers:  "The top reason customers mention is API instability
                       and unresolved billing disputes from Q1."  (the why)

HOW IT WORKS:
  Phase 1 — INGESTION (one time):
    1. Load support_logs.csv
    2. Embed each ticket text using sentence-transformers (all-MiniLM-L6-v2)
    3. Store vectors + metadata in ChromaDB (local, persistent)

  Phase 2 — QUERY (every time):
    1. Embed the user's question
    2. Find the N most similar support tickets (cosine similarity)
    3. Feed those tickets as context to Llama 3
    4. Llama 3 synthesises an insight grounded in real ticket evidence

VECTOR STORE:
  ChromaDB stores the vectors at: 04_agentic_engine/vectorstore/
  After first ingestion, queries are instant (no re-embedding needed).

INPUT:  A natural language question (str)
OUTPUT: dict with keys:
          question      -> original question
          retrieved     -> list of matching ticket dicts
          insight       -> Llama 3's synthesised analysis
          num_retrieved -> how many tickets were used as context
=============================================================================
"""

import os
import sys
from pathlib import Path

os.environ["PYTHONUTF8"] = "1"

PROJECT_ROOT = Path(__file__).parent.parent.parent
LAYER_4_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LAYER_4_ROOT))

import pandas as pd
from loguru import logger
from langchain_core.prompts import ChatPromptTemplate

from utils.llm_config import get_llm

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
VECTORSTORE_DIR = LAYER_4_ROOT / "vectorstore"
VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = "nexus_support_logs"
EMBED_MODEL     = "all-MiniLM-L6-v2"   # Fast, local, no API key needed
TOP_K           = 5                      # Number of tickets to retrieve per query
DATA_PATH       = PROJECT_ROOT / "01_data_pipeline" / "raw" / "support_logs.csv"


# ---------------------------------------------------------------------------
# PROMPT TEMPLATES
# ---------------------------------------------------------------------------

RAG_INSIGHT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a senior Customer Success analyst at NexaCorp, a B2B SaaS company.
You have been given a business question and a set of real customer support tickets as evidence.
Your job is to synthesise the tickets into a clear, concise business insight.

RULES:
- Ground every claim in the actual ticket content provided. Do not invent details.
- Identify patterns across tickets (recurring issues, sentiments, customers at risk).
- If a named customer or company appears multiple times, call that out.
- Keep your response to 3-5 sentences. Be direct and actionable.
- Do NOT say "based on the provided tickets". Just state the insight directly.
"""),
    ("human", """Question: {question}

Relevant Support Tickets:
{context}

Insight:"""),
])


# =============================================================================
# RAG AGENT CLASS
# =============================================================================

class RAGAgent:
    """
    The RAG Research Agent — finds real support ticket evidence for business questions.

    Usage:
        agent = RAGAgent()
        agent.ingest()                              # One-time setup
        result = agent.query("Why are Enterprise customers churning?")
        print(result["insight"])
        print(result["retrieved"])
    """

    def __init__(self):
        self.llm        = get_llm(temperature=0.2)  # Slightly higher temp for synthesis
        self._collection = None
        self._client     = None
        logger.info("RAGAgent initialised. Vectorstore: {}", VECTORSTORE_DIR)

    def _get_collection(self):
        """
        Returns the ChromaDB collection (creates client lazily).
        Lazy init means we don't open ChromaDB until actually needed.
        """
        if self._collection is not None:
            return self._collection

        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        logger.info("Connecting to ChromaDB at {}...", VECTORSTORE_DIR)
        self._client = chromadb.PersistentClient(path=str(VECTORSTORE_DIR))

        # SentenceTransformer embedding function — fully local, no API
        ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

        self._collection = self._client.get_or_create_collection(
            name               = COLLECTION_NAME,
            embedding_function = ef,
            metadata           = {"hnsw:space": "cosine"},  # Cosine similarity
        )

        logger.info(
            "ChromaDB collection '{}' ready. Documents: {}",
            COLLECTION_NAME,
            self._collection.count()
        )
        return self._collection

    def ingest(self, force_reload: bool = False) -> int:
        """
        One-time ingestion: loads support_logs.csv → ChromaDB.

        Args:
            force_reload: If True, deletes and rebuilds the collection.
                          If False, skips ingestion if data already exists.

        Returns:
            Number of documents in the collection after ingestion.
        """
        collection = self._get_collection()

        # Skip if already loaded (unless force reload)
        existing_count = collection.count()
        if existing_count > 0 and not force_reload:
            logger.info(
                "Collection already has {:,} documents. Skipping ingestion. "
                "(Pass force_reload=True to re-ingest)",
                existing_count
            )
            return existing_count

        # Delete and recreate if force reload
        if force_reload and existing_count > 0:
            logger.warning("Force reload: deleting existing {} documents.", existing_count)
            import chromadb
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            self._client.delete_collection(COLLECTION_NAME)
            ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
            self._collection = self._client.get_or_create_collection(
                name=COLLECTION_NAME,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
            collection = self._collection

        # Load CSV
        if not DATA_PATH.exists():
            raise FileNotFoundError(
                f"support_logs.csv not found at {DATA_PATH}.\n"
                "Run: python 01_data_pipeline/synthetic_gen.py first."
            )

        logger.info("Loading support logs from {}...", DATA_PATH)
        df = pd.read_csv(DATA_PATH)
        logger.info("Loaded {:,} support tickets for ingestion.", len(df))

        # Prepare documents for ChromaDB
        # ChromaDB needs: ids (unique), documents (text), metadatas (dict)
        ids       = df["log_id"].astype(str).tolist()
        documents = df["text"].fillna("").tolist()
        metadatas = df[["customer_id", "issue_type", "sentiment", "date"]].to_dict("records")

        # Ingest in batches to avoid memory spikes
        BATCH_SIZE = 100
        total = len(documents)
        for i in range(0, total, BATCH_SIZE):
            batch_ids   = ids[i : i + BATCH_SIZE]
            batch_docs  = documents[i : i + BATCH_SIZE]
            batch_meta  = metadatas[i : i + BATCH_SIZE]
            collection.add(
                ids       = batch_ids,
                documents = batch_docs,
                metadatas = batch_meta,
            )
            logger.info("Ingested {}/{} documents...", min(i + BATCH_SIZE, total), total)

        final_count = collection.count()
        logger.success("Ingestion complete. {:,} documents in ChromaDB.", final_count)
        return final_count

    def query(self, question: str, top_k: int = TOP_K) -> dict:
        """
        Retrieves the most relevant support tickets for the question and
        asks Llama 3 to synthesise an insight grounded in that evidence.

        Args:
            question: Natural language business question
            top_k:    Number of tickets to retrieve (default: 5)

        Returns:
            dict with question, retrieved tickets, insight, num_retrieved
        """
        logger.info("RAGAgent querying: '{}'", question)

        result = {
            "question":      question,
            "retrieved":     [],
            "insight":       None,
            "num_retrieved": 0,
        }

        collection = self._get_collection()

        # Check if collection has data
        if collection.count() == 0:
            logger.warning("ChromaDB collection is empty. Run agent.ingest() first.")
            result["insight"] = (
                "The RAG knowledge base has not been populated yet. "
                "Run RAGAgent.ingest() to load support tickets."
            )
            return result

        # ---- STEP 1: Semantic search ----
        query_results = collection.query(
            query_texts = [question],
            n_results   = top_k,
            include     = ["documents", "metadatas", "distances"],
        )

        documents = query_results["documents"][0]
        metadatas = query_results["metadatas"][0]
        distances = query_results["distances"][0]

        # Build retrieved ticket dicts
        retrieved = []
        for doc, meta, dist in zip(documents, metadatas, distances):
            retrieved.append({
                "text":       doc,
                "issue_type": meta.get("issue_type", ""),
                "sentiment":  meta.get("sentiment", ""),
                "date":       meta.get("date", ""),
                "similarity": round(1 - dist, 4),  # Convert distance to similarity score
            })

        result["retrieved"]     = retrieved
        result["num_retrieved"] = len(retrieved)
        logger.success("Retrieved {} relevant tickets (top similarity: {})",
                       len(retrieved), retrieved[0]["similarity"] if retrieved else 0)

        # ---- STEP 2: Synthesise insight with Llama 3 ----
        # Format tickets as numbered context for the LLM
        context_lines = []
        for i, ticket in enumerate(retrieved, 1):
            context_lines.append(
                f"[Ticket {i}] {ticket['issue_type']} | {ticket['sentiment']} | {ticket['date']}\n"
                f"{ticket['text']}"
            )
        context_str = "\n\n".join(context_lines)

        chain    = RAG_INSIGHT_PROMPT | self.llm
        response = chain.invoke({"question": question, "context": context_str})
        result["insight"] = response.content.strip()

        logger.success("RAG insight generated ({} chars)", len(result["insight"]))
        return result


# =============================================================================
# MAIN — Run this file directly to test the RAG Agent
# =============================================================================

if __name__ == "__main__":
    from rich.console import Console
    from rich.panel import Panel
    from rich.rule import Rule
    from rich.table import Table

    console = Console(highlight=False)
    console.print(Panel.fit(
        "[bold cyan]NEXUS-ABI[/bold cyan] | [white]RAG Agent Test[/white]\n"
        "[dim]Ingesting support logs and querying ChromaDB...[/dim]",
        border_style="cyan"
    ))

    agent = RAGAgent()

    # --- One-time ingestion ---
    console.print("\n[bold yellow]Step 1: Ingesting support logs into ChromaDB...[/bold yellow]")
    count = agent.ingest(force_reload=False)
    console.print(f"[green]Collection size: {count:,} documents[/green]\n")

    # --- Test queries ---
    test_questions = [
        "Why are customers threatening to cancel their subscriptions?",
        "What billing issues are customers complaining about?",
        "Which customers are at highest churn risk right now?",
    ]

    for q in test_questions:
        console.print(Rule(f"[bold yellow]{q}[/bold yellow]"))
        result = agent.query(q)

        # Show top 3 retrieved tickets
        console.print(f"[dim]Retrieved {result['num_retrieved']} tickets[/dim]")
        tbl = Table(show_header=True, header_style="bold magenta", show_lines=True)
        tbl.add_column("Similarity", width=10, justify="center")
        tbl.add_column("Type",       width=25)
        tbl.add_column("Sentiment",  width=10)
        tbl.add_column("Snippet",    width=60)
        for ticket in result["retrieved"][:3]:
            tbl.add_row(
                str(ticket["similarity"]),
                ticket["issue_type"],
                ticket["sentiment"],
                ticket["text"][:80] + "...",
            )
        console.print(tbl)
        console.print(f"\n[bold green]Insight:[/bold green] {result['insight']}\n")
