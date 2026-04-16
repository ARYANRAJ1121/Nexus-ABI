<div align="center">

# 🧠 Nexus-ABI
### Agentic Business Intelligence Platform

**Transform raw business data into governed, predictive, and actionable strategic intelligence — fully local, fully free.**

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange?style=flat-square)](https://github.com/langchain-ai/langgraph)
[![Llama 3](https://img.shields.io/badge/Llama_3-Local_LLM-green?style=flat-square)](https://ollama.ai)
[![XGBoost](https://img.shields.io/badge/XGBoost-Churn_Model-red?style=flat-square)](https://xgboost.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-teal?style=flat-square)](https://fastapi.tiangolo.com)
[![Tests](https://img.shields.io/badge/Tests-14%2F14_passing-brightgreen?style=flat-square)](#evaluation)

</div>

---

## What Is This?

Nexus-ABI is a **production-grade multi-agent BI system** that answers business questions like a senior analyst team — without paying for a single API call.

Ask it: *"Why are our Enterprise customers churning and what should we do?"*

It will:
1. Query the database with governed, hallucination-proof SQL
2. Search 2,000 support tickets semantically for real evidence
3. Score at-risk customers with a trained XGBoost model
4. Synthesise everything into a structured strategy with PRIORITY level, root cause, and time-bound action items

Everything runs on your own machine. No OpenAI. No Pinecone. No cloud.

---

## Live Output

```
PRIORITY: CRITICAL

SITUATION:
Churn rate at 20.64% — nearly one-fifth of customers leaving.
Top 5 at-risk accounts represent $438,000/month in MRR.

ROOT CAUSE:
Billing portal outages in Media & Entertainment — customers Kimberly Lin,
Evan Hobbs, Lisa Reilly citing 25-39 days of unresponsive service.

RECOMMENDED ACTIONS:
1. [72h] Emergency outreach to Campbell, Simpson & Anderson, Hayes-Thomas,
   Kane-Long, Whitehead, Lopez & Ward, and Tapia PLC.
2. [4wk] Launch targeted retention campaign with SLA credit offers.
3. [Ongoing] Audit billing portal architecture to prevent recurrence.

FINANCIAL IMPACT:
At current 20.64% churn rate: $51,816 annual revenue at risk.
```

> Every number is from a real SQL query. Every company name is from a real support ticket. Nothing hallucinated.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USER / POWER BI                             │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ HTTP (REST)
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 5: FastAPI Chat Bridge                                       │
│  POST /ask  ·  GET /metrics  ·  GET /health  ·  Swagger /docs      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Layer 4: Agentic Engine (LangGraph)                                │
│                                                                     │
│  ┌─────────────┐    ┌─────────────┐    ┌──────────────────────┐   │
│  │  SQL Agent  │───▶│  RAG Agent  │───▶│  Strategist Agent    │   │
│  │             │    │             │    │                      │   │
│  │ Semantic    │    │ ChromaDB    │    │ SQL + RAG + XGBoost  │   │
│  │ Layer SQL   │    │ 2K tickets  │    │ + Semantic Layer     │   │
│  │ + Llama 3   │    │ + Llama 3   │    │ → Llama 3 strategy   │   │
│  └─────────────┘    └─────────────┘    └──────────────────────┘   │
│         │                                         ▲                │
│         └── retry loop (max 3) ──────────────────┘                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐
│  Layer 3:       │  │  Layer 2:       │  │  Layer 1:        │
│  Semantic Layer │  │  Predictive     │  │  Data Engine     │
│                 │  │  Core           │  │                  │
│  10 KPIs        │  │  XGBoost Churn  │  │  PySpark         │
│  defined as     │  │  Classifier     │  │  cleaning        │
│  code — the     │  │  XGBoost CLV    │  │  10K customers   │
│  anti-halluc.   │  │  Regressor      │  │  50K txns        │
│  governor       │  │  AUC 0.65       │  │  2K tickets      │
└─────────────────┘  └─────────────────┘  └──────────────────┘
```

---

## Tech Stack

| Category | Technology | Why |
|---|---|---|
| **LLM** | Llama 3 via Ollama | Fully local, zero API cost, same quality as hosted models for structured tasks |
| **Agent Framework** | LangGraph | Stateful multi-agent workflows with conditional retry routing |
| **ML Model** | XGBoost | Best-in-class for tabular business data; trains in seconds, fully explainable |
| **Vector Store** | ChromaDB | Local persistent vector database; no cloud dependency |
| **Embeddings** | all-MiniLM-L6-v2 | Fast, local, production-quality semantic search |
| **Database** | SQLite → SQLAlchemy | Zero-setup local DB; swap one line to move to PostgreSQL |
| **API** | FastAPI + Uvicorn | Auto-generated Swagger docs, Pydantic validation, CORS for Power BI |
| **Data Pipeline** | PySpark | Scales from laptop to 1,000-server cluster without code changes |
| **Evaluation** | Custom DeepEval-style suite | 14/14 tests covering all 5 layers |
| **Logging** | Loguru | Structured timestamped logs with colour levels |

---

## Project Structure

```
nexus-abi/
│
├── 01_data_pipeline/
│   ├── synthetic_gen.py       # Faker-based data generator (10K customers, 50K txns, 2K tickets)
│   └── spark_cleaner.py       # PySpark quality pipeline: dedup, null check, anomaly detection
│
├── 02_predictive_core/
│   ├── train_churn.py         # XGBoost churn classifier + CLV regressor with feature engineering
│   └── models/                # Saved model artefacts (.pkl + feature_names.json)
│
├── 03_semantic_layer/
│   └── metrics_defs.py        # 10 canonical KPIs as code: SQL + Python compute + thresholds
│
├── 04_agentic_engine/
│   ├── agents/
│   │   ├── sql_agent.py       # Semantic-Layer-governed SQL + self-correcting Llama 3 fallback
│   │   ├── rag_agent.py       # ChromaDB semantic search + Llama 3 evidence synthesis
│   │   └── strategist_agent.py # Combines all sources → structured business recommendation
│   ├── utils/
│   │   ├── llm_config.py      # Ollama factory, health check, embedding model
│   │   └── db_config.py       # SQLite engine, CSV loader, schema inspector
│   ├── main_graph.py          # LangGraph workflow: SQL → RAG → Strategist with retry routing
│   └── vectorstore/           # ChromaDB persistent vector store (gitignored)
│
├── 05_chat_bridge/
│   ├── chat_bridge.py         # FastAPI server: /ask /metrics /health /schema
│   └── test_api.py            # Endpoint verification script
│
├── 06_evaluation/
│   └── eval_suite.py          # 14-test quality gate across all 5 layers
│
└── requirements.txt
```

---

## Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/download) installed and running
- 8GB RAM minimum (16GB recommended for smooth Llama 3 inference)

### 1. Clone and Create Environment

```bash
git clone https://github.com/ARYANRAJ1121/Nexus-ABI.git
cd Nexus-ABI
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Pull Llama 3

```bash
ollama pull llama3
ollama pull nomic-embed-text
```

Verify Ollama is running:
```bash
curl http://localhost:11434/api/tags
```

### 4. Generate Data

```bash
# Windows: set PYTHONUTF8=1 first to avoid encoding issues
set PYTHONUTF8=1
python 01_data_pipeline/synthetic_gen.py
```

This creates `customers.csv` (10K rows), `transactions.csv` (50K rows), `support_logs.csv` (2K rows) in `01_data_pipeline/raw/`.

### 5. Train the Models

```bash
python 02_predictive_core/train_churn.py
```

This trains XGBoost churn classifier + CLV regressor and saves them to `02_predictive_core/models/`.

---

## Running the System

### Option A — Test Each Layer Individually

```bash
# Test SQL Agent
python 04_agentic_engine/agents/sql_agent.py

# Test RAG Agent (ingests 2K tickets into ChromaDB on first run)
python 04_agentic_engine/agents/rag_agent.py

# Test Strategist (runs full SQL + RAG + Strategy pipeline)
python 04_agentic_engine/agents/strategist_agent.py

# Test LangGraph orchestrator
python 04_agentic_engine/main_graph.py
```

### Option B — Start the API Server

```bash
# From project root
python -m uvicorn "05_chat_bridge.chat_bridge:app" --host 0.0.0.0 --port 8000

# Then open in browser:
# http://localhost:8000/docs   → Swagger UI (interactive)
# http://localhost:8000/health → System status
# http://localhost:8000/metrics → All 10 live KPIs as JSON
```

Query the API:
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is our churn rate and what should we do?"}'
```

### Option C — Run Evaluation Suite

```bash
# Fast mode (no Llama 3, no server needed) — runs in < 5 seconds
python 06_evaluation/eval_suite.py --skip-llm --skip-api

# Full mode (with LLM agents) — runs in ~90 seconds
python 06_evaluation/eval_suite.py --skip-api

# All tests including API (requires server running on port 8000)
python 06_evaluation/eval_suite.py
```

---

## Evaluation Results

```
┌───────────────────────────┐
│ 14/14 tests passed (100%) │
└───────────────────────────┘
```

| Layer | Test | Result |
|---|---|---|
| Layer 1: Data | Files exist + row counts (10K/50K/2K) | ✅ |
| Layer 1: Data | Churn signal planted: Legacy 40.8% > Growth 20.0% | ✅ |
| Layer 2: ML | XGBoost models saved | ✅ |
| Layer 2: ML | High-risk customer prediction: **74.9% churn probability** | ✅ |
| Layer 3: Semantic | All 10 KPIs registered | ✅ |
| Layer 3: Semantic | Formula accuracy: computed = manual = **20.64%** | ✅ |
| Layer 3: Semantic | Interpretation thresholds: Healthy / Warning / Critical | ✅ |
| Layer 4: SQL | Governance: 3/3 KPI questions routed to Semantic Layer | ✅ |
| Layer 4: SQL | Accuracy: churn rate returns exactly 20.64% | ✅ |
| Layer 4: RAG | Retrieval: 5 tickets retrieved for billing question | ✅ |
| Layer 4: RAG | Relevance: 5/5 retrieved tickets contain billing keywords | ✅ |
| Layer 4: Strategist | Output format: PRIORITY + ACTIONS fields present | ✅ |
| Layer 4: LangGraph | End-to-end: full pipeline returns CRITICAL strategy | ✅ |
| Layer 5: API | GET /health: HTTP 200, all systems green | ✅ |

---

## Key Design Decisions

**Why local Llama 3 instead of GPT-4?**
GPT-4 costs money per token, sends your business data to OpenAI's servers, and requires internet. Llama 3 on Ollama is free, private, and produces equivalent quality for structured reasoning tasks.

**Why the Semantic Layer?**
Without it, the LLM computes KPIs differently every time — this is called metric drift. Defining churn rate, MRR, and CLV once in code ensures the AI can never invent a wrong formula. Airbnb calls their version "Minerva." Spotify calls theirs "Lexikon."

**Why XGBoost instead of a neural network?**
XGBoost trains in seconds on 10K rows, requires no GPU, and tells you *which features* caused the prediction. Neural networks need 100x more data, are black boxes, and are harder to defend in a business context.

**Why LangGraph instead of plain LangChain?**
LangGraph provides shared state between agents and conditional routing with automatic retry. If the SQL Agent fails, the graph reroutes — it doesn't crash. Plain LangChain chains have no loop-back capability.

**Why SQLite instead of PostgreSQL?**
Zero setup. No server. No credentials. The SQLAlchemy abstraction means switching to PostgreSQL in production requires changing exactly one line (the connection URL).

---

## Power BI Integration

With the API server running, connect Power BI directly:

1. **Get Data** → **Web** → URL: `http://localhost:8000/metrics`
2. Power BI fetches the JSON and maps all 10 KPIs automatically
3. Add a **Text/Web** visual that posts to `http://localhost:8000/ask` for free-text queries

---

## What Each Agent Does

**SQL Agent** — The analyst. Translates questions to SQL. For known KPIs, uses the pre-written Semantic Layer SQL. For everything else, Llama 3 generates SQL with a 3-attempt self-correction loop. Tested on questions like *"Which industry has the highest average monthly spend?"* — generated a correct multi-table query on the first attempt.

**RAG Agent** — The researcher. Converts 2,000 support tickets to 384-dimensional vectors using `all-MiniLM-L6-v2`. On each query, finds the 5 most semantically similar tickets using cosine similarity in ChromaDB. Llama 3 then synthesises the tickets into evidence. No keyword matching — pure semantic understanding.

**Strategist Agent** — The advisor. Receives SQL numbers + RAG evidence + XGBoost churn probabilities + Semantic Layer interpretation. Produces a structured output: PRIORITY level (CRITICAL/HIGH/MEDIUM/LOW), SITUATION, ROOT CAUSE, RECOMMENDED ACTIONS (numbered, time-bound), and FINANCIAL IMPACT.

---

## Windows Notes

```bash
# Set this before running any script to prevent Unicode errors with Rich
set PYTHONUTF8=1

# PySpark requires winutils.exe for writing Parquet on Windows
# Install from: https://github.com/cdarlint/winutils
# Or use: pip install pyspark and run cleaner in WSL for production
```

---

<div align="center">

Built with 🧠 Llama 3 · 📊 XGBoost · 🔗 LangGraph · ⚡ FastAPI

</div>