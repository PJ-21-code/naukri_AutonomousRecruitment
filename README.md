# Naukri.com Recruitment & HR Support Agent
## LangGraph Production-Grade Capstone Project

A robust, production-grade domain support agent built for Naukri.com's recruitment ecosystem. The system operates entirely offline using local embeddings, vector retrieval, state checkpointing, and model orchestration under strict guardrails and PII masking policies.

---

## Architecture & Tech Stack

* **Orchestration**: LangGraph (Multi-node graph with dynamic intent routing, state memory, and resilience policies)
* **Backend API**: FastAPI & Uvicorn (Endpoints for querying and document ingestion with JSON-Lines trace logging)
* **Vector Retrieval (RAG)**: ChromaDB with dual chunking strategies (Fixed-size vs. Sentence-based) powered by local `SentenceTransformers` (`all-MiniLM-L6-v2`)
* **State Persistence**: LangGraph Checkpoint SQLite (`checkpoints.sqlite`)
* **Interoperability**: Model Context Protocol (`fastmcp`) for tool exposure over local transport
* **Evaluation & Guardrails**: Custom RAG Triad judge, PII input/output masking, similarity fallback threshold, and Pydantic schema validation

---

## Project Directory Structure

```text
├── data/
│   └── job_application.json          # Seeded deterministic candidate dataset (40+ records)
├── evaluation/
│   └── rag_triad.py                  # RAG Triad evaluation script (Context Relevance, Groundedness, Answer Relevance)
├── knowledgeBase/
│   ├── eligibility_criteria.txt      # 12+ domain-specific HR & recruitment knowledge documents
│   ├── interview_process.txt
│   ├── notice_period.txt
│   └── ... (other HR policy text files)
├── logs/
│   └── app_traces.jsonl              # PII-masked JSON-Lines runtime request/response trace logs
├── src/
│   ├── agent.py                      # LangGraph agent definition, nodes, routing, and guardrails
│   ├── app.py                        # FastAPI server wrapping the agent and logging middleware
│   ├── dataset.py                    # Script generating the deterministic dataset
│   ├── mcp_client.py                 # Local MCP client testing tool execution over HTTP (/mcp)
│   ├── mcp_server.py                 # Self-hosted MCP server wrapping application status lookup using fastmcp
│   ├── rag_core.py                   # ChromaDB indexing, dual chunking, and similarity threshold fallback
│   └── tools.py                      # Application status lookup tool and escalation scoring formula
├── test/
│   ├── checkpoining.py               # Demonstrates state persistence and resume-from-checkpoint capability
│   ├── resilience.py                 # Demonstrates timeouts, retries, and exponential backoff
│   └── checkpoints.sqlite            # Local SQLite database for graph state checkpoints (git-ignored)
├── .gitignore
├── requirements.txt
└── README.md
```
## Installation Guide

1. Clone & Set Up Virtual Environment
Ensure you have Python 3.10+ installed on your local machine. Open your terminal in the project root and run:

python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

2. Install Dependencies
Install all required capstone packages offline via pip:

pip install -r requirements.txt

## Trials & Execution Workflows

Step 1: Initialize Data & Knowledge Base
Generate your deterministic candidate dataset and build your ChromaDB vector store collections:
python -m src.dataset
python -m src.rag_core

Step 2: Run the FastAPI Backend Server
Launch the production-grade FastAPI server locally:
uvicorn src.app:app --reload 

Access interactive API docs at: http://127.0.0.1:8000/docs

Send test queries to POST /ask using a JSON payload:

{
  "query": "What is the policy for notice periods?",
  "thread_id": "session_001"
}

Step 3: Test Model Context Protocol (MCP) Interoperability
Verify tool exposure using the self-hosted MCP server and client script:
python -m src.mcp_server
# In a separate terminal window:
python -m src.mcp_client

Step 4: Execute Checkpointing and Resilience Trials
Verify state persistence across interrupted sessions and error recovery handling:
python -m test.checkpointing
python -m test.resilience

Step 5: Run the RAG Triad Evaluation Suite
Execute the evaluation harness to compute Context Relevance, Groundedness, and Answer Relevance metrics across the test dataset:
python -m evaluation.rag_triad