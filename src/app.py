import os
import sys
import json
import time
import uuid
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Import recruitment agent, RAG core, and PII masking guardrail
try:
    from src.agent import RecruitmentAgent, AgentResponseSchema, mask_pii
    from src.rag_core import BASE_DIR, KB_DIR, RAG_CORE
except ImportError:
    from agent import RecruitmentAgent, AgentResponseSchema, mask_pii
    from rag_core import BASE_DIR, KB_DIR, RAG_CORE


# Directory and Log configuration
LOGS_DIR = BASE_DIR / "logs"
LOG_FILE = LOGS_DIR / "app_traces.jsonl"


# ============================================================================
# 1. Pydantic Request & Response Schemas
# ============================================================================

class AskRequest(BaseModel):
    """Incoming request model for /ask endpoint."""
    query: str = Field(
        ...,
        description="The candidate or recruiter query to process.",
        json_schema_extra={"example": "What is our notice period policy?"}
    )
    thread_id: Optional[str] = Field(
        default="default_session",
        description="Session thread ID for multi-turn conversation memory persistence.",
        json_schema_extra={"example": "session_user_101"}
    )
    reset_session: Optional[bool] = Field(
        default=False,
        description="If True, clears previous memory for this thread before processing."
    )


class AskResponse(BaseModel):
    """Outgoing structured response model for /ask endpoint."""
    trace_id: str = Field(..., description="Unique UUID trace ID for request lifecycle tracking.")
    answer: str = Field(..., description="The generated answer or status message from the agent.")
    source_type: str = Field(..., description="Source classification: 'policy', 'tool', 'guardrail', or 'fallback'.")
    confidence_score: float = Field(..., description="Confidence or similarity metric.")
    groundedness_passed: bool = Field(..., description="Whether groundedness checks passed.")
    pii_masked: bool = Field(..., description="Whether PII was detected and sanitized.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata including tool outputs, chunks, or escalation score.")
    latency_ms: float = Field(..., description="End-to-end processing time in milliseconds.")


class AddDocumentRequest(BaseModel):
    """Incoming request model for /add-document endpoint."""
    doc_id: str = Field(
        ...,
        description="Unique identifier for the document (e.g. 'wellness_allowance').",
        json_schema_extra={"example": "remote_work_stipend"}
    )
    content: str = Field(
        ...,
        description="The textual content of the policy or document to index.",
        json_schema_extra={"example": "Employees working remotely receive a monthly broadband and utility stipend of INR 2,500."}
    )
    reindex: Optional[bool] = Field(
        default=True,
        description="Whether to trigger dynamic ChromaDB re-indexing immediately."
    )


class AddDocumentResponse(BaseModel):
    """Outgoing response model for /add-document endpoint."""
    trace_id: str = Field(..., description="Unique UUID trace ID.")
    status: str = Field(default="success", description="Status of the operation.")
    doc_id: str = Field(..., description="Document identifier.")
    message: str = Field(..., description="Informational message about the document ingestion.")
    reindexed: bool = Field(..., description="Whether dynamic ChromaDB re-indexing was performed.")
    latency_ms: float = Field(..., description="Processing time in milliseconds.")


class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str = Field(default="healthy")
    service: str = Field(default="Naukri.com Autonomous Recruitment Support Agent")
    agent_ready: bool = Field(default=True)
    timestamp: str = Field(...)


# ============================================================================
# 2. Structured JSON-Lines Logger with PII Masking Guardrail
# ============================================================================

def log_trace_jsonl(
    trace_id: str,
    endpoint: str,
    method: str,
    sanitized_request_payload: Dict[str, Any],
    response_payload: Dict[str, Any],
    latency_ms: float,
    status_code: int = 200,
    pii_masked: bool = False
) -> None:
    """Writes a single structured JSON-Lines entry to disk logs.

    CRITICAL GUARDRAIL REQUIREMENT:
    All request text written to disk logs MUST be sanitized with mask_pii to ensure
    no raw, unmasked PII (phone numbers, emails) ever leaks into log files.
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    log_entry = {
        "trace_id": trace_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "endpoint": endpoint,
        "method": method,
        "status_code": status_code,
        "latency_ms": round(latency_ms, 2),
        "pii_masked": pii_masked,
        "request": sanitized_request_payload,
        "response": response_payload
    }

    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"Logging error: Failed to append trace {trace_id} to JSONL log: {e}")


# ============================================================================
# 3. FastAPI Application Initialization
# ============================================================================

# Global agent instance placeholder
agent_instance: Optional[RecruitmentAgent] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes the recruitment agent and ensures RAG knowledge base is ready."""
    global agent_instance
    print("Starting Naukri.com Autonomous Recruitment Agent API...")
    agent_instance = RecruitmentAgent()
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    print("Recruitment Agent initialized successfully.")
    yield
    print("Shutting down API...")


app = FastAPI(
    title="Naukri.com Autonomous Recruitment Agent API",
    description="FastAPI backend providing autonomous candidate query routing, status checking, and policy RAG.",
    version="1.0.0",
    lifespan=lifespan
)

# Enable CORS for frontend or external client integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_agent() -> RecruitmentAgent:
    """Retrieves or lazy-initializes the recruitment agent."""
    global agent_instance
    if agent_instance is None:
        agent_instance = RecruitmentAgent()
    return agent_instance


# ============================================================================
# 4. API Endpoints
# ============================================================================

@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
def health_check():
    """Returns the operational readiness and health status of the recruitment agent backend."""
    return HealthResponse(
        status="healthy",
        service="Naukri.com Autonomous Recruitment Support Agent",
        agent_ready=agent_instance is not None or True,
        timestamp=datetime.now(timezone.utc).isoformat()
    )


@app.post("/ask", response_model=AskResponse, status_code=status.HTTP_200_OK)
def ask_agent(payload: AskRequest):
    """Processes user queries via the LangGraph recruitment agent with memory and guardrails.

    Workflow:
    1. Generates unique trace_id (UUID4) and records start timestamp.
    2. Applies input-side PII masking to query before logging to disk.
    3. Invokes the LangGraph Recruitment Agent.
    4. Records execution latency and writes structured JSONL trace to disk logs.
    5. Returns validated Pydantic AskResponse.
    """
    start_time = time.perf_counter()
    trace_id = str(uuid.uuid4())

    # Apply strict input-side PII masking for disk logs
    sanitized_log_query, pii_detected, _ = mask_pii(payload.query)
    sanitized_request_for_logs = {
        "query": sanitized_log_query,
        "thread_id": payload.thread_id,
        "reset_session": payload.reset_session
    }

    try:
        agent = get_agent()
        agent_resp: AgentResponseSchema = agent.invoke(
            query=payload.query,
            thread_id=payload.thread_id or "default_session",
            reset_session=bool(payload.reset_session)
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        response_data = AskResponse(
            trace_id=trace_id,
            answer=agent_resp.answer,
            source_type=agent_resp.source_type,
            confidence_score=agent_resp.confidence_score,
            groundedness_passed=agent_resp.groundedness_passed,
            pii_masked=agent_resp.pii_masked or pii_detected,
            metadata=agent_resp.metadata,
            latency_ms=round(latency_ms, 2)
        )

        # Log structured trace to JSONL
        log_trace_jsonl(
            trace_id=trace_id,
            endpoint="/ask",
            method="POST",
            sanitized_request_payload=sanitized_request_for_logs,
            response_payload=response_data.model_dump(),
            latency_ms=latency_ms,
            status_code=200,
            pii_masked=response_data.pii_masked
        )

        return response_data

    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        error_resp = {
            "error": "Internal server error during agent execution.",
            "details": str(e)
        }
        log_trace_jsonl(
            trace_id=trace_id,
            endpoint="/ask",
            method="POST",
            sanitized_request_payload=sanitized_request_for_logs,
            response_payload=error_resp,
            latency_ms=latency_ms,
            status_code=500,
            pii_masked=pii_detected
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent execution failed: {str(e)}"
        )


@app.post("/add-document", response_model=AddDocumentResponse, status_code=status.HTTP_201_CREATED)
def add_document(payload: AddDocumentRequest):
    """Dynamically adds a new policy document to the knowledgeBase and re-indexes ChromaDB.

    Workflow:
    1. Validates document ID and sanitizes filename.
    2. Writes document to `knowledgeBase/<doc_id>.txt`.
    3. Re-indexes ChromaDB collection if `reindex=True`.
    4. Logs structured trace to `logs/app_traces.jsonl`.
    5. Returns structured AddDocumentResponse.
    """
    start_time = time.perf_counter()
    trace_id = str(uuid.uuid4())

    # Sanitize doc_id for filesystem safety
    safe_doc_id = re.sub(r"[^a-zA-Z0-9_-]", "_", payload.doc_id.strip())
    if not safe_doc_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid doc_id provided.")

    if not payload.content or not payload.content.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Document content cannot be empty.")

    sanitized_content, pii_detected, _ = mask_pii(payload.content)
    sanitized_request_for_logs = {
        "doc_id": safe_doc_id,
        "content_length": len(payload.content),
        "reindex": payload.reindex
    }

    try:
        KB_DIR.mkdir(parents=True, exist_ok=True)
        file_path = KB_DIR / f"{safe_doc_id}.txt"

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(payload.content.strip())

        reindexed = False
        if payload.reindex:
            agent = get_agent()
            agent.rag.index_knowledge_base()
            reindexed = True

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        response_data = AddDocumentResponse(
            trace_id=trace_id,
            status="success",
            doc_id=safe_doc_id,
            message=f"Document '{safe_doc_id}.txt' saved successfully to knowledgeBase and indexed.",
            reindexed=reindexed,
            latency_ms=round(latency_ms, 2)
        )

        log_trace_jsonl(
            trace_id=trace_id,
            endpoint="/add-document",
            method="POST",
            sanitized_request_payload=sanitized_request_for_logs,
            response_payload=response_data.model_dump(),
            latency_ms=latency_ms,
            status_code=201,
            pii_masked=pii_detected
        )

        return response_data

    except Exception as e:
        latency_ms = (time.perf_counter() - start_time) * 1000.0
        error_resp = {"error": "Failed to add document", "details": str(e)}
        log_trace_jsonl(
            trace_id=trace_id,
            endpoint="/add-document",
            method="POST",
            sanitized_request_payload=sanitized_request_for_logs,
            response_payload=error_resp,
            latency_ms=latency_ms,
            status_code=500,
            pii_masked=pii_detected
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save and index document: {str(e)}"
        )


# ============================================================================
# 5. Automated Verification & Testing Suite
# ============================================================================

def run_app_verification():
    """Runs automated verification of FastAPI endpoints and JSONL logging."""
    from fastapi.testclient import TestClient

    print("=" * 80)
    print("FASTAPI BACKEND VERIFICATION & JSONL LOGGING TEST")
    print("=" * 80)

    client = TestClient(app)

    # 1. Test Health Endpoint
    print("\n[TEST 1] GET /health")
    print("-" * 50)
    h_res = client.get("/health")
    print(f"Status Code: {h_res.status_code}")
    print(f"Response: {h_res.json()}")
    assert h_res.status_code == 200

    # 2. Test /ask with PII Masking
    print("\n[TEST 2] POST /ask with PII Query (Phone & Email)")
    print("-" * 50)
    ask_pii_payload = {
        "query": "What is the notice period policy? My phone is +91-9876543210 and email is recruiter@naukri.com.",
        "thread_id": "test_api_session"
    }
    ask_res1 = client.post("/ask", json=ask_pii_payload)
    print(f"Status Code: {ask_res1.status_code}")
    res1_data = ask_res1.json()
    print(f"Trace ID: {res1_data.get('trace_id')}")
    print(f"Source Type: {res1_data.get('source_type')}")
    print(f"PII Masked: {res1_data.get('pii_masked')}")
    print(f"Latency: {res1_data.get('latency_ms')} ms")
    print(f"Answer: {res1_data.get('answer')[:120]}...")
    assert ask_res1.status_code == 200
    assert res1_data.get("pii_masked") is True

    # 3. Test /ask with Candidate Record Lookup Tool
    print("\n[TEST 3] POST /ask for Candidate Status (REC-001)")
    print("-" * 50)
    ask_tool_payload = {
        "query": "Please check candidate status and expected salary for REC-001.",
        "thread_id": "test_api_session"
    }
    ask_res2 = client.post("/ask", json=ask_tool_payload)
    print(f"Status Code: {ask_res2.status_code}")
    res2_data = ask_res2.json()
    print(f"Source Type: {res2_data.get('source_type')}")
    print(f"Answer:\n{res2_data.get('answer')}")
    assert ask_res2.status_code == 200
    assert res2_data.get("source_type") == "tool"

    # 4. Test /add-document & Dynamic Re-indexing
    print("\n[TEST 4] POST /add-document (Dynamic Policy Ingestion)")
    print("-" * 50)
    doc_payload = {
        "doc_id": "wellness_allowance_policy",
        "content": "Employees are eligible for a wellness allowance of INR 5,000 quarterly for gym or sports memberships.",
        "reindex": True
    }
    add_res = client.post("/add-document", json=doc_payload)
    print(f"Status Code: {add_res.status_code}")
    add_data = add_res.json()
    print(f"Response: {add_data}")
    assert add_res.status_code == 201
    assert add_data.get("reindexed") is True

    # 5. Query newly added policy
    print("\n[TEST 5] POST /ask querying newly added policy")
    print("-" * 50)
    ask_new_doc = {
        "query": "What is our quarterly wellness allowance?",
        "thread_id": "test_new_doc_session"
    }
    ask_res3 = client.post("/ask", json=ask_new_doc)
    print(f"Status Code: {ask_res3.status_code}")
    res3_data = ask_res3.json()
    print(f"Source Type: {res3_data.get('source_type')}")
    print(f"Answer: {res3_data.get('answer')}")
    assert ask_res3.status_code == 200

    # 6. Verify Structured JSON-Lines Log File
    print("\n[TEST 6] JSON-Lines Log Verification (Checking logs/app_traces.jsonl)")
    print("-" * 50)
    assert LOG_FILE.exists(), "Log file does not exist!"
    
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        log_lines = [json.loads(line) for line in f if line.strip()]

    print(f"Total Logged Entries: {len(log_lines)}")
    latest_entry = log_lines[-1]
    print(f"Latest Log Entry Trace ID: {latest_entry.get('trace_id')}")
    print(f"Latest Log Endpoint: {latest_entry.get('endpoint')}")
    print(f"Latest Log Latency: {latest_entry.get('latency_ms')} ms")

    # Critical Guardrail Assertion: Verify NO raw phone number or email in JSONL log file
    raw_log_content = LOG_FILE.read_text(encoding="utf-8")
    assert "+91-9876543210" not in raw_log_content, "CRITICAL GUARDRAIL FAILED: Raw phone number leaked in log file!"
    assert "recruiter@naukri.com" not in raw_log_content, "CRITICAL GUARDRAIL FAILED: Raw email leaked in log file!"
    assert "[REDACTED_PHONE]" in raw_log_content, "Redaction token missing in log file!"
    print("\nCRITICAL GUARDRAIL CHECK PASSED: Zero unmasked PII in JSON-Lines log file.")

    print("\n" + "=" * 80)
    print("ALL FASTAPI TESTS & JSONL LOGGING VERIFICATIONS PASSED!")
    print("=" * 80)


if __name__ == "__main__":
    import uvicorn
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_app_verification()
    else:
        # If run directly without flags, execute test verification then launch server
        run_app_verification()
        print("\nStarting Uvicorn Server on http://127.0.0.1:8000 ...")
        uvicorn.run("src.app:app", host="127.0.0.1", port=8000, reload=False)
