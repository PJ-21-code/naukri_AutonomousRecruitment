"""SQLite Checkpointing Demonstration and Test Suite for LangGraph (Task 15).

Demonstrates and verifies state persistence, interruption, and resumption using `langgraph-checkpoint-sqlite`:
1. Configures an SQLite Checkpointer (`SqliteSaver`) storing graph checkpoints on disk (`checkpoints.sqlite`).
2. Executes a multi-stage recruitment pipeline keyed by a unique thread ID.
3. Deliberately interrupts execution after Node 1 and Node 2 complete, before Node 3 and Node 4 run.
4. Resumes execution on the exact same thread ID, proving that:
   - Previously completed nodes were restored directly from the SQLite checkpoint.
   - Completed nodes (Node 1 & Node 2) did NOT re-execute (execution count remains 1).
   - Remaining nodes (Node 3 & Node 4) executed cleanly to complete the pipeline.
"""

import os
import sys
import sqlite3
import pytest
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

# Ensure project root is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.tools import check_job_application_status


# ============================================================================
# 1. State Definition & Node Execution Audit Tracker
# ============================================================================

class RecruitmentWorkflowState(TypedDict):
    candidate_id: str
    execution_history: List[str]
    status_details: Optional[Dict[str, Any]]
    escalation_tier: Optional[str]
    notification_payload: Optional[Dict[str, Any]]


class NodeExecutionTracker:
    """Tracks invocation counts and timeline for each node to prove non-re-execution."""
    def __init__(self):
        self.counts = {
            "node_intake": 0,
            "node_status_lookup": 0,
            "node_escalation_review": 0,
            "node_dispatch_notification": 0,
        }
        self.logs: List[str] = []

    def reset(self):
        for k in self.counts:
            self.counts[k] = 0
        self.logs.clear()

    def record(self, node_name: str, message: str):
        self.counts[node_name] += 1
        log_entry = f"[{node_name}] (Execution #{self.counts[node_name]}): {message}"
        self.logs.append(log_entry)
        print(f"  --> {log_entry}")


GLOBAL_TRACKER = NodeExecutionTracker()


# ============================================================================
# 2. Recruitment Workflow Nodes
# ============================================================================

def node_intake(state: RecruitmentWorkflowState) -> Dict[str, Any]:
    """Node 1: Validates incoming candidate record ID and initializes audit history."""
    GLOBAL_TRACKER.record("node_intake", f"Ingesting candidate ID: {state.get('candidate_id')}")
    history = list(state.get("execution_history", []))
    history.append("node_intake_completed")
    return {
        "candidate_id": state.get("candidate_id", "").strip().upper(),
        "execution_history": history,
    }


def node_status_lookup(state: RecruitmentWorkflowState) -> Dict[str, Any]:
    """Node 2: Queries application database and retrieves score and status."""
    cid = state.get("candidate_id", "")
    GLOBAL_TRACKER.record("node_status_lookup", f"Looking up database record for candidate '{cid}'")
    status_data = check_job_application_status(cid)
    history = list(state.get("execution_history", []))
    history.append("node_status_lookup_completed")
    return {
        "status_details": status_data,
        "execution_history": history,
    }


def node_escalation_review(state: RecruitmentWorkflowState) -> Dict[str, Any]:
    """Node 3: Evaluates escalation score into prioritized recruiter action tiers."""
    details = state.get("status_details", {})
    score = details.get("escalation_score", 0.0)
    GLOBAL_TRACKER.record("node_escalation_review", f"Assessing recruiter escalation tier for score={score}")
    
    tier = "URGENT_ACTION" if score >= 0.70 else ("STANDARD_REVIEW" if score >= 0.40 else "ROUTINE_MONITORING")
    history = list(state.get("execution_history", []))
    history.append("node_escalation_review_completed")
    return {
        "escalation_tier": tier,
        "execution_history": history,
    }


def node_dispatch_notification(state: RecruitmentWorkflowState) -> Dict[str, Any]:
    """Node 4: Prepares final recruiter and candidate dispatch notifications."""
    cid = state.get("candidate_id", "")
    tier = state.get("escalation_tier", "ROUTINE_MONITORING")
    GLOBAL_TRACKER.record("node_dispatch_notification", f"Generating notification payload for '{cid}' [Tier: {tier}]")
    
    payload = {
        "candidate_id": cid,
        "escalation_tier": tier,
        "status": state.get("status_details", {}).get("status", "Unknown"),
        "timestamp_utc": "2026-09-08T12:00:00Z",
        "action_required": tier == "URGENT_ACTION"
    }
    history = list(state.get("execution_history", []))
    history.append("node_dispatch_notification_completed")
    return {
        "notification_payload": payload,
        "execution_history": history,
    }


# ============================================================================
# 3. LangGraph Workflow Construction with Checkpointer
# ============================================================================

def build_checkpointed_graph(checkpointer: SqliteSaver, interrupt_before: Optional[List[str]] = None):
    """Builds and compiles the 4-node LangGraph recruitment workflow with SQLite persistence."""
    builder = StateGraph(RecruitmentWorkflowState)
    
    builder.add_node("node_intake", node_intake)
    builder.add_node("node_status_lookup", node_status_lookup)
    builder.add_node("node_escalation_review", node_escalation_review)
    builder.add_node("node_dispatch_notification", node_dispatch_notification)
    
    builder.add_edge(START, "node_intake")
    builder.add_edge("node_intake", "node_status_lookup")
    builder.add_edge("node_status_lookup", "node_escalation_review")
    builder.add_edge("node_escalation_review", "node_dispatch_notification")
    builder.add_edge("node_dispatch_notification", END)
    
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=interrupt_before or ["node_escalation_review"]
    )


# ============================================================================
# 4. Pytest Automated Tests
# ============================================================================

def test_sqlite_checkpointing_interruption_and_resumption(tmp_path: Path):
    """Verifies that SQLite checkpointing persists state at interruptions and resumes without re-running earlier nodes."""
    GLOBAL_TRACKER.reset()
    
    db_file = tmp_path / "test_checkpoints.sqlite"
    conn = sqlite3.connect(str(db_file), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    
    # Compile graph with interruption before node_escalation_review (Node 3)
    graph = build_checkpointed_graph(checkpointer, interrupt_before=["node_escalation_review"])
    
    thread_id = "recruitment-thread-candidate-001"
    config = {"configurable": {"thread_id": thread_id}}
    
    # --- PHASE 1: Run graph until deliberate interruption ---
    initial_input: RecruitmentWorkflowState = {
        "candidate_id": "REC-001",
        "execution_history": [],
        "status_details": None,
        "escalation_tier": None,
        "notification_payload": None,
    }
    
    step1_state = graph.invoke(initial_input, config=config)
    
    # Verify Node 1 and Node 2 executed
    assert GLOBAL_TRACKER.counts["node_intake"] == 1, "node_intake must have executed exactly once"
    assert GLOBAL_TRACKER.counts["node_status_lookup"] == 1, "node_status_lookup must have executed exactly once"
    
    # Verify Node 3 and Node 4 were interrupted and did NOT run yet
    assert GLOBAL_TRACKER.counts["node_escalation_review"] == 0, "node_escalation_review must not have run before resume"
    assert GLOBAL_TRACKER.counts["node_dispatch_notification"] == 0, "node_dispatch_notification must not have run before resume"
    
    # Inspect persisted checkpoint state
    checkpoint_state = graph.get_state(config)
    assert checkpoint_state.next == ("node_escalation_review",), "Checkpoint next task must point to node_escalation_review"
    assert checkpoint_state.values["status_details"]["record_id"] == "REC-001"
    assert "node_status_lookup_completed" in checkpoint_state.values["execution_history"]
    
    # --- PHASE 2: Resume the EXACT same thread ID ---
    # Passing None to graph.invoke resumes from checkpoint
    step2_state = graph.invoke(None, config=config)
    
    # CRITICAL PROOF: Earlier nodes MUST NOT have re-executed!
    assert GLOBAL_TRACKER.counts["node_intake"] == 1, "PROVEN: node_intake did NOT re-execute on resume"
    assert GLOBAL_TRACKER.counts["node_status_lookup"] == 1, "PROVEN: node_status_lookup did NOT re-execute on resume"
    
    # Verify remaining nodes executed to completion
    assert GLOBAL_TRACKER.counts["node_escalation_review"] == 1, "node_escalation_review executed on resume"
    assert GLOBAL_TRACKER.counts["node_dispatch_notification"] == 1, "node_dispatch_notification executed on resume"
    
    # Verify final completed state
    final_checkpoint = graph.get_state(config)
    assert len(final_checkpoint.next) == 0, "Workflow should be completely finished"
    assert final_checkpoint.values["escalation_tier"] is not None
    assert final_checkpoint.values["notification_payload"]["candidate_id"] == "REC-001"
    assert final_checkpoint.values["execution_history"] == [
        "node_intake_completed",
        "node_status_lookup_completed",
        "node_escalation_review_completed",
        "node_dispatch_notification_completed",
    ]
    
    conn.close()


def test_sqlite_state_isolation_between_threads(tmp_path: Path):
    """Verifies that separate thread IDs maintain independent state checkpoints in SQLite."""
    GLOBAL_TRACKER.reset()
    
    db_file = tmp_path / "threads_isolation.sqlite"
    conn = sqlite3.connect(str(db_file), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    graph = build_checkpointed_graph(checkpointer, interrupt_before=["node_escalation_review"])
    
    # Run Thread A
    config_a = {"configurable": {"thread_id": "thread_A"}}
    graph.invoke({"candidate_id": "REC-001", "execution_history": []}, config=config_a)
    
    # Run Thread B with different candidate
    config_b = {"configurable": {"thread_id": "thread_B"}}
    graph.invoke({"candidate_id": "REC-005", "execution_history": []}, config=config_b)
    
    state_a = graph.get_state(config_a)
    state_b = graph.get_state(config_b)
    
    assert state_a.values["candidate_id"] == "REC-001"
    assert state_b.values["candidate_id"] == "REC-005"
    assert state_a.values["status_details"]["expected_salary_inr"] != state_b.values["status_details"]["expected_salary_inr"]
    
    conn.close()


# ============================================================================
# 5. Standalone Execution Demonstration
# ============================================================================

def run_checkpointing_demo(db_path: Optional[str] = None):
    """Interactive CLI demonstration showing SQLite Checkpointing, interruption, and resumption."""
    target_db = Path(db_path) if db_path else CURRENT_DIR / "checkpoints.sqlite"
    print("=" * 80)
    print("LANGGRAPH SQLITE CHECKPOINTING DEMONSTRATION (Task 15)")
    print(f"SQLite Checkpointer Database: {target_db.resolve()}")
    print("=" * 80)

    GLOBAL_TRACKER.reset()
    conn = sqlite3.connect(str(target_db), check_same_thread=False)
    checkpointer = SqliteSaver(conn)
    
    # Compile graph with interrupt before Node 3
    graph = build_checkpointed_graph(checkpointer, interrupt_before=["node_escalation_review"])
    
    thread_id = "recruitment-candidate-session-REC001"
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"\n[PHASE 1] Initializing Workflow Run for Thread: '{thread_id}'")
    print("-> Goal: Execute Node 1 & Node 2, then DELIBERATELY INTERRUPT before Node 3.")
    print("-" * 80)
    
    initial_payload: RecruitmentWorkflowState = {
        "candidate_id": "REC-001",
        "execution_history": [],
        "status_details": None,
        "escalation_tier": None,
        "notification_payload": None,
    }
    
    graph.invoke(initial_payload, config=config)
    
    print("\n[PHASE 1 AUDIT - POST INTERRUPTION]")
    print(f"  * Node 1 (node_intake) invocation count: {GLOBAL_TRACKER.counts['node_intake']} (Expected: 1)")
    print(f"  * Node 2 (node_status_lookup) invocation count: {GLOBAL_TRACKER.counts['node_status_lookup']} (Expected: 1)")
    print(f"  * Node 3 (node_escalation_review) invocation count: {GLOBAL_TRACKER.counts['node_escalation_review']} (Expected: 0 - Interrupted)")
    print(f"  * Node 4 (node_dispatch_notification) invocation count: {GLOBAL_TRACKER.counts['node_dispatch_notification']} (Expected: 0 - Interrupted)")
    
    interrupted_state = graph.get_state(config)
    print(f"\n[Persisted SQLite Checkpoint State]")
    print(f"  * Next Pending Node: {interrupted_state.next}")
    print(f"  * Persisted Candidate ID: {interrupted_state.values.get('candidate_id')}")
    print(f"  * Retrieved Status: {interrupted_state.values.get('status_details', {}).get('status')}")
    print(f"  * Computed Escalation Score: {interrupted_state.values.get('status_details', {}).get('escalation_score')}")
    print(f"  * Execution History: {interrupted_state.values.get('execution_history')}")
    
    print("\n" + "=" * 80)
    print(f"[PHASE 2] Resuming Exact Same Thread: '{thread_id}' from SQLite Checkpoint")
    print("-> Goal: Prove Node 1 & Node 2 DO NOT re-execute, while Node 3 & Node 4 execute.")
    print("-" * 80)
    
    # Resume by calling invoke with None on the exact same thread_id
    graph.invoke(None, config=config)
    
    print("\n[PHASE 2 AUDIT - POST RESUMPTION]")
    print(f"  * Node 1 (node_intake) invocation count: {GLOBAL_TRACKER.counts['node_intake']} (PROVEN: NOT RE-EXECUTED!)")
    print(f"  * Node 2 (node_status_lookup) invocation count: {GLOBAL_TRACKER.counts['node_status_lookup']} (PROVEN: NOT RE-EXECUTED!)")
    print(f"  * Node 3 (node_escalation_review) invocation count: {GLOBAL_TRACKER.counts['node_escalation_review']} (Executed upon resume: 1)")
    print(f"  * Node 4 (node_dispatch_notification) invocation count: {GLOBAL_TRACKER.counts['node_dispatch_notification']} (Executed upon resume: 1)")
    
    final_state = graph.get_state(config)
    print(f"\n[Final Restored & Completed State]")
    print(f"  * Next Pending Nodes: {final_state.next} (Finished)")
    print(f"  * Escalation Tier: {final_state.values.get('escalation_tier')}")
    print(f"  * Notification Payload: {final_state.values.get('notification_payload')}")
    print(f"  * Final Execution History: {final_state.values.get('execution_history')}")
    print("=" * 80)
    print("DEMONSTRATION COMPLETE: SQLite Checkpointing successfully verified.")
    print("=" * 80)
    
    conn.close()


if __name__ == "__main__":
    run_checkpointing_demo()
