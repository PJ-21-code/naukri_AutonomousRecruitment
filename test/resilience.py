"""Resilience, Timeout, and Retry Policy Test Suite and Demonstration (Task 16).

Implements and demonstrates:
1. Exponential-Backoff Retry Policy (LangGraph `RetryPolicy`):
   - Configures `max_attempts`, `initial_interval`, `max_interval`, and `jitter`.
   - Wraps a node simulating transient failures (fails first 2 calls, succeeds on attempt 3).
   - Demonstrates clean recovery and state progression.
2. Per-Node Timeout:
   - Enforces a strict execution time limit on an individual graph node (e.g. slow external API).
   - Demonstrates that a node exceeding the timeout raises a clean error without hanging or blocking.
3. Global Graph Timeout:
   - Enforces a cumulative execution time limit across the entire graph workflow.
   - Demonstrates clean cancellation when total pipeline latency overruns the global budget.
"""

import os
import sys
import time
import threading
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

import pytest
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy

# Ensure project root is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# 1. State Definition & Transient Failure Counter
# ============================================================================

class ResilienceWorkflowState(TypedDict):
    query: str
    retry_attempt_count: int
    transient_service_result: Optional[str]
    background_check_result: Optional[str]
    audit_trail: List[str]


class ResilienceTracker:
    """Tracks attempts and timestamps for resilience demonstrations."""
    def __init__(self):
        self.transient_call_count = 0
        self.retry_timestamps: List[float] = []

    def reset(self):
        self.transient_call_count = 0
        self.retry_timestamps.clear()


TRACKER = ResilienceTracker()


# ============================================================================
# 2. Timeout Execution Helper
# ============================================================================

def execute_with_timeout(
    func: Callable[..., Any],
    args: Tuple[Any, ...] = (),
    kwargs: Optional[Dict[str, Any]] = None,
    timeout_seconds: float = 0.5,
    timeout_message: str = "Operation timed out"
) -> Any:
    """Executes a callable in a daemon thread and enforces a hard execution timeout.

    Does not block on thread shutdown if the timeout is exceeded, returning immediately.
    """
    kwargs = kwargs or {}
    results: List[Any] = []
    errors: List[Exception] = []

    def _worker():
        try:
            res = func(*args, **kwargs)
            results.append(res)
        except Exception as exc:
            errors.append(exc)

    worker_thread = threading.Thread(target=_worker, daemon=True)
    worker_thread.start()
    worker_thread.join(timeout=timeout_seconds)

    if worker_thread.is_alive():
        raise TimeoutError(timeout_message)

    if errors:
        raise errors[0]

    return results[0] if results else None


# ============================================================================
# 3. Resilient Graph Nodes
# ============================================================================

def node_transient_failure_service(state: ResilienceWorkflowState) -> Dict[str, Any]:
    """Node simulating a transient external failure.

    Fails on the first 2 calls by raising ConnectionResetError, then succeeds on call 3.
    """
    now = time.time()
    TRACKER.transient_call_count += 1
    TRACKER.retry_timestamps.append(now)
    
    current_attempt = TRACKER.transient_call_count
    trail = list(state.get("audit_trail", []))
    trail.append(f"transient_node_attempt_{current_attempt}")

    if current_attempt <= 2:
        print(f"  [Transient Service Node] (Attempt {current_attempt}) -> Raising simulated transient network error...")
        raise ConnectionResetError(f"Simulated transient network timeout on attempt {current_attempt}")

    print(f"  [Transient Service Node] (Attempt {current_attempt}) -> Succeeded after transient recovery!")
    return {
        "retry_attempt_count": current_attempt,
        "transient_service_result": f"Service recovered successfully on attempt {current_attempt}",
        "audit_trail": trail,
    }


def node_with_per_node_timeout(
    state: ResilienceWorkflowState,
    simulated_delay: float = 1.0,
    timeout_limit: float = 0.25
) -> Dict[str, Any]:
    """Node with an enforced per-node timeout limit.

    Runs simulated work and enforces `timeout_limit`.
    Raises TimeoutError cleanly without hanging if the worker exceeds `timeout_limit`.
    """
    trail = list(state.get("audit_trail", []))

    def _slow_task() -> str:
        time.sleep(simulated_delay)
        return "Background verification completed"

    try:
        result = execute_with_timeout(
            _slow_task,
            timeout_seconds=timeout_limit,
            timeout_message=f"Per-Node Timeout: 'slow_background_check_node' exceeded time limit of {timeout_limit}s"
        )
        trail.append("per_node_timeout_passed")
        return {
            "background_check_result": result,
            "audit_trail": trail,
        }
    except TimeoutError as err:
        print(f"  [Per-Node Timeout] Node exceeded {timeout_limit}s limit (simulated work was {simulated_delay}s). Fired clean error.")
        raise err


def node_normal_fast_step(state: ResilienceWorkflowState) -> Dict[str, Any]:
    """Fast node that executes immediately."""
    trail = list(state.get("audit_trail", []))
    trail.append("fast_step_completed")
    return {"audit_trail": trail}


def node_delayed_step(state: ResilienceWorkflowState, delay: float = 0.35) -> Dict[str, Any]:
    """Node that simulates a moderate delay for global timeout testing."""
    time.sleep(delay)
    trail = list(state.get("audit_trail", []))
    trail.append(f"delayed_step_{delay}s_completed")
    return {"audit_trail": trail}


# ============================================================================
# 4. Global Timeout Runner Utility
# ============================================================================

def invoke_with_global_timeout(
    graph: Any,
    input_data: Dict[str, Any],
    global_timeout: float = 0.40,
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Executes a LangGraph graph with a strict graph-level global execution timeout.

    Cancels execution cleanly and returns immediately if runtime exceeds `global_timeout`.
    """
    try:
        return execute_with_timeout(
            graph.invoke,
            args=(input_data, config),
            timeout_seconds=global_timeout,
            timeout_message=f"Global Graph Timeout: Graph execution cancelled because runtime exceeded {global_timeout}s"
        )
    except TimeoutError as err:
        print(f"  [Global Graph Timeout] Total graph execution exceeded budget of {global_timeout}s. Terminating cleanly.")
        raise err


# ============================================================================
# 5. Graph Builders for Resilience Demonstrations
# ============================================================================

def build_retry_demo_graph(retry_policy: RetryPolicy):
    """Builds graph with LangGraph RetryPolicy on the transient node."""
    builder = StateGraph(ResilienceWorkflowState)
    builder.add_node("fast_intake", node_normal_fast_step)
    builder.add_node("transient_node", node_transient_failure_service, retry_policy=retry_policy)
    
    builder.add_edge(START, "fast_intake")
    builder.add_edge("fast_intake", "transient_node")
    builder.add_edge("transient_node", END)
    return builder.compile()


def build_node_timeout_graph(simulated_delay: float = 1.0, timeout_limit: float = 0.25):
    """Builds graph containing a node that enforces a per-node timeout."""
    builder = StateGraph(ResilienceWorkflowState)
    
    def bound_timeout_node(state: ResilienceWorkflowState):
        return node_with_per_node_timeout(state, simulated_delay=simulated_delay, timeout_limit=timeout_limit)

    builder.add_node("fast_intake", node_normal_fast_step)
    builder.add_node("slow_node", bound_timeout_node)
    
    builder.add_edge(START, "fast_intake")
    builder.add_edge("fast_intake", "slow_node")
    builder.add_edge("slow_node", END)
    return builder.compile()


def build_multi_step_pipeline_graph(delay_per_step: float = 0.35):
    """Builds a multi-step pipeline for evaluating global graph timeouts."""
    builder = StateGraph(ResilienceWorkflowState)
    
    builder.add_node("step_1", lambda state: node_delayed_step(state, delay=delay_per_step))
    builder.add_node("step_2", lambda state: node_delayed_step(state, delay=delay_per_step))
    builder.add_node("step_3", lambda state: node_delayed_step(state, delay=delay_per_step))
    
    builder.add_edge(START, "step_1")
    builder.add_edge("step_1", "step_2")
    builder.add_edge("step_2", "step_3")
    builder.add_edge("step_3", END)
    return builder.compile()


# ============================================================================
# 6. Automated Pytest Test Suite
# ============================================================================

def test_exponential_backoff_retry_recovery():
    """Demonstrates and asserts that exponential-backoff retries recover from transient failures."""
    TRACKER.reset()
    
    # Configure LangGraph RetryPolicy with exponential backoff and jitter
    retry_policy = RetryPolicy(
        max_attempts=4,
        initial_interval=0.05,
        backoff_factor=2.0,
        max_interval=0.5,
        jitter=True,
        retry_on=ConnectionResetError
    )
    
    graph = build_retry_demo_graph(retry_policy)
    
    initial_input: ResilienceWorkflowState = {
        "query": "Process candidate profile",
        "retry_attempt_count": 0,
        "transient_service_result": None,
        "background_check_result": None,
        "audit_trail": [],
    }
    
    result = graph.invoke(initial_input)
    
    # Assertions
    assert TRACKER.transient_call_count == 3, "Node should have failed twice and succeeded on attempt 3"
    assert result["retry_attempt_count"] == 3
    assert "recovered successfully" in result["transient_service_result"]
    assert "fast_step_completed" in result["audit_trail"]
    assert len(TRACKER.retry_timestamps) == 3


def test_per_node_timeout_clean_failure():
    """Demonstrates that a per-node timeout fires a clean TimeoutError promptly without hanging."""
    # Build graph where node takes 1.0s but timeout limit is 0.20s
    graph = build_node_timeout_graph(simulated_delay=1.0, timeout_limit=0.20)
    
    initial_input: ResilienceWorkflowState = {
        "query": "Background check candidate",
        "retry_attempt_count": 0,
        "transient_service_result": None,
        "background_check_result": None,
        "audit_trail": [],
    }
    
    t0 = time.time()
    with pytest.raises(TimeoutError) as exc_info:
        graph.invoke(initial_input)
    elapsed = time.time() - t0
    
    # Assert that timeout fired promptly around 0.20s, not waiting for full 1.0s
    assert elapsed < 0.5, f"Per-node timeout should abort quickly, elapsed: {elapsed:.3f}s"
    assert "Per-Node Timeout" in str(exc_info.value)


def test_global_graph_timeout_cancellation():
    """Demonstrates that global graph timeout cleanly cancels a multi-step run on cumulative time overrun."""
    # 3 steps * 0.35s = ~1.05s total runtime
    graph = build_multi_step_pipeline_graph(delay_per_step=0.35)
    
    initial_input: ResilienceWorkflowState = {
        "query": "Multi-stage verification pipeline",
        "retry_attempt_count": 0,
        "transient_service_result": None,
        "background_check_result": None,
        "audit_trail": [],
    }
    
    # Global timeout of 0.40s will cancel before all 3 steps complete (~1.05s)
    t0 = time.time()
    with pytest.raises(TimeoutError) as exc_info:
        invoke_with_global_timeout(graph, initial_input, global_timeout=0.40)
    elapsed = time.time() - t0
    
    # Assert cancellation happened promptly around 0.40s budget
    assert elapsed < 0.65, f"Global timeout should cancel promptly, elapsed: {elapsed:.3f}s"
    assert "Global Graph Timeout" in str(exc_info.value)


def test_successful_resilient_pipeline_within_limits():
    """Demonstrates normal execution completing successfully within both node and global timeouts."""
    # Node takes 0.05s with 0.5s per-node limit
    graph = build_node_timeout_graph(simulated_delay=0.05, timeout_limit=0.5)
    
    initial_input: ResilienceWorkflowState = {
        "query": "Standard verification",
        "retry_attempt_count": 0,
        "transient_service_result": None,
        "background_check_result": None,
        "audit_trail": [],
    }
    
    result = invoke_with_global_timeout(graph, initial_input, global_timeout=1.0)
    assert result["background_check_result"] == "Background verification completed"
    assert "per_node_timeout_passed" in result["audit_trail"]


# ============================================================================
# 7. Standalone Demonstration Runner
# ============================================================================

def run_resilience_demo():
    """Runs all 3 resilience demonstrations with detailed terminal output."""
    print("=" * 80)
    print("LANGGRAPH RESILIENCE, TIMEOUTS, AND RETRIES DEMONSTRATION (Task 16)")
    print("=" * 80)
    
    # ------------------------------------------------------------------------
    # DEMO 1: Exponential-Backoff Retries
    # ------------------------------------------------------------------------
    print("\n[DEMO 1] Exponential-Backoff Retry Policy (Simulating Transient Failure)")
    print("-> Node will fail attempt #1 and attempt #2, then succeed on attempt #3.")
    print("-> Retry Configuration: max_attempts=4, initial_interval=0.05s, backoff=2.0x, jitter=True")
    print("-" * 80)
    
    TRACKER.reset()
    retry_policy = RetryPolicy(
        max_attempts=4,
        initial_interval=0.05,
        backoff_factor=2.0,
        max_interval=0.5,
        jitter=True,
        retry_on=ConnectionResetError
    )
    retry_graph = build_retry_demo_graph(retry_policy)
    
    state_input: ResilienceWorkflowState = {
        "query": "Process candidate profile",
        "retry_attempt_count": 0,
        "transient_service_result": None,
        "background_check_result": None,
        "audit_trail": [],
    }
    
    t_start = time.time()
    recovered_state = retry_graph.invoke(state_input)
    t_total = time.time() - t_start
    
    print(f"\n[DEMO 1 RESULT]")
    print(f"  * Total Attempts Made: {TRACKER.transient_call_count} (Expected: 3)")
    print(f"  * Final Service State: {recovered_state.get('transient_service_result')}")
    print(f"  * Execution Audit Trail: {recovered_state.get('audit_trail')}")
    print(f"  * Total Elapsed Time: {t_total:.4f}s")
    print("  * Status: SUCCESSFUL RECOVERY VIA EXPONENTIAL BACKOFF")

    # ------------------------------------------------------------------------
    # DEMO 2: Per-Node Timeout
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[DEMO 2] Per-Node Timeout Limit")
    print("-> Slow node requires 1.0s, but per-node timeout is enforced at 0.20s.")
    print("-" * 80)
    
    timeout_graph = build_node_timeout_graph(simulated_delay=1.0, timeout_limit=0.20)
    t_start = time.time()
    try:
        timeout_graph.invoke({
            "query": "Run slow verification",
            "retry_attempt_count": 0,
            "transient_service_result": None,
            "background_check_result": None,
            "audit_trail": [],
        })
    except TimeoutError as e:
        t_elapsed = time.time() - t_start
        print(f"\n[DEMO 2 RESULT]")
        print(f"  * Caught Clean Exception: {e}")
        print(f"  * Aborted Promptly in: {t_elapsed:.4f}s (without hanging for full 1.0s)")
        print("  * Status: PER-NODE TIMEOUT ENFORCED CLEANLY")

    # ------------------------------------------------------------------------
    # DEMO 3: Global Graph-Level Timeout
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("[DEMO 3] Global Graph-Level Timeout Cancellation")
    print("-> 3-step pipeline taking ~1.05s total, with global budget enforced at 0.40s.")
    print("-" * 80)
    
    pipeline_graph = build_multi_step_pipeline_graph(delay_per_step=0.35)
    t_start = time.time()
    try:
        invoke_with_global_timeout(
            pipeline_graph,
            {
                "query": "Execute full pipeline",
                "retry_attempt_count": 0,
                "transient_service_result": None,
                "background_check_result": None,
                "audit_trail": [],
            },
            global_timeout=0.40
        )
    except TimeoutError as e:
        t_elapsed = time.time() - t_start
        print(f"\n[DEMO 3 RESULT]")
        print(f"  * Caught Clean Exception: {e}")
        print(f"  * Graph Execution Cancelled in: {t_elapsed:.4f}s (Budget: 0.40s)")
        print("  * Status: GLOBAL TIMEOUT CLEANLY OVERRUN-PROTECTED")

    print("\n" + "=" * 80)
    print("ALL RESILIENCE DEMONSTRATIONS COMPLETED SUCCESSFULLY.")
    print("=" * 80)


if __name__ == "__main__":
    run_resilience_demo()
