import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from fastmcp import FastMCP

# Ensure project root is in sys.path for standalone execution
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import domain tool
try:
    from src.tools import check_job_application_status as _check_status_impl
except ImportError:
    from tools import check_job_application_status as _check_status_impl

# 1. FastMCP Server Initialization

mcp = FastMCP(
    name="NaukriAutonomousRecruitmentServer"
)

# 2. Tool Registration & Documentation

@mcp.tool()
def check_job_application_status(record_id: str) -> Dict[str, Any]:
    """Retrieves candidate application status, salary expectations, and calculated escalation score.

    Searches the recruitment database for a specific candidate `record_id` (e.g., 'REC-001')
    and computes the mathematical escalation score to prioritize recruiter follow-ups.

    Mathematical Formula:
    ---------------------
    escalation_score = (w_flag * I_flag) + (w_recency * normalized_recency)
                     = (0.5 * (1.0 if flagged_priority_review else 0.0)) +
                       (0.5 * min(days_since_created / 30.0, 1.0))

    Score Range & Actionable Priority Thresholds:
    --------------------------------------------
    - Score in [0.0, 1.0]
    - Score >= 0.70 : HIGH Priority Review (Urgent recruiter intervention required)
    - 0.40 <= Score < 0.70 : MEDIUM Priority Review (Standard follow-up queue)
    - Score < 0.40 : LOW Priority / Routine Application

    Parameters:
    -----------
    record_id : str
        The unique alphanumeric application identifier (e.g. "REC-001", "REC-005").

    Returns:
    --------
    Dict[str, Any]:
        If candidate record is found:
            {
                "record_id": str,
                "status": str,
                "expected_salary_inr": int,
                "escalation_score": float,
                "category": str,
                "days_since_created": int,
                "flagged_priority_review": bool
            }
        If candidate record is not found or invalid:
            {
                "error": str,
                "status": "Not Found",
                "record_id": str
            }
    """
    return _check_status_impl(record_id=record_id)


# Expose ASGI application for mounting or testing
app = mcp.http_app()

# 3. Standalone Server Execution

if __name__ == "__main__":
    host = os.getenv("MCP_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_PORT", "8000"))
    print("=" * 70)
    print(f"Starting Naukri FastMCP Server on http://{host}:{port}/mcp")
    print(f"Registered Tools: check_job_application_status")
    print("=" * 70)
    
    # Run using FastMCP HTTP transport (mounted at /mcp by default)
    mcp.run(transport="http", host=host, port=port)
