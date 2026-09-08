import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Base directory path for loading dataset
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "job_application.json"


def load_application_data(data_path: Optional[str | Path] = None) -> List[Dict[str, Any]]:
    """Loads the job applications dataset from job_application.json.

    Parameters
    ----------
    data_path : Optional[str | Path]
        Custom path to the JSON file. Defaults to data/job_application.json.

    Returns
    -------
    List[Dict[str, Any]]
        List of application records.
    """
    target_path = Path(data_path) if data_path else DATA_FILE

    if target_path.exists() and target_path.is_file():
        try:
            with open(target_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from {target_path}: {e}")
            return []

    print(f"Warning: Job application dataset not found at {target_path}")
    return []


def calculate_escalation_score(
    flagged_priority_review: bool,
    days_since_created: int | float,
    max_recency_days: float = 30.0
) -> float:
    """Calculates the escalation score for a candidate's job application.

    Mathematical Formula:
    ---------------------
    escalation_score = (w_flag * I_flag) + (w_recency * normalized_recency)

    Where:
      - w_flag = 0.5 (Weight assigned to priority review flag)
      - I_flag = 1.0 if flagged_priority_review is True else 0.0
      - w_recency = 0.5 (Weight assigned to recency signal)
      - normalized_recency = clamp(days_since_created / max_recency_days, 0.0, 1.0)
      - max_recency_days = 30.0 (Normalization cap)

    Formula Expansion:
      escalation_score = (0.5 * (1.0 if flagged_priority_review else 0.0)) + 
                         (0.5 * min(max(days_since_created, 0.0) / 30.0, 1.0))

    Final Score Range:
      - escalation_score in [0.0, 1.0]

    Recommended Priority Thresholds:
    --------------------------------
      - Score >= 0.70 : HIGH Priority Review (e.g. flagged priority candidate pending for >= 12 days,
                        or long-standing application needing urgent escalation)
      - 0.40 <= Score < 0.70 : MEDIUM Priority Review (standard candidate follow-up queue)
      - Score < 0.40 : LOW Priority / Routine Application

    Parameters
    ----------
    flagged_priority_review : bool
        Whether the application was flagged for priority review.
    days_since_created : int | float
        Number of days elapsed since the application was created.
    max_recency_days : float, default=30.0
        The maximum normalization factor for days elapsed.

    Returns
    -------
    float
        Escalation score bounded between 0.0 and 1.0, rounded to 4 decimal places.
    """
    flag_weight = 0.5 if bool(flagged_priority_review) else 0.0
    safe_days = max(float(days_since_created), 0.0)
    recency_signal = min(safe_days / max_recency_days, 1.0)
    recency_weight = 0.5 * recency_signal

    score = flag_weight + recency_weight
    # Ensure strict clamping between 0.0 and 1.0
    clamped_score = max(0.0, min(1.0, score))
    return round(clamped_score, 4)


def check_job_application_status(record_id: str, data_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Retrieves application status, expected salary, and computed escalation score for a candidate.

    Searches the job application dataset for a specific record_id and computes
    the composite escalation_score to prioritize recruiter review queues.

    Mathematical Formula:
    ---------------------
    escalation_score = (0.5 * (1.0 if flagged_priority_review else 0.0)) +
                       (0.5 * min(days_since_created / 30.0, 1.0))

    Recommended Threshold:
    ----------------------
    - escalation_score >= 0.70: High priority review candidates. Requires prompt recruiter attention.
    - escalation_score < 0.70: Standard processing pipeline.

    Parameters
    ----------
    record_id : str
        The unique identifier for the application (e.g., "REC-001").
    data_path : Optional[str | Path]
        Optional path to the dataset file if non-default.

    Returns
    -------
    Dict[str, Any]
        If found:
            {
                "record_id": str,
                "status": str,
                "expected_salary_inr": int,
                "escalation_score": float,
                "category": str,
                "days_since_created": int,
                "flagged_priority_review": bool
            }
        If not found:
            {
                "error": str,
                "status": "Not Found",
                "record_id": str
            }
    """
    if not record_id or not isinstance(record_id, str):
        return {
            "error": "Invalid record_id provided. Must be a non-empty string.",
            "status": "Not Found",
            "record_id": str(record_id) if record_id is not None else ""
        }

    dataset = load_application_data(data_path)
    if not dataset:
        return {
            "error": "Application database is empty or unavailable.",
            "status": "Not Found",
            "record_id": record_id
        }

    target_id = record_id.strip().upper()
    for record in dataset:
        curr_id = str(record.get("record_id", "")).strip().upper()
        if curr_id == target_id:
            flagged = bool(record.get("flagged_priority_review", False))
            days = int(record.get("days_since_created", 0))
            escalation_score = calculate_escalation_score(flagged, days)

            return {
                "record_id": record.get("record_id", record_id),
                "status": str(record.get("status", "Unknown")),
                "expected_salary_inr": int(record.get("expected_salary_inr", 0)),
                "escalation_score": escalation_score,
                "category": record.get("category", "Unknown"),
                "days_since_created": days,
                "flagged_priority_review": flagged
            }

    return {
        "error": f"Application record '{record_id}' not found.",
        "status": "Not Found",
        "record_id": record_id
    }


# Optional LangChain tool wrapper for agent integration
try:
    from langchain_core.tools import tool

    @tool
    def check_job_application_status_tool(record_id: str) -> Dict[str, Any]:
        """Search job applications database for a candidate record_id and return status, salary, and escalation score."""
        return check_job_application_status(record_id)
except ImportError:
    check_job_application_status_tool = None


if __name__ == "__main__":
    print("Testing check_job_application_status tool:")
    print("-" * 50)
    
    # Test valid record
    rec_001 = check_job_application_status("REC-001")
    print(f"REC-001 Lookup: {rec_001}")
    
    # Test another record
    rec_005 = check_job_application_status("REC-005")
    print(f"REC-005 Lookup: {rec_005}")

    # Test case sensitivity / whitespace trimming
    rec_lower = check_job_application_status("  rec-002  ")
    print(f"REC-002 (lowercase & padded) Lookup: {rec_lower}")

    # Test non-existent record
    rec_invalid = check_job_application_status("REC-999")
    print(f"REC-999 (Invalid) Lookup: {rec_invalid}")

    # Test empty / malformed record ID
    rec_empty = check_job_application_status("")
    print(f"Empty Record Lookup: {rec_empty}")

    # Test escalation score calculations
    print("\nEscalation Score Unit Tests:")
    print(f"  Flagged=False, Days=0  -> Score: {calculate_escalation_score(False, 0)} (Expected: 0.0)")
    print(f"  Flagged=False, Days=30 -> Score: {calculate_escalation_score(False, 30)} (Expected: 0.5)")
    print(f"  Flagged=True,  Days=0  -> Score: {calculate_escalation_score(True, 0)} (Expected: 0.5)")
    print(f"  Flagged=True,  Days=15 -> Score: {calculate_escalation_score(True, 15)} (Expected: 0.75)")
    print(f"  Flagged=True,  Days=30 -> Score: {calculate_escalation_score(True, 30)} (Expected: 1.0)")
    print(f"  Flagged=True,  Days=45 -> Score: {calculate_escalation_score(True, 45)} (Expected: 1.0, clamped)")
