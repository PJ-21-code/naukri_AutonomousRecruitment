import argparse
import asyncio
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error

from fastmcp import Client

# Ensure project root is in sys.path
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mcp_server import app as mcp_asgi_app, mcp as mcp_instance


def is_server_alive(endpoint: str, timeout: float = 1.0) -> bool:
    """Checks whether the MCP HTTP endpoint is reachable."""
    try:
        req = urllib.request.Request(endpoint, headers={"User-Agent": "FastMCP-Client"})
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status in (200, 404, 405, 400)
    except urllib.error.HTTPError:
        # HTTP error (e.g. 405 Method Not Allowed or 400 on GET) means the server is active
        return True
    except Exception:
        return False


def start_local_server_thread(host: str = "127.0.0.1", port: int = 8000) -> threading.Thread:
    """Spawns an in-process Uvicorn server thread hosting the MCP ASGI app for standalone demos."""
    import uvicorn

    server = uvicorn.Server(
        config=uvicorn.Config(
            app=mcp_asgi_app,
            host=host,
            port=port,
            log_level="error",
            access_log=False
        )
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    return thread


async def call_mcp_tool(
    endpoint: str,
    tool_name: str,
    arguments: Dict[str, Any]
) -> Dict[str, Any]:
    """Invokes an MCP tool on the given endpoint using the FastMCP Client.

    Parameters:
    -----------
    endpoint : str
        The HTTP MCP server endpoint (e.g. 'http://127.0.0.1:8000/mcp').
    tool_name : str
        The name of the MCP tool to call (e.g. 'check_job_application_status').
    arguments : Dict[str, Any]
        Arguments dictionary matching the tool's parameter signature.

    Returns:
    --------
    Dict[str, Any]:
        Standardized tool execution result dictionary.
    """
    async with Client(endpoint) as client:
        result = await client.call_tool(tool_name, arguments)
        
        # Extract structured content if available
        if hasattr(result, "structured_content") and result.structured_content:
            return result.structured_content
        elif hasattr(result, "data") and isinstance(result.data, dict):
            return result.data
        elif hasattr(result, "content") and result.content:
            # Parse text content JSON
            for item in result.content:
                if hasattr(item, "text"):
                    try:
                        return json.loads(item.text)
                    except json.JSONDecodeError:
                        return {"raw_text": item.text}
        
        return {"raw_result": str(result)}


async def run_client_queries(
    endpoint: str,
    record_ids: List[str]
) -> Dict[str, Dict[str, Any]]:
    """Runs status lookup queries across multiple candidate record IDs via the MCP Client.

    Parameters:
    -----------
    endpoint : str
        The HTTP MCP server endpoint URL.
    record_ids : List[str]
        List of candidate record IDs to query.

    Returns:
    --------
    Dict[str, Dict[str, Any]]:
        Mapping of record_id -> standardized tool response payload.
    """
    print(f"\n[MCP Client] Connecting to MCP Server at: {endpoint}")
    
    # List available tools on server
    async with Client(endpoint) as client:
        tools = await client.list_tools()
        tool_names = [t.name for t in tools]
        print(f"[MCP Client] Remote Server Discovered Tools: {tool_names}\n")

    results: Dict[str, Dict[str, Any]] = {}
    
    for rid in record_ids:
        print(f"--- Querying Candidate Record: {rid} ---")
        response = await call_mcp_tool(
            endpoint=endpoint,
            tool_name="check_job_application_status",
            arguments={"record_id": rid}
        )
        results[rid] = response
        
        # Format and display standardized output
        print(f"Status Result for [{rid}]:")
        print(json.dumps(response, indent=2))
        
        if "escalation_score" in response:
            score = response["escalation_score"]
            priority = "HIGH" if score >= 0.70 else ("MEDIUM" if score >= 0.40 else "LOW")
            print(f"-> Calculated Escalation Score: {score:.4f} [{priority} Priority Review Queue]")
        elif "error" in response:
            print(f"-> Lookup Notice: {response.get('error')}")
        print()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="FastMCP Client for Naukri Autonomous Recruitment")
    parser.add_argument("--endpoint", type=str, default="http://127.0.0.1:8000/mcp", help="MCP server endpoint URL")
    parser.add_argument("--records", nargs="+", default=["REC-001", "REC-005", "REC-002", "REC-999"], help="Candidate record IDs to query")
    args = parser.parse_args()

    endpoint = args.endpoint
    record_ids = args.records

    print("=" * 70)
    print("Naukri Autonomous Recruitment MCP Client")
    print("=" * 70)

    # Check if MCP server is already listening; if not, spawn local in-process server
    if not is_server_alive(endpoint):
        print(f"[*] No active MCP server detected at {endpoint}. Starting in-process demo server...")
        start_local_server_thread(host="127.0.0.1", port=8000)
        time.sleep(1.0)
    else:
        print(f"[*] Active MCP server detected at {endpoint}.")

    results = asyncio.run(run_client_queries(endpoint=endpoint, record_ids=record_ids))
    print("=" * 70)
    print(f"MCP Client finished successfully. Queried {len(results)} candidate records.")
    print("=" * 70)


if __name__ == "__main__":
    main()
