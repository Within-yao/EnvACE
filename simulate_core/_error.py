"""Unified tool-execution error template.

Used by all tool paths (simulator / cache / schema / httpx fallback) to
emit format-consistent errors that downstream envs detect via the
`"error_tool_call" in parsed` check.

Output JSON shape:
    {"error_tool_call": {"code": "X", "message": "Y", "details": {...}}}
"""
import json
from typing import Any, Dict, Optional


def format_tool_error(
    code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
) -> str:
    payload: Dict[str, Any] = {
        "error_tool_call": {
            "code": code,
            "message": message,
        }
    }
    if details is not None:
        payload["error_tool_call"]["details"] = details
    return json.dumps(payload, ensure_ascii=True)
