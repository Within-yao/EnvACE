# Tool-call parser for the Checklist env.
#
# Strictly follows upstream-1107 — strict-parse +
# structured error semantics. Any malformed tool_call returns an `error` dict
# with a byte-equivalent payload that the env will JSON-dump into a tool
# message before terminating the trajectory.

import json
import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

TOOL_CALL_START = "<tool_call>"
BOT_TOKEN = "\n<tool_call>\n"
EOT_TOKEN = "\n</tool_call>"
_TOOL_CALL_RE = re.compile(
    rf"{re.escape(BOT_TOKEN)}(.*?){re.escape(EOT_TOKEN)}", re.DOTALL
)


@dataclass
class FunctionCall:
    name: str
    arguments: str


def parse_tool_calls(
    text: str,
) -> Tuple[str, List[FunctionCall], Optional[Dict]]:
    """Strict tool-call parsing aligned with upstream sglang_rollout.py:1019-1107.

    Returns:
        (content, function_calls, error)
        - content: raw `text` (upstream line 1097 passes original content to add_assistant_message).
        - function_calls: list of validated FunctionCall (empty on any error).
        - error: None on success; dict {code, payload} on failure.
            * code: one of NO_TOOL_CALL_FOUND, JSON_DECODE_ERROR, INVALID_TOOL_NAME,
                    NO_ARGUMENTS, NON_SERIALIZABLE_ARGUMENTS.
            * payload: dict with `error_tool_call` key, byte-equivalent to what upstream
                       wraps in ToolResponse(text=json.dumps(payload)).

    Break-then-clear semantics: a single bad block invalidates the whole batch
    (upstream line 1042-1048 + 1053-1055).
    """
    if TOOL_CALL_START not in text:
        return text, [], None

    matches = _TOOL_CALL_RE.findall(text)
    if not matches:
        return text, [], {
            "code": "NO_TOOL_CALL_FOUND",
            "payload": {
                "error_tool_call": (
                    "No tool call found. Please check your output format. "
                    "The correct (<tool_call>\n{\"name\": ..., \"arguments\": { ... }}\n</tool_call>)."
                ),
            },
        }

    function_calls: List[FunctionCall] = []
    for match_result in matches:
        stripped = match_result.strip()

        try:
            parsed_call = json.loads(stripped)
        except json.JSONDecodeError:
            return text, [], {
                "code": "JSON_DECODE_ERROR",
                "payload": {
                    "error_tool_call": f"One tool call can not be parsed as JSON: {stripped}",
                },
            }

        actions = parsed_call if isinstance(parsed_call, list) else [parsed_call]

        for act in actions:
            name = act.get("name") if isinstance(act, dict) else None
            arguments = act.get("parameters") if isinstance(act, dict) else None
            if arguments is None and isinstance(act, dict):
                arguments = act.get("arguments", None)

            if arguments is None:
                return text, [], {
                    "code": "NO_ARGUMENTS",
                    "payload": {
                        "error_tool_call": "No argumets found in one tool call. Use empty dict if no argument.",
                    },
                }

            try:
                arguments_json = json.dumps(arguments, ensure_ascii=False)
            except Exception as e:
                return text, [], {
                    "code": "NON_SERIALIZABLE_ARGUMENTS",
                    "payload": {
                        "error_tool_call": "NON_SERIALIZABLE_ARGUMENTS in one tool call",
                        "detail": str(e),
                    },
                }
            from .mcp_tool import MCPChecklistTool as _MCP
            arguments_json = _MCP._sanitize_text_for_tokenizer(arguments_json)

            if not isinstance(name, str) or not name.strip():
                return text, [], {
                    "code": "INVALID_TOOL_NAME",
                    "payload": {
                        "error_tool_call": "INVALID_TOOL_NAME in one tool call",
                        "raw_name": str(name),
                        "arguments": str(arguments),
                    },
                }

            function_calls.append(FunctionCall(name=name.strip(), arguments=arguments_json))

    return text, function_calls, None
