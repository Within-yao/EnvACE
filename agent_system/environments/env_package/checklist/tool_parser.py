# Tool-call parser for the Checklist env.
#
# Strictly mirrors upstream sglang_rollout.py:872-916, 1019-1107 — strict-parse +
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
    arguments: str  # JSON-encoded argument string


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
        # Final-answer turn: no tool call attempted. Mirror upstream line 1108-1110.
        return text, [], None

    matches = _TOOL_CALL_RE.findall(text)
    if not matches:
        # bot_token present but pattern didn't match (missing \n boundary or eot).
        # Mirror upstream line 1101-1107.
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

        # Step 1: JSON decode (upstream line 1040-1043)
        try:
            parsed_call = json.loads(stripped)
        except json.JSONDecodeError:
            return text, [], {
                "code": "JSON_DECODE_ERROR",
                "payload": {
                    "error_tool_call": f"One tool call can not be parsed as JSON: {stripped}",
                },
            }

        # Normalize to list (upstream parse_base_json line 873-874)
        actions = parsed_call if isinstance(parsed_call, list) else [parsed_call]

        for act in actions:
            # Step 2: extract fields. upstream line 880-883: parameters PRIORITIZED,
            # fallback to arguments.
            name = act.get("name") if isinstance(act, dict) else None
            arguments = act.get("parameters") if isinstance(act, dict) else None
            if arguments is None and isinstance(act, dict):
                arguments = act.get("arguments", None)

            # Step 3: arguments must be present (upstream line 886-888)
            if arguments is None:
                return text, [], {
                    "code": "NO_ARGUMENTS",
                    "payload": {
                        "error_tool_call": "No argumets found in one tool call. Use empty dict if no argument.",
                    },
                }

            # Step 4: arguments must be JSON-serializable (upstream line 890-894)
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
            # ensure_ascii=False preserves any lone surrogate / noncharacter that
            # was inside `arguments` (json.loads accepts surrogate escapes verbatim
            # and round-trips them through dumps). These would later crash
            # apply_chat_template via msg["tool_calls"][i].function.arguments. Run
            # the same sanitize as caller_text entry-point (envs.py:306).
            from .mcp_tool import MCPChecklistTool as _MCP
            arguments_json = _MCP._sanitize_text_for_tokenizer(arguments_json)

            # Step 5: name validation (upstream line 897-906)
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
