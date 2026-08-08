"""Prompt builders + conversation history serializer.

These functions assemble per-call user prompts for the simulator. The
*system* prompts are loaded separately (see _prompt_loader.py).

Shared by:
- training-time httpx fallback (mcp_tool.py via re-export shim)
- training-time Tool Simulator Agent
- evaluation-time tau2 / BFCL SimulateRouter subclasses
"""
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _append_think_enabled() -> bool:
    return os.environ.get("SIM_APPEND_THINK_SWITCH", "true").lower() not in (
        "false", "0", "no", "off", ""
    )


def build_few_shot_examples_lines(
    tool_call_response_map: Dict[str, List[Tuple[Dict, Any]]],
    sanitize_fn=None,
) -> List[str]:
    """Build the few-shot block (list of lines) from a sample's
    `tool_call_response_map = {tool_name: [(args_dict, content), ...]}`.

    Empty maps and exception failures both degrade to the single-line form
    `['Previous tool calls and results (few-shot): <unavailable>']` so that
    downstream prompts carry an explicit "no prior calls" signal instead of a
    bare header followed by a blank line.
    """
    if sanitize_fn is None:
        sanitize_fn = lambda s: s
    if not tool_call_response_map:
        return ["Previous tool calls and results (few-shot): <unavailable>"]
    lines = ["Previous tool calls and results (few-shot):"]
    try:
        for ex_tool_name, pairs in tool_call_response_map.items():
            for ex_args, ex_content in pairs:
                ex_args_str = sanitize_fn(json.dumps(ex_args, ensure_ascii=True, indent=0))
                if isinstance(ex_content, (dict, list)):
                    ex_content_str = sanitize_fn(json.dumps(ex_content, ensure_ascii=True, indent=0))
                else:
                    assert isinstance(ex_content, str), f"Unexpected ex_content type: {type(ex_content)}"
                    ex_content_str = sanitize_fn(ex_content)
                lines.append(
                    "Tool name:\n" + ex_tool_name + "\n"
                    + "Tool arguments:\n" + ex_args_str + "\n"
                    + "Tool execution result:\n" + ex_content_str + "\n"
                )
    except Exception:
        logger.warning("Failed to build few-shot examples, using <unavailable> instead")
        return ["Previous tool calls and results (few-shot): <unavailable>"]
    return lines


def build_single_call_user_prompt(
    examples_lines: List[str],
    schema_str: str,
    tool_name: str,
    arguments_obj: Dict[str, Any],
    sanitize_fn=None,
    tool_description: str = "",
) -> str:
    """Assemble the user prompt for a single tool execution simulation.

    Default behavior (tool_description=""): byte-identical to upstream
    mcp_checklist_tool.py:572-585. Shared by httpx fallback (Phase 3) and
    simulator agent.

    When tool_description is non-empty (currently only build_simulator_sft.py's
    offline data generation passes it), an extra `Current tool description:`
    line is inserted between the tool name and the input schema.
    """
    if sanitize_fn is None:
        sanitize_fn = lambda s: s
    desc_line = ""
    if tool_description:
        desc_line = "Current tool description: " + tool_description.strip() + "\n"
    user_prompt = (
        "\n".join(examples_lines)
        + "\n\n"
        + "Current tool name: " + tool_name + "\n"
        + desc_line
        + "Current tool input schema (JSON Schema):\n" + schema_str + "\n"
        + "Current arguments (JSON):\n"
        + sanitize_fn(json.dumps(arguments_obj, ensure_ascii=True, indent=0))
        + "\n"
        + "Generate tool execution result in JSON format."
    )
    return sanitize_fn(user_prompt)


def serialize_conversation_history(messages: List[Dict[str, Any]]) -> str:
    """Serialize a conversation history (system/user/assistant/tool) into a flat
    text block for inclusion in the simulator's `Current conversation history:`
    section.

    Mirrors the role-prefixed format from Simia-RL `_format_history_for_prompt`:
    - role=user        → 'User: <content>'
    - role=assistant   → 'Assistant: <content>' (with `<tool_call>{...}</tool_call>`
                          blocks rendered inline if `tool_calls` field is present)
    - role=tool        → 'Tool[<name>]: <content>'
    - role=system      → skipped (system_prompt is injected as a separate section)
    Empty / unknown roles are skipped.
    """
    if not messages:
        return "No conversation history"
    lines: List[str] = []
    for msg in messages:
        role = msg.get("role", "")
        content = (msg.get("content", "") or "").strip()
        if role == "system":
            continue
        if role == "user":
            if content:
                lines.append(f"User: {content}")
        elif role == "assistant":
            assistant_text = content
            tcs = msg.get("tool_calls") or []
            if tcs:
                for tc in tcs:
                    fn = tc.get("function", {}) if isinstance(tc, dict) else {}
                    nm = fn.get("name", "")
                    args = fn.get("arguments", "")
                    if not isinstance(args, str):
                        try:
                            args = json.dumps(args, ensure_ascii=True)
                        except Exception:
                            args = str(args)
                    assistant_text += f"\n<tool_call>\n{{\"name\": \"{nm}\", \"arguments\": {args}}}\n</tool_call>"
            if assistant_text:
                lines.append(f"Assistant: {assistant_text}")
        elif role in ("tool", "observation"):
            tool_name = msg.get("name", "")
            label = f"Tool[{tool_name}]" if tool_name else "Tool"
            if content:
                lines.append(f"{label}: {content}")
    return "\n\n".join(lines) if lines else "No conversation history"


def build_validation_user_prompt(
    examples_lines: List[str],
    schema_str: str,
    tool_name: str,
    arguments_obj: Dict[str, Any],
    *,
    candidate_tools: Optional[List[Dict[str, Any]]] = None,
    conversation_history: Optional[str] = None,
    raw_caller_message: Optional[str] = None,
    sanitize_fn=None,
) -> str:
    """Validation-mode simulator USER prompt — per-call DATA only.

    All persistent rules (validation procedure, error_tool_call JSON contract,
    output XML format) live in `VALIDATION_SIMULATOR_SYSTEM_INSTRUCTION` so the
    system prompt is stable across rollouts and benefits from sglang prefix
    cache. This builder emits only the per-call sections that DO change between
    samples: candidate_tools, few-shot, history, raw caller text, current
    name/schema/args.

    Section ordering mirrors Simia-RL `_generate_next_environment_message`,
    minus the `System prompt (task description and rules):` section.

    Each optional section is opt-in: pass None / empty to skip.
    """
    if sanitize_fn is None:
        sanitize_fn = lambda s: s

    parts: List[str] = []

    if candidate_tools:
        try:
            ct_text = json.dumps(candidate_tools, ensure_ascii=True, indent=0)
        except Exception:
            ct_text = str(candidate_tools)
        parts.append(
            "Candidate tools (whitelist for this sample):\n"
            + sanitize_fn(ct_text)
        )

    parts.append(sanitize_fn("\n".join(examples_lines)))

    if conversation_history:
        parts.append(
            "Current conversation history:\n" + sanitize_fn(conversation_history)
        )

    if raw_caller_message:
        parts.append(
            "RL model's latest response:\n" + sanitize_fn(raw_caller_message)
        )

    parts.append(
        "Current tool to simulate:\n"
        f"- name: {tool_name}\n"
        "- input schema (JSON Schema):\n" + schema_str + "\n"
        "- arguments (JSON, parsed from caller's tool_call):\n"
        + sanitize_fn(json.dumps(arguments_obj, ensure_ascii=True, indent=0))
    )

    full_prompt = sanitize_fn("\n\n".join(parts))

    if _append_think_enabled():
        full_prompt = full_prompt + "\n\n/think"

    return full_prompt
