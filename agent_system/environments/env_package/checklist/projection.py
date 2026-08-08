"""Projection for the Checklist env.

Given a list of raw LLM decoded texts, parse each into a dict with keys:
    content: str                     — raw assistant text (upstream line 1097 passes original)
    tool_calls: list[FunctionCall]   — parsed tool calls (empty on any error)
    error: dict | None               — structured parse error (see tool_parser.parse_tool_calls)

valids[i] is 1 if there's substantive content or tool_calls; parse errors do
NOT mark the action invalid here — alignment with upstream is via hard-terminate +
error_tool_call message in env, not via this repo's apply_invalid_action_penalty.
"""
from typing import List, Tuple
from .tool_parser import parse_tool_calls


def checklist_projection(text_actions: List[str]) -> Tuple[List[dict], List[int]]:
    results: List[dict] = []
    valids: List[int] = []
    for text in text_actions:
        if text is None:
            text = ""
        content, tool_calls, error = parse_tool_calls(text)
        results.append({"content": content, "tool_calls": tool_calls, "error": error})
        is_valid = 1 if (tool_calls or content.strip()) else 0
        valids.append(is_valid)
    return results, valids
