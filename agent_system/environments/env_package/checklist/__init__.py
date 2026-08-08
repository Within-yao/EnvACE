"""Checklist tool-use environment (ported from upstream tool-use reference).

Exposes:
    build_checklist_envs, checklist_projection  — for env_manager.make_envs
    MCPChecklistTool, ChecklistInteraction       — shared singletons inside batched env
    parse_tool_calls, FunctionCall               — tool-call parsing
    checklist_reward                             — sglang LLM-as-Judge primitives
"""

from . import checklist_reward
from .interaction import ChecklistInteraction
from .mcp_tool import MCPChecklistTool, ToolResponse, NEED_SIMULATOR
from .tool_parser import parse_tool_calls, FunctionCall

def build_checklist_envs(*args, **kwargs):
    from .envs import build_checklist_envs as _impl
    return _impl(*args, **kwargs)


def checklist_projection(*args, **kwargs):
    from .projection import checklist_projection as _impl
    return _impl(*args, **kwargs)


__all__ = [
    "build_checklist_envs",
    "checklist_projection",
    "ChecklistInteraction",
    "MCPChecklistTool",
    "ToolResponse",
    "NEED_SIMULATOR",
    "parse_tool_calls",
    "FunctionCall",
    "checklist_reward",
]
