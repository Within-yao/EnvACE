"""simulate_core — simulate protocol primitives for the checklist env.

Public API (re-exported by
agent_system/environments/env_package/checklist/mcp_tool.py):
  - Prompts:  FALLBACK_SYSTEM_INSTRUCTION      (tool_return_mode=="httpx")
              VALIDATION_SIMULATOR_SYSTEM_INSTRUCTION (tool_return_mode=="agent")
  - Builders: build_validation_user_prompt, build_few_shot_examples_lines,
              build_single_call_user_prompt, serialize_conversation_history
  - Errors:   format_tool_error

Stdlib-only. Import requires the repo root on sys.path (satisfied by running
`python -m verl.trainer.main_ppo` from the repo root).
"""
from simulate_core._prompt_loader import (
    FALLBACK_SYSTEM_INSTRUCTION,
    VALIDATION_SIMULATOR_SYSTEM_INSTRUCTION,
)
from simulate_core._builders import (
    build_validation_user_prompt,
    build_few_shot_examples_lines,
    build_single_call_user_prompt,
    serialize_conversation_history,
)
from simulate_core._error import format_tool_error

__all__ = [
    "FALLBACK_SYSTEM_INSTRUCTION",
    "VALIDATION_SIMULATOR_SYSTEM_INSTRUCTION",
    "build_validation_user_prompt",
    "build_few_shot_examples_lines",
    "build_single_call_user_prompt",
    "serialize_conversation_history",
    "format_tool_error",
]
