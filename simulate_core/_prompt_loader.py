"""Eager-load prompt text files at import time.

Reads prompts/simulator/*.txt into module-level constants. All multi-KB
system prompts live as plain text under prompts/; this loader makes them
importable as Python constants matching the legacy literal names.

Behavior:
  - rstrip() trailing whitespace (preserves intent of `\"\"\".rstrip()`
    pattern used in original Python literals).
  - Raise FileNotFoundError on missing file (fail-fast at import).
  - Read once at import; modules consuming these constants treat them
    as immutable.
"""
from pathlib import Path

_PROMPTS_ROOT = Path(__file__).parent / "prompts"


def _load(category: str, name: str) -> str:
    """Read prompts/<category>/<name>.txt; rstrip trailing whitespace."""
    path = _PROMPTS_ROOT / category / f"{name}.txt"
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    return path.read_text().rstrip()


# ---- Simulator prompts ----
# FALLBACK  — tool_return_mode=="httpx" (caller-only training); remote API is
#             told to emit {"analysis": ..., "execution_result": ...} raw JSON.
# VALIDATION — tool_return_mode=="agent" (share / noshare training); the local
#             Tool Simulator Agent runs the 4-step whitelist/schema validation
#             and wraps its output in <execution_result> tags.
FALLBACK_SYSTEM_INSTRUCTION = _load("simulator", "fallback_system_instruction")
VALIDATION_SIMULATOR_SYSTEM_INSTRUCTION = _load(
    "simulator", "validation_system_instruction"
)
