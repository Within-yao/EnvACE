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


FALLBACK_SYSTEM_INSTRUCTION = _load("simulator", "fallback_system_instruction")
VALIDATION_SIMULATOR_SYSTEM_INSTRUCTION = _load(
    "simulator", "validation_system_instruction"
)
