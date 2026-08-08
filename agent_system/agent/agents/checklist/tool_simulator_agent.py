# Copyright 2026 Nanyang Technological University (NTU), Singapore
# Licensed under the Apache License, Version 2.0.

"""Tool Simulator Agent.

Consumes per-sample `simulator_payload` (a single pending tool-call spec under
the per-call cursor design, plan §3.5 + §10.(g)) and emits — for each original
sample — a single tool-result string that ChecklistEnv writes back as the next
`{role:"tool"}` message.

Output contract (plan §10.(c)):
    <think>... CoT ...</think>
    <execution_result>
    <non-empty JSON object or array>
    </execution_result>

Parser (`_parse_sim_response`) strict-parses this:
  Phase 0: strip <think>...</think>
  Phase 1: detect '<execution_result>' substring  (mirrors caller TOOL_CALL_START)
  Phase 2: regex extract <execution_result>...</execution_result> block (mirrors _TOOL_CALL_RE)
  Phase 3: json.loads(block)
  Phase 4+5: type constraint — non-empty dict OR list (upstream response_format anyOf)
  Phase 6: serializable
  (Length truncation removed: aligned with upstream sglang_rollout.py which does NOT
   truncate single tool responses; rollout_max_prompt_length is the safety net.)

All errors go through `mcp_tool.format_tool_error(...)` nested-object format —
byte-equivalent to mcp_tool's INDEX_NOT_FOUND / SCHEMA_VALIDATION_FAILED /
INVALID_PARAMETERS errors. envs.py:265-280 detects `error_tool_call` key and
hard-terminates trajectory.

Rollout-backend exceptions are caught and emit a flat-string sentinel
mirroring upstream `tool_agent_loop._call_tool` — trajectory continues, judge
penalises via low score (matches upstream behaviour for tool dispatch failures).

Batch-size contract: returned DataProto always has size == B (one row per
original sample). Inactive samples get a dummy prompt + agent_active_mask=False.
"""
from typing import Any, Dict, List, Optional, Tuple
import json
import re

import numpy as np
import torch
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

from agent_system.agent.agents.base import BaseAgent
from agent_system.agent.registry import AgentRegistry
from simulate_core import (
    VALIDATION_SIMULATOR_SYSTEM_INSTRUCTION,
    build_validation_user_prompt,
    format_tool_error,
)


import logging

logger = logging.getLogger(__name__)



EXECUTION_RESULT_START = "<execution_result>"
EXECUTION_RESULT_BOT = "\n<execution_result>\n"
EXECUTION_RESULT_EOT = "\n</execution_result>"
_EXECUTION_RESULT_RE = re.compile(
    rf"{re.escape(EXECUTION_RESULT_BOT)}(.*?){re.escape(EXECUTION_RESULT_EOT)}",
    re.DOTALL,
)
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _parse_sim_response(text: str) -> str:
    """Strict-parse simulator output. Returns:
      success → result JSON string (json.dumps of execution_result content)
      failure → format_tool_error(...) JSON string

    Length truncation removed: upstream checklist actually runs in sglang_rollout.py
    (rollout.mode=sync, traj_n48.sh:91), where single-tool-response truncation
    does NOT happen. The previous 256 char middle-truncate was borrowed from
    upstream tool_agent_loop.py:457-464 — but that path is never reached in upstream
    checklist (tool_agent_loop only fires when rollout.mode=async). To stay
    aligned with upstream real production behavior, we now keep the simulator
    output verbatim and rely on caller-side rollout_max_prompt_length=12000
    left-truncation as the global safety net.

    See plan §10.(c) for full design rationale + scenario coverage table."""

    if text:
        text = _THINK_BLOCK_RE.sub("", text)
        if not text.strip():
            text = ""

    if not text:
        return format_tool_error(
            code="EMPTY_SIMULATOR_OUTPUT",
            message="Empty simulator output",
            details={"expected": "<execution_result>...</execution_result> XML block"},
        )
    if EXECUTION_RESULT_START not in text:
        return format_tool_error(
            code="NO_EXECUTION_RESULT_TAG_FOUND",
            message="No '<execution_result>' tag found in simulator output",
            details={
                "expected_format": (
                    "\\n<execution_result>\\n<non-empty JSON object or array>\\n</execution_result>"
                ),
                "raw_preview": text[:200],
            },
        )

    matches = _EXECUTION_RESULT_RE.findall(text)
    if not matches:
        return format_tool_error(
            code="EXECUTION_RESULT_TAG_MALFORMED",
            message="'<execution_result>' tag found but \\n boundary or closing tag is missing",
            details={
                "expected_format": (
                    "\\n<execution_result>\\n<non-empty JSON object or array>\\n</execution_result>"
                ),
                "raw_preview": text[:200],
            },
        )
    block = matches[0].strip()

    try:
        parsed = json.loads(block)
    except json.JSONDecodeError as e:
        return format_tool_error(
            code="SIMULATOR_JSON_DECODE_ERROR",
            message="Content inside <execution_result> tags cannot be parsed as JSON",
            details={"raw_preview": block[:200], "error": str(e)},
        )

    if isinstance(parsed, dict):
        if not parsed:
            return format_tool_error(
                code="INVALID_EXECUTION_RESULT_TYPE",
                message="execution_result must be a non-empty object or an array",
                details={"got_type": "object", "got_value": "{}"},
            )
    elif isinstance(parsed, list):
        pass
    else:
        return format_tool_error(
            code="INVALID_EXECUTION_RESULT_TYPE",
            message="execution_result must be a non-empty object or an array",
            details={"got_type": type(parsed).__name__},
        )

    try:
        er_str = json.dumps(parsed, ensure_ascii=False)
    except Exception as e:
        return format_tool_error(
            code="NON_SERIALIZABLE_EXECUTION_RESULT",
            message="execution_result is not JSON-serializable",
            details={"error": str(e)},
        )

    return er_str


def _build_prompt(payload: Dict[str, Any]) -> List[Dict[str, str]]:
    """Build chat messages for one simulator firing.

    System prompt: always VALIDATION_SIMULATOR_SYSTEM_INSTRUCTION.
    User prompt: `build_validation_user_prompt(...)` — per-call DATA only
      (candidate_tools, few-shot, history, raw_caller_message, current
      name/schema/args), each section gated by `inject_*` payload booleans.
    """
    try:
        args_obj = json.loads(payload["arguments"]) if payload.get("arguments") else {}
    except Exception:
        args_obj = {}

    user_content = build_validation_user_prompt(
        examples_lines=payload["examples_lines"],
        schema_str=payload["schema_str"],
        tool_name=payload["name"],
        arguments_obj=args_obj,
        candidate_tools=(payload.get("candidate_tools")
                         if payload.get("inject_candidate_tools") else None),
        conversation_history=(payload.get("conversation_history")
                              if payload.get("inject_conversation_history") else None),
        raw_caller_message=(payload.get("raw_caller_message")
                            if payload.get("inject_raw_caller_message") else None),
        sanitize_fn=lambda s: s,
    )

    return [
        {"role": "system", "content": VALIDATION_SIMULATOR_SYSTEM_INSTRUCTION},
        {"role": "user", "content": user_content},
    ]


@AgentRegistry.register("Tool Simulator Agent")
class ToolSimulatorAgent(BaseAgent):
    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        super().__init__(
            "Tool Simulator Agent",
            "{env_prompt}\n{team_context}\nstep={step}",
            wg_id=wg_id, tokenizer=tokenizer, processor=processor, config=config,
        )

    def call(
        self,
        gen_batch: DataProto,
        env_obs: Dict[str, Any],
        team_context: List[str],
        actor_rollout_wg,
        agent_active_mask: np.ndarray,
        step: int,
    ) -> Tuple[DataProto, List[str]]:
        B = len(env_obs["anchor"])
        sim_payloads = env_obs.get("simulator_payload", [None] * B)

        max_prompt_length = self.config.data.get("rollout_max_prompt_length", None) or self.config.data.max_prompt_length
        truncation = self.config.data.get("rollout_truncation", None) or self.config.data.truncation
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})

        rows: List[Dict[str, Any]] = []
        for i in range(B):
            active = bool(agent_active_mask[i])
            payload = sim_payloads[i] if (sim_payloads is not None and active) else None

            if active and isinstance(payload, dict):
                chat = _build_prompt(payload)
            else:
                chat = [{"role": "system", "content": "."}, {"role": "user", "content": "."}]

            prompt_str = self.tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False,
                **apply_chat_template_kwargs,
            )
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_str, tokenizer=self.tokenizer,
                max_length=max_prompt_length, pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True, truncation=truncation,
            )
            position_ids = compute_position_id_with_mask(attention_mask)
            raw_prompt_ids = self.tokenizer.encode(prompt_str, add_special_tokens=False)
            if len(raw_prompt_ids) > max_prompt_length:
                if truncation == "left":
                    raw_prompt_ids = raw_prompt_ids[-max_prompt_length:]
                elif truncation == "right":
                    raw_prompt_ids = raw_prompt_ids[:max_prompt_length]
                elif truncation == "middle":
                    lh = max_prompt_length // 2
                    raw_prompt_ids = raw_prompt_ids[:lh] + raw_prompt_ids[-(max_prompt_length - lh):]
            rows.append({
                "input_ids": input_ids[0],
                "attention_mask": attention_mask[0],
                "position_ids": position_ids[0],
                "raw_prompt_ids": raw_prompt_ids,
                "raw_prompt": chat,
                "anchor_obs": [],
                "data_source": "checklist_simulator",
                "index": i,
            })

        batched = collate_fn(rows)
        batch = DataProto.from_single_dict(data=batched, meta_info=gen_batch.meta_info)

        try:
            batch, text_responses = self._generate_with_llm(
                batch, actor_rollout_wg, agent_active_mask.astype(bool), gen_batch.meta_info,
            )
        except Exception as e:
            logger.warning(f"[ToolSimulatorAgent] _generate_with_llm raised: {e}")
            text_responses = [
                f"Error when executing tool: simulator_rollout_failed: {e}" if agent_active_mask[i] else ""
                for i in range(B)
            ]

        batch.non_tensor_batch["is_action_valid"] = np.array(
            [1 if (text_responses[i] or "").strip() and agent_active_mask[i] else 0 for i in range(B)],
            dtype=np.int32,
        )
        batch.non_tensor_batch["env_step"] = np.array([step] * B, dtype=object)

        outputs: List[str] = [""] * B
        for i in range(B):
            if not agent_active_mask[i] or sim_payloads[i] is None:
                continue
            raw = text_responses[i] or ""
            if raw.startswith("Error when executing tool:"):
                outputs[i] = raw
            else:
                outputs[i] = _parse_sim_response(raw)

        return batch, outputs
