"""Checklist tool-use environment.

State-machine per sample matching upstream `ToolAgentLoop`:

    PENDING → AWAIT_CALLER → (no tool_calls) → INTERACTING → AWAIT_CALLER / TERMINATED
                           ↘ (tool_calls)   → PROCESSING_TOOLS → AWAIT_CALLER        (httpx mode; or agent mode with all cache-hits/schema-errors)
                                            ↘ PROCESSING_TOOLS → AWAIT_SIMULATOR → AWAIT_CALLER   (agent mode, cache miss + schema ok)

External states (AWAIT_CALLER, AWAIT_SIMULATOR, TERMINATED) are where step()
returns control to rollout_loop so an agent can generate.
Internal states (PROCESSING_TOOLS, INTERACTING) are run to completion inside
step() without returning.

env.step consumes:
  * state == AWAIT_CALLER      → action: raw decoded assistant text
  * state == AWAIT_SIMULATOR   → action: JSON-encoded list[str] matching len(pending_sim_calls)

env.step returns (messages, turn_reward, done, info) where `info['need_agent']`
is 'caller' | 'simulator' | 'none' so the Orchestra can route the next step.

Rewards are emitted only when INTERACTING fires judge (one reward per user-turn
that completes). rollout_loop accumulates into episode_rewards automatically.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import enum
import json
import logging
import random
import uuid as _uuid
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import torch

from .interaction import ChecklistInteraction
from .mcp_tool import MCPChecklistTool, ToolResponse, NEED_SIMULATOR
from simulate_core import build_few_shot_examples_lines, serialize_conversation_history
from .tool_parser import parse_tool_calls, FunctionCall

logger = logging.getLogger(__name__)


class ChecklistState(enum.Enum):
    PENDING = "pending"
    AWAIT_CALLER = "await_caller"
    AWAIT_SIMULATOR = "await_simulator"
    PROCESSING_TOOLS = "processing_tools"
    INTERACTING = "interacting"
    TERMINATED = "terminated"


# X.2 incremental: BASE chat history used as sentinel context for prefix-slicing.
# Byte-aligned with upstream schemas.py:31-34 (non-empty content for defensive robustness
# against chat_templates that may skip empty-content messages or insert extra tokens).
_BASE_CHAT_HISTORY: List[Dict[str, str]] = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
]


# -------- helpers --------

def _aggregate_turn_score(score_tuple, checklist_list: List[Any], turn_idx: int) -> float:
    """Compute the weighted-checklist turn score, equivalent to upstream
    ChecklistRewardManager's per-turn aggregation.

    Args:
        score_tuple: (per_step_results_list, turns_list, call_success_list)
            per_step_results_list: list of (per-checklist) lists of step-level standard results
            shape = (num_checklists, num_assistant_steps_in_turn, num_standards_in_turn)
        checklist_list: list of checklists (num_checklists × num_turns × num_standards_per_turn)
        turn_idx: which user-turn we just evaluated.

    Returns:
        Sum over checklists of weighted turn score (same scale as upstream before normalization).
    """
    if not score_tuple:
        return 0.0
    per_step_results_list = score_tuple[0]
    total = 0.0
    for j, checklist_j in enumerate(checklist_list):
        if j >= len(per_step_results_list):
            break
        if turn_idx >= len(checklist_j):
            continue
        this_turn_standards = checklist_j[turn_idx]
        n_std = len(this_turn_standards)
        if n_std == 0:
            continue
        # OR aggregation across assistant steps within this turn
        aggregated = [False] * n_std
        for step_standards in per_step_results_list[j]:
            for k in range(min(n_std, len(step_standards))):
                if step_standards[k]:
                    aggregated[k] = True
        weights = [float(s.get("weight", 0.0)) for s in this_turn_standards]
        total += sum((1.0 if a else 0.0) * w for a, w in zip(aggregated, weights))
    return float(total)


# -------- single env --------

class ChecklistEnv:
    """Single-sample checklist state machine (async-friendly; driven via asyncio.run).

    Shared resources (tool_executor, interaction) are injected by the batch wrapper.
    """

    def __init__(
        self,
        cfg: Any,
        tool_executor: MCPChecklistTool,
        interaction: ChecklistInteraction,
    ):
        self.cfg = cfg
        # Default aligns with run_checklist.sh: "agent" means cache-miss tool
        # responses come from the co-trained Tool Simulator Agent, not external
        # sglang. A missing config must NOT silently switch to "httpx".
        self.tool_return_mode = getattr(cfg, "tool_return_mode", "agent")
        self.max_assistant_turns = int(getattr(cfg, "max_assistant_turns", 30))
        self.max_user_turns = int(getattr(cfg, "max_user_turns", 30))
        self.max_parallel_calls = int(getattr(cfg, "max_parallel_calls", 20))
        self.tool_executor = tool_executor
        self.interaction = interaction

        # Per-episode state (populated in reset)
        self.messages: List[Dict[str, Any]] = []
        self.all_messages: List[Dict[str, Any]] = []
        self.tools: List[Dict[str, Any]] = []
        self.checklist_list: List[Any] = []
        self.uuid: str = ""
        self.iid: str = ""
        self.state: ChecklistState = ChecklistState.PENDING
        self.assistant_turns: int = 0
        self.user_turns: int = 0
        self.per_step_turn_rewards: List[float] = []
        self.pending_tool_calls: List[FunctionCall] = []
        # When in hybrid/fallback-to-simulator: which of pending_tool_calls
        # still need Simulator content. Keyed by index within pending_tool_calls.
        self.pending_sim_indices: List[int] = []
        self.pending_nonsim_results: Dict[int, str] = {}  # idx → already-resolved text (cache/schema/httpx)
        # Per-call cursor: AWAIT_SIMULATOR fires once per index in pending_sim_indices.
        # Each firing resolves pending_sim_indices[cursor] then advances; when cursor
        # reaches len(pending_sim_indices) all tool messages are flushed and state
        # transitions to AWAIT_CALLER.
        self.pending_sim_cursor: int = 0
        self._last_tool_failed: bool = False
        self._last_error_code: Optional[str] = None  # parse-error code (wandb only)

        # ---- wandb monitoring (X.1.2 patch follow-up) ----
        # Per-step bookkeeping to surface in info dict for wandb. Reset every step.
        # _last_step_tool_lens: List[int] — char length of each tool response written this step
        # _last_step_tool_sources: List[str] — source classification: "cache" / "simulator" / "schema_error" / "httpx" / "error_tool_call"
        # _last_step_name_hits / _last_step_args_hits / _last_step_few_shot_counts:
        #   parallel per-call lists. name_hit = name in cache map (regardless of args);
        #   args_hit = full (name,args) cache hit; few_shot_count = #(args,content) pairs
        #   available for this sample at execute() time. error_tool_call (caller-side parse
        #   error) and tool-not-found pre-checks contribute False/False/0.
        self._last_step_tool_lens: List[int] = []
        self._last_step_tool_sources: List[str] = []
        self._last_step_name_hits: List[bool] = []
        self._last_step_args_hits: List[bool] = []
        self._last_step_few_shot_counts: List[int] = []
        # Per-call metrics dict surviving PROCESSING_TOOLS → AWAIT_SIMULATOR boundary
        # (sim calls only get their tool message flushed after the simulator agent fires).
        self._pending_call_metrics: Dict[int, Dict[str, Any]] = {}

        # ---- Caller-side incremental token state (X.2 plan; upstream-aligned) ----
        # When config.env.checklist.use_caller_incremental=True, ChecklistEnv
        # maintains a token-level accumulator for caller. Each turn appends
        # delta tokens via BASE-prefix slicing (mirrors upstream schemas.py:385-475).
        # Used by ToolCallerAgent._preprocess_checklist_batch_incremental to
        # bypass per-turn apply_chat_template re-rendering. Simulator path
        # is unaffected (it always renders fresh prompts; see plan §X.2.2).
        self.use_caller_incremental: bool = bool(getattr(cfg, "use_caller_incremental", False))
        self.tokenizer = None  # injected by ChecklistMultiProcessEnv.reset via env_kwargs (step 5.1)
        self.apply_chat_template_kwargs: Dict[str, Any] = dict(
            getattr(cfg, "apply_chat_template_kwargs", {}) or {}
        )
        self.caller_input_ids: Optional[torch.Tensor] = None              # [1, L_total]
        self.caller_attention_mask: Optional[torch.Tensor] = None         # [1, L_total]
        self.caller_loss_mask: Optional[torch.Tensor] = None              # [1, L_total]; 1=caller, 0=other
        self.generation_prompt_ids: Optional[torch.Tensor] = None         # [1, k]; "<|im_start|>assistant\n"
        self.base_conv_wo_gen_prompt_end_pos: int = 0
        self.base_conv_with_gen_prompt_end_pos: int = 0

        # ---- Length budget for caller (X.2 step 12; upstream sglang_rollout.py:976-980 alignment) ----
        # When caller_input_ids exceeds this budget, episode terminates with
        # error_code=LENGTH_EXCEEDED instead of left-truncating prompt.
        # max_model_len defaults to sglang context (typically 20000-32000).
        # max_response_length is reserved as caller's generation budget so prompt + response
        # < max_model_len. Mirrors upstream:
        #   if prompt_length + 1 >= max_model_len: finish_reason=LENGTH; break
        # Only used when use_caller_incremental=True.
        self.max_model_len: int = int(getattr(cfg, "max_model_len", 20000))
        self.max_response_length: int = int(getattr(cfg, "max_response_length", 8000))

    # ---------- async entry points ----------

    async def async_reset(self, env_kwargs: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        self.messages = list(env_kwargs.get("messages", []))
        self.all_messages = list(env_kwargs.get("all_messages", self.messages))
        # Deep-sanitize dataset-supplied tool schemas: they reach apply_chat_template
        # at every turn (envs.py:_bos_append_helper / _append_tool_messages_batch),
        # so an unsanitized invalid code point in any tool description / parameter
        # docstring crashes the Rust tokenizer with TextEncodeInput TypeError.
        # Mirrors the LLM-output entry-point sanitize at envs.py:306/391 (commit
        # 2fa0ded) but for the static, dataset-side input.
        self.tools = MCPChecklistTool._deep_sanitize_for_tokenizer(
            list(env_kwargs.get("tools", []))
        )
        self.checklist_list = list(env_kwargs.get("checklist_list", []))
        self.uuid = str(env_kwargs.get("uuid", ""))       # sample identity (may be shared across rollouts)
        self.assistant_turns = 0
        self.user_turns = 0
        self.per_step_turn_rewards = []
        self.pending_tool_calls = []
        self.pending_sim_indices = []
        self.pending_nonsim_results = {}
        self.pending_sim_cursor = 0
        self._last_tool_failed = False
        self._last_error_code = None
        # Reset wandb monitoring buckets (X.1.2 follow-up)
        self._last_step_tool_lens = []
        self._last_step_tool_sources = []
        self._last_step_name_hits = []
        self._last_step_args_hits = []
        self._last_step_few_shot_counts = []
        self._pending_call_metrics = {}
        # Reset caller token state (X.2 incremental)
        self.caller_input_ids = None
        self.caller_attention_mask = None
        self.caller_loss_mask = None
        self.generation_prompt_ids = None
        self.base_conv_wo_gen_prompt_end_pos = 0
        self.base_conv_with_gen_prompt_end_pos = 0
        # Pad / dummy slot — VecEnv pads batch with empty checklist_list when
        # real samples < batch_size. Mirror math/search per-env behavior:
        # gracefully consume the dummy by entering zero-state immediately,
        # without invoking the stateful interaction.
        if not self.checklist_list:
            self.iid = None
            self.state = ChecklistState.TERMINATED
            # Don't init caller token state for dummy samples — they're skipped by Orchestra
            info = self._build_info(turn_reward=0.0, done=True)
            return self.messages, info
        # CRITICAL: interaction instance_id must be unique per env/rollout.
        # When env.rollout.n > 1, multiple envs share the same sample uuid;
        # using self.uuid as instance_id would cause ChecklistInteraction._instance_dict
        # to be clobbered across concurrent rollouts. Generate a fresh uuid here.
        fresh_iid = f"{self.uuid or 'sample'}_{_uuid.uuid4().hex}"
        self.iid = await self.interaction.start_interaction(
            instance_id=fresh_iid, checklist_list=self.checklist_list,
        )
        self.state = ChecklistState.AWAIT_CALLER

        # ---- Initialize caller-side incremental token state (X.2 plan) ----
        # Read tokenizer from env_kwargs (injected by ChecklistMultiProcessEnv.reset, step 5.1).
        # Compute BASE_CHAT_HISTORY end positions and full initial prompt token list.
        self.tokenizer = env_kwargs.get("tokenizer", None)
        if self.use_caller_incremental:
            if self.tokenizer is None:
                raise RuntimeError(
                    "use_caller_incremental=True but tokenizer not injected via env_kwargs. "
                    "Update ChecklistMultiProcessEnv.reset to pass tokenizer (see plan step 5.1)."
                )
            self._init_caller_token_state()
            # X.2 step 12: edge case — if even initial prompt exceeds budget, terminate immediately.
            # Mirrors upstream behavior where pre-rollout length check would prevent the episode.
            if not self._check_caller_length_or_terminate():
                info = self._build_info(turn_reward=0.0, done=True)
                return self.messages, info

        info = {
            "need_agent": "caller",
            "state": self.state.value,
            "uuid": self.uuid,
            "num_checklists": len(self.checklist_list),
            "max_num_turns": len(self.checklist_list[0]) if self.checklist_list else 1,
            "data_source": str(env_kwargs.get("data_source", "checklist")),
        }
        return self.messages, info

    async def async_step(self, action: Any) -> Tuple[List[Dict[str, Any]], float, bool, Dict[str, Any]]:
        turn_reward = 0.0
        # Reset per-step tool monitoring buckets (wandb)
        self._last_step_tool_lens = []
        self._last_step_tool_sources = []
        self._last_step_name_hits = []
        self._last_step_args_hits = []
        self._last_step_few_shot_counts = []

        # ---- Phase 1: consume action based on current state ----
        if self.state == ChecklistState.AWAIT_CALLER:
            caller_text = action if isinstance(action, str) else ""
            # sglang detokenize 在 mid-byte 截断时会留 lone UTF-16 surrogate（Python str 合法、
            # 但 fast tokenizer Rust 后端 encode_batch 直接拒绝 → TextEncodeInput TypeError）。
            # 在状态机入口剥掉，下游 self.messages / tool_calls.arguments / error_tool_call /
            # caller-incremental token 拼接 / simulator payload 全部受益。
            caller_text = self.tool_executor._sanitize_text_for_tokenizer(caller_text)
            # Mirror upstream sglang_rollout.py:1019-1107. On any parse error,
            # append the assistant text + an error_tool_call tool message,
            # then hard-terminate the trajectory.
            #
            # NOTE on content: upstream stores raw `text` (with <tool_call> tags) in
            # add_assistant_message because it uses skip_tokenizer_init and
            # bypasses chat_template on re-render. this repo goes through
            # apply_chat_template every turn, which would double-render the
            # tags (once from content, once from the tool_calls field). So when
            # tool_calls are present we use `normal_text` (first-tag-prefix
            # only) — semantically equivalent to upstream's rendered output.
            content, tool_calls, parse_error = parse_tool_calls(caller_text)
            # Cache raw (sanitized) caller_text for simulator validation prompt
            # injection. Survives until next AWAIT_CALLER turn; consumed by
            # _build_info when state == AWAIT_SIMULATOR.
            self._last_caller_text = caller_text
            if tool_calls:
                idx = caller_text.find("\n<tool_call>\n")
                normal_text = caller_text[:idx] if idx != -1 else caller_text
                msg_content = normal_text
            else:
                msg_content = content  # raw text (final answer or parse error)
            assistant_msg: Dict[str, Any] = {
                "role": "assistant",
                "content": msg_content,
            }
            if tool_calls:
                # record as openai-style tool_calls list-of-dict for downstream judge/interaction
                assistant_msg["tool_calls"] = [
                    {
                        "id": f"call_{self.uuid}_{self.assistant_turns}_{i}",
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments},
                    }
                    for i, tc in enumerate(tool_calls)
                ]
            self.messages.append(assistant_msg)
            self.assistant_turns += 1

            # X.2 incremental: append assistant tokens to caller_input_ids
            if self.use_caller_incremental:
                self._append_assistant_to_caller(
                    content=msg_content,
                    tool_calls=assistant_msg.get("tool_calls"),
                    response_token_ids=None,  # re-render via _bos_append_helper (no LLM raw tokens passed in)
                )

            if parse_error is not None:
                # upstream line 1060-1066 / 1080-1084 / 1102-1107: append error_tool_call
                # tool message and STOP the trajectory.
                _err_content = json.dumps(parse_error["payload"], ensure_ascii=True)
                self.messages.append({
                    "role": "tool",
                    "name": "error_tool_call",
                    "content": _err_content,
                })
                self._last_step_tool_lens.append(len(_err_content))
                self._last_step_tool_sources.append("error_tool_call")
                # Caller-side parse error never executed mcp_tool, so no hit/few-shot signal
                self._last_step_name_hits.append(False)
                self._last_step_args_hits.append(False)
                self._last_step_few_shot_counts.append(0)
                # X.2 incremental: append error tool message to caller_input_ids
                if self.use_caller_incremental:
                    self._append_tool_to_caller("error_tool_call", _err_content)
                self._last_error_code = parse_error["code"]
                self._last_tool_failed = True
                self.state = ChecklistState.TERMINATED
                done = True
                info = self._build_info(turn_reward=0.0, done=done)
                return self.messages, 0.0, done, info

            if tool_calls:
                self.pending_tool_calls = tool_calls
                self.state = ChecklistState.PROCESSING_TOOLS
            else:
                self.state = ChecklistState.INTERACTING

        elif self.state == ChecklistState.AWAIT_SIMULATOR:
            # action: single-string tool result for pending_sim_indices[cursor].
            # Each AWAIT_SIMULATOR firing handles ONE sim call (per-call cursor;
            # see plan §3.4). When cursor reaches end, all tool messages flush.
            sim_text = action if isinstance(action, str) else ""
            # 同 AWAIT_CALLER：剥掉 simulator detokenize 截断的 lone surrogate
            sim_text = self.tool_executor._sanitize_text_for_tokenizer(sim_text)
            if self.pending_sim_cursor >= len(self.pending_sim_indices):
                # Defensive: shouldn't happen, but if it does reset and bounce back to caller.
                logger.warning("[ChecklistEnv] AWAIT_SIMULATOR cursor overrun, forcing AWAIT_CALLER")
                self.pending_tool_calls = []
                self.pending_sim_indices = []
                self.pending_nonsim_results = {}
                self.pending_sim_cursor = 0
                self.state = ChecklistState.AWAIT_CALLER
            else:
                cur_idx = self.pending_sim_indices[self.pending_sim_cursor]
                self.pending_nonsim_results[cur_idx] = sim_text
                self.pending_sim_cursor += 1

                if self.pending_sim_cursor < len(self.pending_sim_indices):
                    # More sim calls pending in this turn → stay in AWAIT_SIMULATOR.
                    # _build_info will surface the next call's payload (cursor advanced).
                    info = self._build_info(turn_reward=0.0, done=False)
                    return self.messages, 0.0, False, info

                # All sim calls resolved → flush tool messages in original tool_calls order
                # (mirroring the pre-cursor design's batched flush).
                sim_failed_any = False
                _sim_indices_snapshot = list(self.pending_sim_indices)  # snapshot for X.2 source check
                _tool_msgs_batch: List[Dict[str, Any]] = []
                for i, tc in enumerate(self.pending_tool_calls):
                    content = self.pending_nonsim_results.get(i, "")
                    _tool_msg = {
                        "role": "tool",
                        "name": tc.name,
                        "content": content,
                    }
                    self.messages.append(_tool_msg)
                    self._last_step_tool_lens.append(len(content))
                    # Source: simulator if originally NEED_SIMULATOR, else preserve original source
                    if i in _sim_indices_snapshot:
                        self._last_step_tool_sources.append("simulator")
                    else:
                        self._last_step_tool_sources.append("cache_or_schema")  # already-resolved by mcp_tool
                    _m = self._pending_call_metrics.get(i, {})
                    self._last_step_name_hits.append(bool(_m.get("name_hit", False)))
                    self._last_step_args_hits.append(bool(_m.get("args_hit", False)))
                    self._last_step_few_shot_counts.append(int(_m.get("few_shot_count", 0)))
                    if self.use_caller_incremental:
                        _tool_msgs_batch.append(_tool_msg)
                    try:
                        parsed = json.loads(content) if content else {}
                        if isinstance(parsed, dict) and "error_tool_call" in parsed:
                            sim_failed_any = True
                    except Exception:
                        pass
                # X.2 incremental: append all tool messages in one chat_template call
                # (mirrors upstream schemas.py:449)
                if self.use_caller_incremental and _tool_msgs_batch:
                    self._append_tool_messages_batch(_tool_msgs_batch)
                # Reset bookkeeping
                self.pending_tool_calls = []
                self.pending_sim_indices = []
                self.pending_nonsim_results = {}
                self.pending_sim_cursor = 0
                self._pending_call_metrics = {}
                if sim_failed_any:
                    self._last_tool_failed = True
                    self.state = ChecklistState.TERMINATED
                    done = True
                    info = self._build_info(turn_reward=0.0, done=done)
                    return self.messages, 0.0, done, info
                self.state = ChecklistState.AWAIT_CALLER
                # X.2 incremental: ensure generation_prompt at tail before next caller turn
                if self.use_caller_incremental:
                    self._ensure_generation_prompt_at_tail()
                    # X.2 step 12: length check — terminate if caller can't fit another turn
                    if not self._check_caller_length_or_terminate():
                        done = True
                        info = self._build_info(turn_reward=0.0, done=done)
                        return self.messages, 0.0, done, info

        elif self.state == ChecklistState.TERMINATED:
            # Already done; this shouldn't be called. Return idempotent terminal.
            info = self._build_info(turn_reward=0.0, done=True)
            return self.messages, 0.0, True, info

        else:
            raise RuntimeError(f"ChecklistEnv.step called in unexpected state: {self.state}")

        # ---- Phase 2: run internal transitions (PROCESSING_TOOLS, INTERACTING) ----
        while self.state in (ChecklistState.PROCESSING_TOOLS, ChecklistState.INTERACTING):

            if self.state == ChecklistState.PROCESSING_TOOLS:
                # Resolve each pending tool call via MCPChecklistTool.
                # Any that return NEED_SIMULATOR (only possible when mode=='agent')
                # are queued for the Simulator agent; others get text directly.
                self.pending_nonsim_results = {}
                self.pending_sim_indices = []
                self._pending_nonsim_sources: Dict[int, str] = {}  # idx → source from metrics
                self._pending_call_metrics = {}
                tool_failed_any = False
                for i, tc in enumerate(self.pending_tool_calls):
                    try:
                        arguments = json.loads(tc.arguments) if tc.arguments else {}
                    except Exception:
                        arguments = {}
                    resp, _, metrics = await self.tool_executor.execute(
                        name=tc.name,
                        parameters=arguments,
                        original_index=self.uuid,
                        mode=self.tool_return_mode,
                        instance_id=self.uuid,
                    )
                    # Capture hit/few-shot signal regardless of branch (sim/non-sim) so the
                    # AWAIT_SIMULATOR flush can read it back when sim calls finally finish.
                    self._pending_call_metrics[i] = {
                        "name_hit": bool(metrics.get("name_hit", False)),
                        "args_hit": bool(metrics.get("args_hit", False)),
                        "few_shot_count": int(metrics.get("few_shot_count", 0)),
                    }
                    if resp is NEED_SIMULATOR:
                        self.pending_sim_indices.append(i)
                    else:
                        # resp is a ToolResponse
                        text = resp.text if hasattr(resp, "text") else str(resp)
                        self.pending_nonsim_results[i] = text or ""
                        # Record source from metrics (cache / schema_error / httpx / unknown)
                        self._pending_nonsim_sources[i] = str(metrics.get("source", "httpx"))
                        if not metrics.get("success", True):
                            tool_failed_any = True
                        # Detect upstream-style error_tool_call payloads (schema/unknown-index errors)
                        try:
                            parsed = json.loads(text) if text else {}
                            if isinstance(parsed, dict) and "error_tool_call" in parsed:
                                tool_failed_any = True
                        except Exception:
                            pass
                self._last_tool_failed = self._last_tool_failed or tool_failed_any

                if self.pending_sim_indices:
                    # Agent mode fallback needed. Yield control to Orchestra for Simulator.
                    # Reset cursor — AWAIT_SIMULATOR fires once per pending sim index.
                    self.pending_sim_cursor = 0
                    self.state = ChecklistState.AWAIT_SIMULATOR
                    break
                # All tools resolved inline → append tool messages and go back to Caller
                _tool_msgs_batch: List[Dict[str, Any]] = []
                for i, tc in enumerate(self.pending_tool_calls):
                    _content = self.pending_nonsim_results.get(i, "")
                    _tool_msg = {
                        "role": "tool",
                        "name": tc.name,
                        "content": _content,
                    }
                    self.messages.append(_tool_msg)
                    self._last_step_tool_lens.append(len(_content))
                    self._last_step_tool_sources.append(self._pending_nonsim_sources.get(i, "httpx"))
                    _m = self._pending_call_metrics.get(i, {})
                    self._last_step_name_hits.append(bool(_m.get("name_hit", False)))
                    self._last_step_args_hits.append(bool(_m.get("args_hit", False)))
                    self._last_step_few_shot_counts.append(int(_m.get("few_shot_count", 0)))
                    if self.use_caller_incremental:
                        _tool_msgs_batch.append(_tool_msg)
                # X.2 incremental: append all tool messages in one chat_template call
                # (mirrors upstream schemas.py:449 single-call render of N tool messages)
                if self.use_caller_incremental and _tool_msgs_batch:
                    self._append_tool_messages_batch(_tool_msgs_batch)
                self.pending_tool_calls = []
                self.pending_nonsim_results = {}
                self._pending_nonsim_sources = {}
                self._pending_call_metrics = {}
                # Mirror upstream line 954-960 / 962-965: any error_tool_call or
                # metrics.success==False → STOP the trajectory immediately.
                if tool_failed_any:
                    self.state = ChecklistState.TERMINATED
                    break
                self.state = ChecklistState.AWAIT_CALLER
                # X.2 incremental: ensure generation_prompt at tail before next caller turn
                if self.use_caller_incremental:
                    self._ensure_generation_prompt_at_tail()
                    # X.2 step 12: length check — break out of while loop if exceeded
                    if not self._check_caller_length_or_terminate():
                        break

            elif self.state == ChecklistState.INTERACTING:
                # End of a user-turn: judge this turn, emit reward, optionally pull next user msg.
                turn_idx = self.user_turns  # turn we just completed
                try:
                    terminate, next_user_content, score_tuple = \
                        await self.interaction.generate_response(
                            self.iid, self.messages, self.all_messages,
                            checklist_list=self.checklist_list,
                        )
                except Exception as e:
                    logger.warning(f"[ChecklistEnv] interaction.generate_response failed: {e}")
                    terminate, next_user_content, score_tuple = True, "", None

                turn_reward = _aggregate_turn_score(score_tuple, self.checklist_list, turn_idx)
                if self._last_tool_failed:
                    turn_reward = 0.0
                self.per_step_turn_rewards.append(turn_reward)

                # Advance user_turn (since we just scored user_turn=turn_idx)
                self.user_turns += 1

                hit_assistant_cap = self.assistant_turns >= self.max_assistant_turns
                hit_user_cap = self.user_turns >= self.max_user_turns
                if terminate or hit_assistant_cap or hit_user_cap or not next_user_content:
                    self.state = ChecklistState.TERMINATED
                else:
                    # 同 caller_text/sim_text: judge-side user-turn output 也可能挟带
                    # sglang/openai 截断的非法 Unicode (lone surrogate / NUL /
                    # noncharacter), 进 apply_chat_template 会炸 Rust tokenizer。
                    next_user_content = self.tool_executor._sanitize_text_for_tokenizer(
                        next_user_content
                    )
                    self.messages.append({"role": "user", "content": next_user_content})
                    # X.2 incremental: append user message + ensure generation_prompt at tail
                    if self.use_caller_incremental:
                        self._append_user_to_caller(next_user_content)
                        self._ensure_generation_prompt_at_tail()
                        # X.2 step 12: length check — terminate if caller can't fit next turn
                        if not self._check_caller_length_or_terminate():
                            break  # state TERMINATED, fall through to done check
                    self.state = ChecklistState.AWAIT_CALLER
                break  # Only one INTERACTING per step

        done = (self.state == ChecklistState.TERMINATED)
        info = self._build_info(turn_reward=turn_reward, done=done)
        return self.messages, turn_reward, done, info

    # ---------- helpers ----------

    # ============ X.2 Caller-side incremental token state helpers ============
    # When use_caller_incremental=True, ChecklistEnv maintains a token-level
    # accumulator (caller_input_ids) so ToolCallerAgent can skip per-turn
    # apply_chat_template re-rendering. Mirrors upstream schemas.py:385-475 BASE
    # prefix-slicing trick. Only the caller path is incremental; simulator
    # always renders fresh prompts (see plan §X.2.2).

    def _init_caller_token_state(self) -> None:
        """Compute initial caller_input_ids + BASE end positions.

        Uses module-level _BASE_CHAT_HISTORY as a sentinel context to anchor
        incremental rendering: any future message is rendered as [BASE + msg] and
        prefix-cut to extract the message's standalone tokens. Mirrors upstream
        schemas.py:204-219 (with byte-aligned BASE content from upstream schemas.py:31-34).
        """
        base_wo = self.tokenizer.apply_chat_template(
            _BASE_CHAT_HISTORY,
            tools=self.tools,
            add_generation_prompt=False,
            tokenize=True,
            **self.apply_chat_template_kwargs,
        )
        base_with = self.tokenizer.apply_chat_template(
            _BASE_CHAT_HISTORY,
            tools=self.tools,
            add_generation_prompt=True,
            tokenize=True,
            **self.apply_chat_template_kwargs,
        )
        self.base_conv_wo_gen_prompt_end_pos = len(base_wo)
        self.base_conv_with_gen_prompt_end_pos = len(base_with)

        # Render initial messages with generation prompt — this is what caller sees on turn 1
        full_with = self.tokenizer.apply_chat_template(
            self.messages,
            tools=self.tools,
            add_generation_prompt=True,
            tokenize=True,
            **self.apply_chat_template_kwargs,
        )
        full_wo = self.tokenizer.apply_chat_template(
            self.messages,
            tools=self.tools,
            add_generation_prompt=False,
            tokenize=True,
            **self.apply_chat_template_kwargs,
        )

        self.caller_input_ids = torch.tensor([full_with], dtype=torch.long)
        delta = len(full_with) - len(full_wo)
        if delta <= 0:
            # Fallback if tokenizer doesn't add a generation prompt (extremely rare)
            self.generation_prompt_ids = torch.zeros((1, 0), dtype=torch.long)
        else:
            self.generation_prompt_ids = self.caller_input_ids[..., -delta:].clone()

        self.caller_loss_mask = torch.zeros_like(self.caller_input_ids)
        self.caller_attention_mask = torch.ones_like(self.caller_input_ids)

    def _log_text_encode_input_crash(self, where: str, payload: Any) -> None:
        """Diagnostic for TextEncodeInput TypeError from Rust fast tokenizer.

        Walks `payload` and `self.tools`, prints first invalid code point seen
        (range/index/40-char window) so the next recurrence pinpoints which
        field — content / tool_calls.arguments / tools schema — slipped through
        sanitize. Best-effort; never raises.
        """
        def _scan(s: Any, path: str):
            if isinstance(s, str):
                for i, ch in enumerate(s):
                    c = ord(ch)
                    if (
                        (0xD800 <= c <= 0xDFFF)
                        or c == 0x00
                        or (0xFDD0 <= c <= 0xFDEF)
                        or ((c & 0xFFFE) == 0xFFFE)
                    ):
                        win = s[max(0, i - 20): i + 20]
                        return (path, i, hex(c), repr(win))
                return None
            if isinstance(s, dict):
                for k, v in s.items():
                    hit = _scan(v, f"{path}.{k}")
                    if hit:
                        return hit
            elif isinstance(s, (list, tuple)):
                for i, v in enumerate(s):
                    hit = _scan(v, f"{path}[{i}]")
                    if hit:
                        return hit
            return None

        try:
            hit_msg = _scan(payload, "msg")
            hit_tools = _scan(self.tools, "tools")
            logger.error(
                "[ChecklistEnv][%s] TextEncodeInput crash uuid=%s turn=%d "
                "msg_hit=%s tools_hit=%s",
                where, getattr(self, "uuid", "?"),
                getattr(self, "assistant_turns", -1),
                hit_msg, hit_tools,
            )
        except Exception:
            pass

    def _bos_append_helper(self, msg: Dict[str, Any], prefix_end_pos: int) -> torch.Tensor:
        """Render BASE + [msg] then slice off the BASE prefix, returning standalone msg tokens.

        Mirrors upstream schemas.py:391, 410, 449. The trick:
        - BASE [system, user] gives the new message a stable preceding context
        - chat_template renders msg with its proper boundary tokens (no `loop.first` artifacts)
        - Slicing off prefix_end_pos removes the BASE; what remains is byte-correct
          for concatenation onto self.caller_input_ids.

        Args:
            msg: dict with role/content/tool_calls
            prefix_end_pos: base_conv_wo_gen_prompt_end_pos (for tool/user msgs that
                follow a user role) OR base_conv_with_gen_prompt_end_pos (for
                assistant msgs that follow a generation prompt position)

        Returns:
            torch.LongTensor of shape [1, L_msg]
        """
        try:
            rendered = self.tokenizer.apply_chat_template(
                _BASE_CHAT_HISTORY + [msg],
                tools=self.tools,
                add_generation_prompt=False,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            )
        except TypeError as e:
            if "TextEncodeInput" in str(e):
                self._log_text_encode_input_crash("_bos_append_helper", msg)
            raise
        return torch.tensor([rendered], dtype=torch.long)[..., prefix_end_pos:]

    def _append_assistant_to_caller(
        self,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]],
        response_token_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Append an assistant message to caller_input_ids.

        Two paths:
        1. response_token_ids != None: use raw caller-LLM output tokens (NO retokenize).
           Mirrors upstream schemas.py:409 (when content_ids is provided). Avoids BPE drift.
           Caller LLM output may not have trailing <|im_end|>\\n (e.g. finish_reason=length),
           so we need to ensure them — mirrors upstream schemas.py:418-420.
        2. response_token_ids == None: use _bos_append_helper. Used when caller text
           has been parsed into content + tool_calls and we need chat_template to render
           correct <tool_call> boundary tokens. chat_template already adds <|im_end|>\\n
           at message boundary, so NO extra <|im_end|>\\n is appended (this differs from
           upstream schemas.py:418-420 which always appends — upstream effectively double-adds in
           this path; we deliberately diverge here so incremental output is byte-aligned
           with full-rendered output for the same messages list).
        """
        msg: Dict[str, Any] = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = tool_calls

        if response_token_ids is not None:
            content_ids = response_token_ids
            if content_ids.dim() == 1:
                content_ids = content_ids.unsqueeze(0)
            # Path 1: ensure trailing <|im_end|>\n (upstream schemas.py:418-420)
            eos_token_ids = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
            nl_token_ids = self.tokenizer.encode("\n", add_special_tokens=False)
            if not eos_token_ids:
                eos_token_ids = [self.tokenizer.eos_token_id]
            if not nl_token_ids:
                nl_token_ids = []

            last_tok = content_ids[0, -1].item() if content_ids.numel() > 0 else None
            if last_tok != eos_token_ids[-1]:
                content_ids = torch.cat(
                    [content_ids, torch.tensor([eos_token_ids], dtype=torch.long)], dim=-1
                )
            if nl_token_ids:
                content_ids = torch.cat(
                    [content_ids, torch.tensor([nl_token_ids], dtype=torch.long)], dim=-1
                )
        else:
            # Path 2: assistant follows user role → use with_gen_prompt prefix
            # _bos_append_helper output already includes <|im_end|>\n from chat_template,
            # so we do NOT append again (would cause duplicate boundary tokens).
            content_ids = self._bos_append_helper(msg, self.base_conv_with_gen_prompt_end_pos)

        self._extend_caller_state(content_ids, loss=True)

    def _append_tool_to_caller(self, name: str, content: str) -> None:
        """Append a single tool/observation message. loss_mask=False (not caller's output).

        Convenience wrapper around _append_tool_messages_batch for single-tool-call paths
        (e.g. parse_error tool message). For N tool messages from one assistant turn, prefer
        _append_tool_messages_batch (single chat_template call, mirrors upstream schemas.py:449).
        """
        self._append_tool_messages_batch([{"role": "tool", "name": name, "content": content}])

    def _append_tool_messages_batch(self, tool_msgs: List[Dict[str, Any]]) -> None:
        """Append N tool/observation messages in one chat_template call. loss_mask=False.

        Mirrors upstream schemas.py:449 ``messages = [*BASE_CHAT_HISTORY, *self.messages[-N:]]``
        — single _handle_apply_chat_template invocation for all N tool messages from one
        assistant turn, then prefix-cut to extract their concatenated tokens. Saves N-1
        chat_template calls compared to looping single-msg _bos_append_helper N times.
        """
        if not tool_msgs:
            return
        try:
            rendered = self.tokenizer.apply_chat_template(
                _BASE_CHAT_HISTORY + tool_msgs,
                tools=self.tools,
                add_generation_prompt=False,
                tokenize=True,
                **self.apply_chat_template_kwargs,
            )
        except TypeError as e:
            if "TextEncodeInput" in str(e):
                self._log_text_encode_input_crash("_append_tool_messages_batch", tool_msgs)
            raise
        content_ids = torch.tensor([rendered], dtype=torch.long)[..., self.base_conv_wo_gen_prompt_end_pos:]
        self._extend_caller_state(content_ids, loss=False)

    def _append_user_to_caller(self, content: str) -> None:
        """Append a user message (next-turn injection from interaction). loss_mask=False."""
        msg = {"role": "user", "content": content}
        content_ids = self._bos_append_helper(msg, self.base_conv_wo_gen_prompt_end_pos)
        self._extend_caller_state(content_ids, loss=False)

    def _extend_caller_state(self, new_ids: torch.Tensor, loss: bool) -> None:
        """Concat new_ids (shape [1, L_new]) onto caller_input_ids/attention_mask/loss_mask."""
        if new_ids.numel() == 0:
            return
        if new_ids.dim() == 1:
            new_ids = new_ids.unsqueeze(0)
        self.caller_input_ids = torch.cat([self.caller_input_ids, new_ids], dim=-1)
        self.caller_attention_mask = torch.cat(
            [self.caller_attention_mask, torch.ones_like(new_ids)], dim=-1
        )
        self.caller_loss_mask = torch.cat(
            [self.caller_loss_mask, torch.full_like(new_ids, int(loss))], dim=-1
        )

    def _ensure_generation_prompt_at_tail(self) -> None:
        """Before caller inference, ensure caller_input_ids ends with <|im_start|>assistant\\n.

        Called when transitioning back to AWAIT_CALLER (after PROCESSING_TOOLS,
        AWAIT_SIMULATOR flush, or INTERACTING). Mirrors upstream schemas.py:362-368
        where get_generation_prompt_ids checks tail equality before appending.
        """
        if self.generation_prompt_ids is None or self.generation_prompt_ids.numel() == 0:
            return
        delta = self.generation_prompt_ids.shape[-1]
        if self.caller_input_ids.shape[-1] >= delta:
            tail = self.caller_input_ids[..., -delta:]
            if torch.equal(tail, self.generation_prompt_ids):
                return  # already at tail
        self._extend_caller_state(self.generation_prompt_ids, loss=False)

    def _check_caller_length_or_terminate(self) -> bool:
        """X.2 step 12: check if caller can run another turn without exceeding max_model_len.

        Mirrors upstream sglang_rollout.py:976-980:
            prompt_length = len(_req.get_generation_prompt_ids(...))
            if prompt_length + 1 >= self.config.max_model_len:
                finish_reason_type = FinishReasonTypeEnum.LENGTH
                break

        this repo adaptation: also reserve max_response_length budget so caller has
        actual space to generate (not just barely fit prompt).

        Returns:
            True if length OK and caller can proceed.
            False if length exceeded → state set to TERMINATED + error_code=LENGTH_EXCEEDED.
        """
        if not self.use_caller_incremental:
            # Full-render mode falls back to existing left-truncation in
            # _preprocess_checklist_batch; no length check at env level.
            return True
        if self.caller_input_ids is None:
            return True

        cur_len = self.caller_input_ids.shape[-1]
        budget_needed = cur_len + self.max_response_length
        if budget_needed >= self.max_model_len:
            self._last_error_code = "LENGTH_EXCEEDED"
            self.state = ChecklistState.TERMINATED
            logger.info(
                f"[ChecklistEnv {self.uuid}] caller_input_ids length {cur_len} + "
                f"max_response_length {self.max_response_length} >= max_model_len "
                f"{self.max_model_len}; terminating episode (mirrors upstream sglang_rollout.py:976)."
            )
            return False
        return True

    # ============ end X.2 helpers ============

    def _build_info(self, turn_reward: float, done: bool) -> Dict[str, Any]:
        need_agent = {
            ChecklistState.AWAIT_CALLER: "caller",
            ChecklistState.AWAIT_SIMULATOR: "simulator",
            ChecklistState.TERMINATED: "none",
            ChecklistState.PENDING: "none",
            ChecklistState.PROCESSING_TOOLS: "none",
            ChecklistState.INTERACTING: "none",
        }[self.state]
        # Build simulator-prompt payload when needed.
        # Per-call cursor design (plan §3.5): emit a single dict for the call
        # at pending_sim_indices[cursor], embedding schema_str + examples_lines so
        # ToolSimulatorAgent can render a prompt byte-identical to mcp_tool's
        # httpx fallback (95% byte-identical with upstream fallback;§10.(g)).
        sim_payload = None
        if self.state == ChecklistState.AWAIT_SIMULATOR and self.pending_sim_indices:
            cursor = min(self.pending_sim_cursor, len(self.pending_sim_indices) - 1)
            cur_idx = self.pending_sim_indices[cursor]
            tc = self.pending_tool_calls[cur_idx]
            # schema_str — mirrors mcp_tool.py:577 path
            try:
                schema_dict = self.tool_executor._id_by_candidate_tools[self.uuid][tc.name]
                schema_str = self.tool_executor._sanitize_text_for_tokenizer(
                    json.dumps(schema_dict, ensure_ascii=True, indent=0)
                )
            except Exception:
                schema_str = "<schema unavailable>"
            # examples_lines — shared builder with mcp_tool httpx path
            try:
                tcr_map = self.tool_executor._id_by_tool_call_response[self.uuid]
                examples_lines = build_few_shot_examples_lines(
                    tcr_map, self.tool_executor._sanitize_text_for_tokenizer
                )
            except Exception:
                examples_lines = ["Previous tool calls and results (few-shot): <unavailable>"]
            # Validation-mode context (upstream style, minus caller_system_prompt
            # which is intentionally NOT injected — it is mostly chat-template
            # boilerplate that confuses the simulator).
            #   - candidate_tools:      list of full tool schemas for this sample
            #   - conversation_history: serialized self.messages[:-1] (excludes current
            #                            caller assistant turn already appended @ envs.py:343)
            #   - raw_caller_message:   sanitized caller_text from this turn (with <tool_call>)
            try:
                _candidate_tools = list(
                    self.tool_executor._id_by_candidate_tools.get(self.uuid, {}).values()
                )
            except Exception:
                _candidate_tools = []
            try:
                _history_text = serialize_conversation_history(
                    self.messages[:-1] if len(self.messages) >= 1 else []
                )
            except Exception:
                _history_text = ""

            sim_payload = {
                "name": tc.name,
                "arguments": tc.arguments,
                "schema_str": schema_str,
                "examples_lines": examples_lines,
                # ---- validation mode extras (consumed by ToolSimulatorAgent._build_prompt) ----
                "candidate_tools": _candidate_tools,
                "conversation_history": _history_text,
                "raw_caller_message": getattr(self, "_last_caller_text", None),
                "inject_candidate_tools":         getattr(self.tool_executor, "sim_inject_candidate_tools",         True),
                "inject_conversation_history":    getattr(self.tool_executor, "sim_inject_conversation_history",    True),
                "inject_raw_caller_message":      getattr(self.tool_executor, "sim_inject_raw_caller_message",      True),
            }
        return {
            "need_agent": need_agent,
            "state": self.state.value,
            "uuid": self.uuid,
            "is_action_valid": True,  # NOTE: hard-coded True to match upstream (no apply_invalid_action_penalty signal)
            "won": bool(done and sum(self.per_step_turn_rewards) > 0),
            "tool_failed": bool(self._last_tool_failed),
            "error_tool_call": self._last_error_code,  # parse-error code for wandb stats; not used in reward/loss
            "num_checklists": len(self.checklist_list),
            "max_num_turns": len(self.checklist_list[0]) if self.checklist_list else 1,
            "user_turn_idx": max(0, self.user_turns - 1) if done else self.user_turns,
            "simulator_payload": sim_payload,
            "assistant_turns": self.assistant_turns,
            "user_turns": self.user_turns,
            "data_source": "checklist",
            "tool_calling": float(len(self.messages and [m for m in self.messages if m.get("role") == "tool"])),
            # ---- wandb monitoring (X.1.2 patch follow-up): tool response length distribution ----
            # Per-step aggregates over all tool messages written this step.
            # Char count (not token); ChecklistEnv has no tokenizer in full-render mode.
            # If incremental mode (X.2) lands, switch to token count for precision.
            "tool_response_count": len(self._last_step_tool_lens),
            "tool_response_char_len_max": max(self._last_step_tool_lens) if self._last_step_tool_lens else 0,
            "tool_response_char_len_sum": sum(self._last_step_tool_lens),
            "tool_response_sources": ",".join(self._last_step_tool_sources) if self._last_step_tool_sources else "",
            # Source counters for fine-grained wandb breakdown (per-step)
            "tool_response_count_cache": sum(1 for s in self._last_step_tool_sources if s == "cache"),
            "tool_response_count_simulator": sum(1 for s in self._last_step_tool_sources if s == "simulator"),
            "tool_response_count_schema_error": sum(1 for s in self._last_step_tool_sources if s == "schema_error"),
            "tool_response_count_httpx": sum(1 for s in self._last_step_tool_sources if s == "httpx"),
            "tool_response_count_error_tool_call": sum(1 for s in self._last_step_tool_sources if s == "error_tool_call"),
            # ---- Hit-rate breakdown: orthogonal to source classifier above ----
            # cache_full_hit  = name + args 全部命中 cache (= source == "cache")
            # name_hit_args_miss = 名字命中但参数不同 → 走 schema/sim/httpx
            # no_hit          = 名字都不在 cache 里
            # error_tool_call (caller parse error) 计入 no_hit (False/False/0)
            "tool_response_count_cache_full_hit": sum(self._last_step_args_hits),
            "tool_response_count_name_hit_args_miss": sum(
                1 for nh, ah in zip(self._last_step_name_hits, self._last_step_args_hits) if nh and not ah
            ),
            "tool_response_count_no_hit": sum(1 for nh in self._last_step_name_hits if not nh),
            # Few-shot block size (per-call), aggregated this step
            "tool_response_few_shot_sum": sum(self._last_step_few_shot_counts),
            "tool_response_few_shot_max": max(self._last_step_few_shot_counts) if self._last_step_few_shot_counts else 0,
        }

    async def async_close(self) -> None:
        try:
            if self.iid:
                await self.interaction.finalize_interaction(self.iid)
        except Exception:
            pass

    # ---------- sync wrappers (for thread-pool use) ----------

    def reset(self, env_kwargs: Dict[str, Any]):
        return asyncio.get_event_loop().run_until_complete(self.async_reset(env_kwargs))

    def step(self, action: Any):
        return asyncio.get_event_loop().run_until_complete(self.async_step(action))

    def close(self):
        try:
            asyncio.get_event_loop().run_until_complete(self.async_close())
        except Exception:
            pass


# -------- batched wrapper --------

class ChecklistMultiProcessEnv:
    """Batch wrapper mirroring SearchMultiProcessEnv:
    ThreadPoolExecutor + single asyncio event loop drives all ChecklistEnv instances concurrently.
    tool_executor and interaction are shared singletons across the batch (they manage their own
    httpx connection pools + semaphores internally).
    """

    def __init__(
        self,
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        is_train: bool = True,
        env_config: Any = None,
        tokenizer: Any = None,
    ):
        self.env_num = env_num
        self.group_n = group_n
        self.batch_size = env_num * group_n
        self.is_train = is_train
        self.env_config = env_config
        # X.2 incremental: caller tokenizer for token-level state initialization.
        # Injected via env_kwargs["tokenizer"] in reset() so each ChecklistEnv can call
        # _init_caller_token_state. None means use_caller_incremental=False; if True
        # but tokenizer=None, ChecklistEnv.async_reset will raise.
        self._tokenizer = tokenizer

        checklist_cfg = getattr(env_config, "checklist", env_config)

        # ---- Build shared tool executor ----
        # Hydra → MCPChecklistTool config bridge. Defaults here MUST mirror the
        # `__init__` defaults in mcp_tool.py:467-570; otherwise behavior diverges
        # silently between the env path (this dict) and the verl yaml path
        # (examples/drmas_trainer/config/checklist_tool_config.yaml).
        # When adding a new config key to MCPChecklistTool, add a row here too —
        # `__init__`'s `config.get(..., default)` is the only fallback if not.
        tool_cfg_raw = dict(
            (k, v) for k, v in {
                "return_raw": True,
                "dataset_path": getattr(checklist_cfg, "dataset_path", None),
                "sglang_url": getattr(checklist_cfg, "tool_sglang_url", []),
                "sglang_model": getattr(checklist_cfg, "tool_sglang_model", None),
                "temperature": getattr(checklist_cfg, "tool_temperature", 0.6),
                "top_p": getattr(checklist_cfg, "tool_top_p", 0.8),
                "max_new_tokens": getattr(checklist_cfg, "tool_max_new_tokens", 8000),
                "max_tokens": getattr(checklist_cfg, "tool_max_tokens", 8000),
                "retry_attempts": getattr(checklist_cfg, "tool_retry_attempts", 1),
                "semaphore_size": getattr(checklist_cfg, "tool_semaphore_size", 100),
                "timeout": getattr(checklist_cfg, "tool_timeout", 120),
                # Optional API key for OpenAI-compatible httpx fallback (e.g. gpt-5).
                # When set, MCPChecklistTool adds Authorization: Bearer <key> header.
                "api_key": getattr(checklist_cfg, "tool_api_key", None),
                # ---- Phase 0/1/2 + cache + schema toggles (default True ⇒ simulator-only
                # validation, upstream style). Override with `+env.checklist.tool_disable_*=false`
                # to re-enable backend short-circuit (defense-in-depth ablation).
                "disable_phase0_index_check": getattr(checklist_cfg, "tool_disable_phase0_index_check", True),
                "disable_phase0_tool_check":  getattr(checklist_cfg, "tool_disable_phase0_tool_check",  True),
                "disable_phase0_param_check": getattr(checklist_cfg, "tool_disable_phase0_param_check", True),
                "disable_cache":              getattr(checklist_cfg, "tool_disable_cache",              True),
                "disable_schema":             getattr(checklist_cfg, "tool_disable_schema",             True),
                "passthrough_tool_call":      getattr(checklist_cfg, "tool_passthrough_tool_call",      False),
                # ---- Simulator prompt injection toggles ----
                "sim_inject_candidate_tools":         getattr(checklist_cfg, "tool_sim_inject_candidate_tools",         True),
                "sim_inject_conversation_history":    getattr(checklist_cfg, "tool_sim_inject_conversation_history",    True),
                "sim_inject_raw_caller_message":      getattr(checklist_cfg, "tool_sim_inject_raw_caller_message",      True),
            }.items() if v is not None
        )
        self.tool_executor = MCPChecklistTool(tool_cfg_raw)

        # Fail fast on silent mis-config: if tool_return_mode=="httpx" but no
        # tool-side sglang URL was provided, the tool path has nowhere to send
        # requests. Catch this at construction rather than at rollout time.
        _effective_mode = getattr(checklist_cfg, "tool_return_mode", "agent")
        if _effective_mode == "httpx" and not tool_cfg_raw.get("sglang_url"):
            raise ValueError(
                "tool_return_mode='httpx' requires env.checklist.tool_sglang_url "
                "to be set explicitly (no fallback to env.checklist.sglang_url). "
                "Set it, or switch to tool_return_mode='agent' (default) to route "
                "cache-miss tool responses to the co-trained Tool Simulator Agent."
            )

        # ---- Build shared interaction (judge) ----
        interaction_cfg = {
            "sglang_url": getattr(checklist_cfg, "sglang_url", []),
            "sglang_model": getattr(checklist_cfg, "sglang_model", None),
            "judge_backend": getattr(checklist_cfg, "judge_backend", "sglang"),
            "judge_api_key": getattr(checklist_cfg, "judge_api_key", None),
            "retry_times": getattr(checklist_cfg, "retry_times", 2),
            "semaphore_size": getattr(checklist_cfg, "semaphore_size", 500),
            "temperature": getattr(checklist_cfg, "temperature", 0.6),
            "top_p": getattr(checklist_cfg, "top_p", 0.8),
            "max_new_tokens": getattr(checklist_cfg, "max_new_tokens", 6000),
            "max_tokens": getattr(checklist_cfg, "max_tokens", 6000),
            "timeout": getattr(checklist_cfg, "timeout", 1200),
        }
        self.interaction = ChecklistInteraction(interaction_cfg)

        # ---- Per-slot ChecklistEnv ----
        self.envs = [
            ChecklistEnv(checklist_cfg, self.tool_executor, self.interaction)
            for _ in range(self.batch_size)
        ]

        max_workers = min(self.batch_size, 256)
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

    def _run_async(self, coro):
        return self._loop.run_until_complete(coro)

    @staticmethod
    async def _gather(coros):
        return await asyncio.gather(*coros)

    def get_caller_token_state(self):
        """Per-slot caller token state, for use_caller_incremental rollouts.

        Returns (caller_input_ids, caller_attention_mask), each a list of
        batch_size tensors in slot order. Padding slots hold None, which
        ToolCallerAgent._preprocess_checklist_batch_incremental treats as a
        zero-length placeholder.
        """
        return (
            [env.caller_input_ids for env in self.envs],
            [env.caller_attention_mask for env in self.envs],
        )

    def reset(self, kwargs: List[Dict]):
        if kwargs is None:
            kwargs = []
        if len(kwargs) > self.batch_size:
            raise ValueError(f"Got {len(kwargs)} kwarg dicts, batch_size={self.batch_size}")
        pad_n = self.batch_size - len(kwargs)
        dummy_kw = {"messages": [], "tools": [], "checklist_list": [], "all_messages": [], "uuid": "_pad_"}
        padded = list(kwargs) + [dummy_kw] * pad_n
        valid = [True] * len(kwargs) + [False] * pad_n

        # X.2 incremental: inject tokenizer into each env_kwargs so ChecklistEnv.async_reset
        # can call _init_caller_token_state when use_caller_incremental=True.
        if self._tokenizer is not None:
            for kw in padded:
                if isinstance(kw, dict):
                    kw["tokenizer"] = self._tokenizer

        tasks = [env.async_reset(kw) for env, kw in zip(self.envs, padded)]
        results = self._run_async(self._gather(tasks))

        obs_list, info_list = zip(*results)
        obs_list = [o for o, keep in zip(obs_list, valid) if keep]
        info_list = [i for i, keep in zip(info_list, valid) if keep]
        return list(obs_list), list(info_list)

    def step(self, actions: List[Any]):
        if len(actions) > self.batch_size:
            raise ValueError(f"Got {len(actions)} actions, batch_size={self.batch_size}")
        pad_n = self.batch_size - len(actions)
        padded = list(actions) + [""] * pad_n
        valid = [True] * len(actions) + [False] * pad_n

        tasks = [env.async_step(act) for env, act in zip(self.envs, padded)]
        results = self._run_async(self._gather(tasks))

        obs_list, reward_list, done_list, info_list = zip(*results)
        obs_list = [o for o, keep in zip(obs_list, valid) if keep]
        reward_list = [r for r, keep in zip(reward_list, valid) if keep]
        done_list = [d for d, keep in zip(done_list, valid) if keep]
        info_list = [i for i, keep in zip(info_list, valid) if keep]
        return list(obs_list), list(reward_list), list(done_list), list(info_list)

    def close(self):
        if getattr(self, "_closed", False):
            return
        try:
            self._run_async(self._gather([env.async_close() for env in self.envs]))
        except Exception:
            pass
        self._executor.shutdown(wait=True)
        self._loop.close()
        self._closed = True

    def __del__(self):
        self.close()


def build_checklist_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
    env_config: Any = None,
    tokenizer: Any = None,
):
    return ChecklistMultiProcessEnv(
        seed=seed, env_num=env_num, group_n=group_n, is_train=is_train, env_config=env_config,
        tokenizer=tokenizer,
    )
