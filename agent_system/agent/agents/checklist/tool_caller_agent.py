# Copyright 2026 Nanyang Technological University (NTU), Singapore
# Licensed under the Apache License, Version 2.0.

"""Tool Caller Agent for the Checklist multi-agent RL pipeline.

Unlike Search/Math agents (which wrap a single-user observation with a system
prompt template), Tool Caller receives a full multi-turn conversation
(`env_obs['anchor'][i]` is the evolving messages list) and renders it via
`tokenizer.apply_chat_template(messages, tools=..., add_generation_prompt=True)`.
"""
import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

from agent_system.agent.agents.base import BaseAgent
from agent_system.agent.registry import AgentRegistry
from agent_system.multi_turn_rollout.utils import torch_to_numpy


def _preprocess_checklist_batch(
    gen_batch: DataProto,
    env_obs: Dict,
    config,
    tokenizer: PreTrainedTokenizer,
) -> DataProto:
    """Tokenize per-sample messages (list of dicts) + tools into model inputs.

    Returns a DataProto with standard keys (input_ids, attention_mask,
    position_ids, raw_prompt_ids, raw_prompt, data_source, anchor_obs, index).
    """
    anchor_list = env_obs.get("anchor")
    tools_list = env_obs.get("tools", None)
    if anchor_list is None:
        raise ValueError("ChecklistEnvironmentManager must provide env_obs['anchor']")

    batch_size = len(anchor_list)
    data_sources = gen_batch.non_tensor_batch.get(
        "data_source", np.array(["checklist"] * batch_size, dtype=object)
    )
    max_prompt_length = config.data.rollout_max_prompt_length
    truncation = config.data.rollout_truncation
    apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})

    processed_samples: List[Dict[str, Any]] = []
    for i in range(batch_size):
        messages = anchor_list[i]
        if tools_list is not None and i < len(tools_list):
            tools = tools_list[i]
        else:
            tools = gen_batch.non_tensor_batch.get("tools", [None] * batch_size)[i]

        prompt_str = tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
            **apply_chat_template_kwargs,
        )
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_str,
            tokenizer=tokenizer,
            max_length=max_prompt_length,
            pad_token_id=tokenizer.pad_token_id,
            left_pad=True,
            truncation=truncation,
        )
        position_ids = compute_position_id_with_mask(attention_mask)
        raw_prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)

        _pre_trunc_len = len(raw_prompt_ids)

        if _pre_trunc_len > max_prompt_length:
            logger.info(
                f"[caller _preprocess] sample {i}: prompt left-truncated "
                f"{_pre_trunc_len} → {max_prompt_length} (truncation={truncation})"
            )
            if truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-max_prompt_length:]
            elif truncation == "right":
                raw_prompt_ids = raw_prompt_ids[:max_prompt_length]
            elif truncation == "middle":
                lh = max_prompt_length // 2
                rh = max_prompt_length - lh
                raw_prompt_ids = raw_prompt_ids[:lh] + raw_prompt_ids[-rh:]
            elif truncation == "error":
                raise RuntimeError(
                    f"Checklist prompt length {_pre_trunc_len} > {max_prompt_length}"
                )

        processed_samples.append({
            "input_ids": input_ids[0],
            "attention_mask": attention_mask[0],
            "position_ids": position_ids[0],
            "raw_prompt_ids": raw_prompt_ids,
            "raw_prompt": list(messages),
            "data_source": str(data_sources[i]) if i < len(data_sources) else "checklist",
            "anchor_obs": list(messages),
            "index": i,
        })

    batched = collate_fn(processed_samples)
    return DataProto.from_single_dict(data=batched, meta_info=gen_batch.meta_info)


def _preprocess_checklist_batch_incremental(
    gen_batch: DataProto,
    env_obs: Dict,
    config,
    tokenizer: PreTrainedTokenizer,
    agent_active_mask: Optional[np.ndarray] = None,
) -> DataProto:
    """Read caller_input_ids from env_obs (maintained by ChecklistEnv via incremental
    BASE-prefix slicing, mirrors upstream schemas.py:385-475). Skip apply_chat_template
    entirely — token state is already byte-correct.

    Args:
        gen_batch: original PPO batch (carries meta_info, data_source, etc.)
        env_obs: must contain "caller_input_ids", "caller_attention_mask"
            each a List[B] of torch.LongTensor[1, L_i] (variable per sample, may be None for dummies).
            NOTE: env layer (envs.py:612, 741) still maintains a caller_loss_mask
            tensor for future mode B (episode-level PPO), but it's intentionally
            NOT exposed via env_manager.py obs dict and NOT pulled into DataProto
            here — see comment at the rows.append site below.
        config: must have config.data.rollout_max_prompt_length / rollout_truncation
        tokenizer: needed for pad_token_id

    Returns:
        DataProto with batch keys: input_ids, attention_mask, position_ids
        (left-padded to max_len_in_batch). non_tensor_batch keys: raw_prompt_ids,
        raw_prompt, data_source, anchor_obs, index.
    """
    import torch.nn.functional as F

    caller_ids_list = env_obs["caller_input_ids"]
    caller_mask_list = env_obs["caller_attention_mask"]
    anchor_list = env_obs["anchor"]

    B = len(caller_ids_list)
    if B == 0:
        raise ValueError("Empty batch in _preprocess_checklist_batch_incremental")

    max_prompt_length = config.data.rollout_max_prompt_length
    truncation = config.data.rollout_truncation
    pad_token_id = tokenizer.pad_token_id

    data_sources = gen_batch.non_tensor_batch.get(
        "data_source", np.array(["checklist"] * B, dtype=object)
    )

    truncated_ids: List[torch.Tensor] = []
    truncated_mask: List[torch.Tensor] = []

    for i in range(B):
        ids_i = caller_ids_list[i]
        mask_i = caller_mask_list[i]
        if ids_i is None:
            ids_i = torch.zeros((1, 0), dtype=torch.long)
            mask_i = torch.zeros((1, 0), dtype=torch.long)

        pre_len = ids_i.shape[-1]
        is_active = (
            bool(agent_active_mask[i]) if agent_active_mask is not None else True
        )

        if pre_len > max_prompt_length:
            if is_active:
                raise RuntimeError(
                    f"[caller incremental] sample {i}: caller_input_ids length {pre_len} > "
                    f"rollout_max_prompt_length {max_prompt_length}. Env should have terminated "
                    f"this episode (_check_caller_length_or_terminate). This is a bug — "
                    f"check max_model_len ({max_prompt_length} budget) and X.2.4 step 12 logic."
                )
            ids_i = ids_i[..., -max_prompt_length:]
            mask_i = mask_i[..., -max_prompt_length:]
        truncated_ids.append(ids_i)
        truncated_mask.append(mask_i)

    batch_max = max_prompt_length

    rows: List[Dict[str, Any]] = []
    for i in range(B):
        ids_i = truncated_ids[i]
        mask_i = truncated_mask[i]
        pad_len = batch_max - ids_i.shape[-1]
        if pad_len > 0:
            ids_padded = F.pad(ids_i, (pad_len, 0), value=pad_token_id)
            mask_padded = F.pad(mask_i, (pad_len, 0), value=0)
        else:
            ids_padded = ids_i
            mask_padded = mask_i

        position_ids = compute_position_id_with_mask(mask_padded)
        raw_prompt_ids = ids_i[0].tolist()

        rows.append({
            "input_ids": ids_padded[0],
            "attention_mask": mask_padded[0],
            "position_ids": position_ids[0],
            "raw_prompt_ids": raw_prompt_ids,
            "raw_prompt": list(anchor_list[i]),
            "data_source": str(data_sources[i]) if i < len(data_sources) else "checklist",
            "anchor_obs": list(anchor_list[i]),
            "index": i,
        })

    batched = collate_fn(rows)
    return DataProto.from_single_dict(data=batched, meta_info=gen_batch.meta_info)


CALLER_PROMPT_PLACEHOLDER = "{env_prompt}\n{team_context}\nstep={step}"


@AgentRegistry.register("Tool Caller Agent")
class ToolCallerAgent(BaseAgent):
    """Primary tool-using agent.

    Consumes the full multi-turn conversation from env_obs['anchor'] and emits
    assistant text (possibly containing <tool_call>...</tool_call> blocks).
    """

    def __init__(self, wg_id: str, tokenizer: PreTrainedTokenizer, processor, config: Any):
        super().__init__(
            "Tool Caller Agent",
            CALLER_PROMPT_PLACEHOLDER,
            wg_id=wg_id,
            tokenizer=tokenizer,
            processor=processor,
            config=config,
        )

    def call(
        self,
        gen_batch: DataProto,
        env_obs: Dict[str, Any],
        team_context: List[str],
        actor_rollout_wg,
        agent_active_mask,
        step: int,
    ) -> Tuple[DataProto, List[str]]:
        use_incremental = (
            env_obs.get("caller_input_ids") is not None
            and getattr(self.config.env.checklist, "use_caller_incremental", False)
        )
        if use_incremental:
            batch = _preprocess_checklist_batch_incremental(
                gen_batch=gen_batch,
                env_obs=env_obs,
                config=self.config,
                tokenizer=self.tokenizer,
                agent_active_mask=agent_active_mask,
            )
        else:
            batch = _preprocess_checklist_batch(
                gen_batch=gen_batch,
                env_obs=env_obs,
                config=self.config,
                tokenizer=self.tokenizer,
            )
        batch, text_responses = self._generate_with_llm(
            batch, actor_rollout_wg, agent_active_mask, gen_batch.meta_info
        )
        valids = np.array([1 if (t and t.strip()) else 0 for t in text_responses], dtype=np.int32)
        batch.non_tensor_batch["is_action_valid"] = valids
        batch.non_tensor_batch["env_step"] = np.array([step] * len(text_responses), dtype=object)
        return batch, text_responses
