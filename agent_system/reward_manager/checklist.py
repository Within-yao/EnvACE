# Copyright 2026 Nanyang Technological University (NTU), Singapore
# Licensed under the Apache License, Version 2.0.

"""Checklist reward manager.

Equivalent to upstream final reward placement (but for this repo's multi-row batch
layout): take `episode_rewards` (already accumulated by rollout_loop as
Σ_t turn_reward_t), divide by num_checklists × max_num_turns, write at the
response's last valid token. Caller and Simulator rows each receive the same
traj-level score; GRPO `group_by_agent_id=True` handles per-agent baselining.
"""
from __future__ import annotations

import numpy as np
import torch

from verl import DataProto


class ChecklistEpisodeRewardManager:
    def __init__(self, tokenizers, num_examine, normalize_by_length=False) -> None:
        self.tokenizers = tokenizers
        self.num_examine = num_examine
        self.normalize_by_length = bool(normalize_by_length)  # accepted for signature parity

    def __call__(self, data: DataProto, return_dict: bool = False):
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        already_print = {}
        # Returned as reward_extra_info so the per-sample normalization factors
        # land in the rollout dump when trainer.rollout_data_dir is set (see
        # ray_trainer._dump_generations); without them the dumped reward is a
        # bare [0,1] number with no way to recover its absolute magnitude.
        max_num_turns_list: list[float] = []
        num_checklists_list: list[float] = []

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()

            agent_id = str(data_item.non_tensor_batch["agent_id"])
            wg_id = data_item.non_tensor_batch["wg_id"]
            data_source = data_item.non_tensor_batch.get("data_source", "checklist")

            episode_rewards = float(data_item.non_tensor_batch["episode_rewards"])
            num_checklists = float(data_item.non_tensor_batch.get("num_checklists", 1)) or 1.0
            max_num_turns = float(data_item.non_tensor_batch.get("max_num_turns", 1)) or 1.0
            max_num_turns_list.append(max_num_turns)
            num_checklists_list.append(num_checklists)

            # Core placement: the traj's total checklist-score, normalized.
            score = episode_rewards / num_checklists / max_num_turns

            if valid_response_length > 0:
                reward_tensor[i, valid_response_length - 1] = torch.tensor(
                    score, dtype=torch.float32, device=prompt_ids.device
                )

            tag = f"{data_source}:{agent_id}"
            if tag not in already_print:
                already_print[tag] = 0
            if already_print[tag] < self.num_examine and np.random.random() < 0.1:
                already_print[tag] += 1
                try:
                    response_ids = data_item.batch["responses"]
                    valid_response_ids = response_ids[:valid_response_length]
                    response_str = self.tokenizers[wg_id].decode(
                        valid_response_ids, skip_special_tokens=False
                    )
                    print(f"[{data_source}][{agent_id}][response] {response_str}")
                except Exception:
                    pass
                print(
                    f"[{data_source}][{agent_id}][score] episode={episode_rewards} "
                    f"num_checklists={num_checklists} max_num_turns={max_num_turns} → {score}"
                )

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": {
                    "max_num_turns": list(max_num_turns_list),
                    "num_checklists": list(num_checklists_list),
                },
            }
        return reward_tensor
