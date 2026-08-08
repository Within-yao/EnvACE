# Copyright 2026 Nanyang Technological University (NTU), Singapore
# Licensed under the Apache License, Version 2.0.

"""Checklist orchestra — schedules Tool Caller + (optional) Tool Simulator
based on env state-machine's `need_agent` hint passed through env_obs.

env_obs contract (produced by ChecklistEnvironmentManager):
    obs["anchor"]          : list[B] of messages list                     — primary input to both agents
    obs["tools"]            : list[B] of tool schema lists (from dataset) — for chat_template
    obs["need_agent"]       : list[B] of 'caller' | 'simulator' | 'none'   — routing decision
    obs["simulator_payload"]: list[B] of list[{name, arguments, schema_str, ...}] | None
                                                                          — per-sample pending tool_calls
"""
from __future__ import annotations
from typing import Any, Dict, List, Tuple
import importlib
import numpy as np

from transformers import PreTrainedTokenizer
from verl import DataProto

from agent_system.agent.orchestra.base import BaseOrchestra


class ChecklistMultiAgentOrchestra(BaseOrchestra):
    CALLER = "Tool Caller Agent"
    SIMULATOR = "Tool Simulator Agent"

    def __init__(
        self,
        agent_ids: List[str],
        model_ids: List[str],
        agents_to_wg_mapping: Dict[str, str],
        tokenizers: Dict[str, PreTrainedTokenizer] = None,
        processors: Dict[str, Any] = None,
        config: Any = None,
    ):
        importlib.import_module("agent_system.agent.agents.checklist")

        super().__init__(
            agent_ids=agent_ids,
            model_ids=model_ids,
            agents_to_wg_mapping=agents_to_wg_mapping,
            tokenizers=tokenizers,
            processors=processors,
            config=config,
        )
        if self.CALLER not in self.agent_ids:
            raise ValueError(f"Checklist orchestra requires '{self.CALLER}' in agent_ids")
        self.has_simulator = self.SIMULATOR in self.agent_ids

    def run(
        self,
        gen_batch: DataProto,
        env_obs: Dict[str, Any],
        actor_rollout_wgs,
        active_masks: np.ndarray,
        step: int,
    ) -> Tuple[List[str], List[Dict]]:
        self.reset_buffer()
        B = len(env_obs["anchor"])

        need_agent = env_obs.get("need_agent")
        if need_agent is None:
            need_agent = ["caller"] * B
        need_arr = np.array(need_agent, dtype=object)

        text_actions: List[str] = [""] * B

        caller_mask = np.array([n == "caller" for n in need_arr]) & np.array(active_masks, dtype=bool)
        simulator_mask = np.array([n == "simulator" for n in need_arr]) & np.array(active_masks, dtype=bool)

        if caller_mask.any():
            caller_wg = actor_rollout_wgs[self.agents_to_wg_mapping[self.CALLER]]
            batch, caller_texts = self.agents[self.CALLER].call(
                gen_batch=gen_batch,
                env_obs=env_obs,
                team_context=[""] * B,
                actor_rollout_wg=caller_wg,
                agent_active_mask=caller_mask.astype(bool),
                step=step,
            )
            self.save_to_buffer(self.CALLER, batch)
            for i in np.where(caller_mask)[0]:
                text_actions[i] = caller_texts[i]

        if simulator_mask.any():
            if not self.has_simulator:
                for i in np.where(simulator_mask)[0]:
                    text_actions[i] = "[]"
            else:
                sim_wg = actor_rollout_wgs[self.agents_to_wg_mapping[self.SIMULATOR]]
                batch, sim_texts = self.agents[self.SIMULATOR].call(
                    gen_batch=gen_batch,
                    env_obs=env_obs,
                    team_context=[""] * B,
                    actor_rollout_wg=sim_wg,
                    agent_active_mask=simulator_mask.astype(bool),
                    step=step,
                )
                self.save_to_buffer(self.SIMULATOR, batch)
                for i in np.where(simulator_mask)[0]:
                    text_actions[i] = sim_texts[i]

        return text_actions, self.multiagent_batch_buffer
