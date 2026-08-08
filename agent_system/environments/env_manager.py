from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory, SearchMemory

import time

class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        self.memory.reset(batch_size=len(obs))

        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy()
        }
        
        return observations, infos

    def step(self, text_actions: List[str]):
        if not self.config.agent.multi_agent:
            actions, valids = self.projection_f(text_actions)
        else:
            actions = text_actions

        time1 = time.time()
        next_obs, rewards, dones, infos = self.envs.step(actions)
        time2 = time.time()
        print(f"SearchEnv step time: {time2 - time1:.4f} seconds")

        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy()
        }
        
        if not self.config.agent.multi_agent:
            for i, info in enumerate(infos):
                info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        for i in range(len(text_obs)):
            if init or self.config.env.history_length <= 0:
                if self.config.agent.multi_agent:
                    obs_i = SEARCH_MULTIAGENT_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i]
                    )
                else:
                    obs_i = SEARCH_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i]
                    )
            else:
                if self.config.agent.multi_agent:
                    obs_i = SEARCH_MULTIAGENT_TEMPLATE.format(
                        task_description=self.tasks[i],
                        memory_context="{memory}" if self.config.agent.use_agent_memory else memory_ctx[i],
                        step_count=len(self.memory[i]),
                    )
                else:
                    obs_i = SEARCH_TEMPLATE.format(
                        task_description=self.tasks[i],
                        memory_context=memory_ctx[i],
                        step_count=len(self.memory[i]),
                    )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return
            


class MathEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for MathEnv.
    """
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        observations = {
            "text": self.build_text_obs(obs),
            "image": None,
            "anchor": obs.copy()
        }
        
        return observations, infos

    def step(self, text_actions: List[str]):
        if not self.config.agent.multi_agent:
            actions, valids = self.projection_f(text_actions)
        else:
            actions = text_actions

        time1 = time.time()
        next_obs, rewards, dones, infos = self.envs.step(actions)
        time2 = time.time()
        print(f"MathEnv step time: {time2 - time1:.4f} seconds")

        next_observations = {
            "text": None,
            "image": None,
            "anchor": None
        }
        
        if not self.config.agent.multi_agent:
            for i, info in enumerate(infos):
                info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        for i in range(len(text_obs)):
            if self.config.agent.multi_agent:
                obs_i = MATH_MULTIAGENT_TEMPLATE.format(
                    task_description=self.tasks[i]
                )
            else:
                obs_i = MATH_TEMPLATE.format(
                    task_description=self.tasks[i]
                )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return

class ChecklistEnvironmentManager(EnvironmentManagerBase):
    """EnvironmentManager for the Checklist tool-use task.

    Each underlying ChecklistEnv is a state machine (see env_package.checklist.envs).
    This manager:
      * routes `env_kwargs` (messages/tools/checklist_list/all_messages/uuid) into the batched env at reset
      * passes text_actions straight to env.step (the env parses internally based on its current state)
      * derives the per-agent 'text' prompt from the evolving `messages` via tokenizer.apply_chat_template
    """

    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
        self._last_tools: List[Any] = []

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        self._last_tools = []
        if kwargs is not None:
            self._last_tools = [k.get("tools", []) if isinstance(k, dict) else [] for k in kwargs]
        obs_list, info_list = self.envs.reset(kwargs=kwargs)
        observations = self._build_obs(obs_list, info_list)
        return observations, info_list

    def step(self, text_actions: List[str]):
        next_obs_list, rewards, dones, infos = self.envs.step(text_actions)
        observations = self._build_obs(next_obs_list, infos)
        for info in infos:
            info.setdefault("is_action_valid", True)
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)
        return observations, rewards, dones, infos

    def _build_obs(self, messages_list: List[List[Dict]], infos: List[Dict]) -> Dict[str, Any]:
        B = len(messages_list)
        need_agent = [info.get("need_agent", "caller") for info in infos]
        sim_payload = [info.get("simulator_payload") for info in infos]
        tools = list(self._last_tools)
        if len(tools) < B:
            tools = tools + [[]] * (B - len(tools))
        elif len(tools) > B:
            tools = tools[:B]
        obs = {
            "text": None,
            "image": None,
            "anchor": [list(m) for m in messages_list],
            "tools": tools,
            "need_agent": need_agent,
            "simulator_payload": sim_payload,
        }
        if getattr(self.config.env.checklist, "use_caller_incremental", False):
            caller_ids, caller_mask = self.envs.get_caller_token_state()
            obs["caller_input_ids"] = caller_ids[:B]
            obs["caller_attention_mask"] = caller_mask[:B]
        return obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info.get('won', 0.0))
                success['success_rate'].append(won_value)
                data_source = info.get("data_source", "checklist")
                success[f"{data_source}_success_rate"].append(won_value)
                return


def make_envs(config, caller_tokenizer=None):
    """
    Create enviroments

    Args:
        config: full PPO config
        caller_tokenizer: tokenizer of the caller agent (X.2 incremental). When
            config.env.checklist.use_caller_incremental=True, ChecklistEnv needs
            this to compute BASE_CHAT_HISTORY token positions and render the
            initial caller_input_ids. None means caller increment will fail at
            first reset if the flag is True.
    """
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    val_group_n = getattr(config.env.rollout, 'val_n', 1)

    if "checklist" in config.env.env_name.lower():
        from agent_system.environments.env_package.checklist import build_checklist_envs, checklist_projection
        _envs = build_checklist_envs(seed=config.env.seed, env_num=config.data.train_batch_size,
                                     group_n=group_n, is_train=True, env_config=config.env,
                                     tokenizer=caller_tokenizer)
        _val_envs = build_checklist_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size,
                                         group_n=val_group_n, is_train=False, env_config=config.env,
                                         tokenizer=caller_tokenizer)
        projection_f = partial(checklist_projection)
        envs = ChecklistEnvironmentManager(_envs, projection_f, config)
        val_envs = ChecklistEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "math" in config.env.env_name.lower():
        from agent_system.environments.env_package.math import build_math_envs, math_projection
        _envs = build_math_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True)
        _val_envs = build_math_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=val_group_n, is_train=False)
        
        projection_f = partial(math_projection)
        envs = MathEnvironmentManager(_envs, projection_f, config)
        val_envs = MathEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "search" in config.env.env_name.lower():
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=val_group_n, is_train=False, env_config=config.env)

        projection_f = partial(search_projection)
        envs = SearchEnvironmentManager(_envs, projection_f, config)
        val_envs = SearchEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)
