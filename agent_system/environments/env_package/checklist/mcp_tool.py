# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple, Optional
import httpx
from collections import defaultdict
import threading
import pickle
import hashlib
import copy
import random

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# ---------------------------------------------------------------------------
# Sim httpx I/O dump (caller_only mode — no Tool Simulator Agent rollout dump).
#
# Concurrency safety (3 layers):
#   1. Per-host file name → cross-node never collides.
#   2. threading.Lock     → in-process serialize (multiple Ray worker threads).
#   3. fcntl.flock        → cross-process serialize on overlayfs.
# Single-line ≤ PIPE_BUF (4 KB) is also POSIX-atomic; flock is belt-and-suspenders.
#
# Output structure: $SIM_HTTPX_DUMP_DIR/{step}_{hostname}.jsonl
# Compatible w/ verl rollout_data_dir reader (one JSON record per line).
# ---------------------------------------------------------------------------
import socket as _socket_dump
import fcntl as _fcntl_dump
import time as _time_dump

_DUMP_LOCK = threading.Lock()
_DUMP_HOSTNAME = _socket_dump.gethostname().replace('.', '_').replace(' ', '_')[:30]


def _dump_sim_httpx(payload, output_text, finish_reason, error):
    """Append sim httpx I/O to per-step per-host JSONL. Never raises."""
    dump_dir = os.environ.get("SIM_HTTPX_DUMP_DIR", "").strip()
    if not dump_dir:
        return
    try:
        step = os.environ.get("SIM_CURRENT_STEP", "0")
        msgs = payload.get("messages", []) if isinstance(payload, dict) else []
        sys_content = next((m.get("content", "") for m in msgs if m.get("role") == "system"), "")
        user_content = next((m.get("content", "") for m in msgs if m.get("role") == "user"), "")
        record = {
            "input": f"system\n{sys_content}\nuser\n{user_content}\nassistant\n",
            "output": output_text or "",
            "score": 0.0,
            "step": int(step) if str(step).isdigit() else step,
            "timestamp": _time_dump.strftime("%Y-%m-%d %H:%M:%S"),
            "finish_reason": finish_reason,
            "error": error,
            "model": payload.get("model") if isinstance(payload, dict) else None,
            "source": "sim_httpx",
            "hostname": _DUMP_HOSTNAME,
        }
        line = json.dumps(record, ensure_ascii=False) + "\n"
        os.makedirs(dump_dir, exist_ok=True)
        path = os.path.join(dump_dir, f"{step}_{_DUMP_HOSTNAME}.jsonl")
        with _DUMP_LOCK:
            with open(path, "a", encoding="utf-8") as f:
                try:
                    _fcntl_dump.flock(f.fileno(), _fcntl_dump.LOCK_EX)
                    f.write(line)
                    f.flush()
                finally:
                    _fcntl_dump.flock(f.fileno(), _fcntl_dump.LOCK_UN)
    except Exception:
        # Never let logging break training.
        pass


# Local replacement for verl.tools.schemas.ToolResponse (avoid dragging in the full
# verl tools stack + fastmcp + MCP client runtime).
@dataclass
class ToolResponse:
    text: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# Sentinel returned by execute() when cache misses, schema passes, AND the
# caller requested agent-mode simulation. ChecklistEnv catches this and routes
# the tool_call to the Tool Simulator Agent instead of sglang fallback.
NEED_SIMULATOR = object()


# Prompt literals + prompt builders live in simulate_core/ (stdlib-only, no
# dependency on this package). Import only what this module itself uses --
# envs.py and tool_simulator_agent.py import their own symbols from
# simulate_core directly rather than routing through here.
from simulate_core import (
    FALLBACK_SYSTEM_INSTRUCTION,
    build_few_shot_examples_lines,
    build_single_call_user_prompt,
    format_tool_error,
)




class MCPChecklistTool:
    """Ported from upstream `mcp_checklist_tool.py` with all MCP/verl dependencies stripped.

    Behavior (identical to upstream at the cache+schema+fallback layers):
      1. If the caller's parameters exactly match a cached (name, args) pair
         from the training dataset  → return the ground-truth cached response.
      2. Else, if schema validation fails → return a formatted error JSON.
      3. Else (cache miss + schema ok):
           - mode="httpx": call sglang fallback (few-shot prompt → simulated JSON)
           - mode="agent": return NEED_SIMULATOR sentinel so ChecklistEnv can route
             to the Tool Simulator Agent.
    """

    _dataset_cache: Dict[str, Dict[str, Any]] = {}
    _cache_lock = threading.Lock()

    def __init__(self, config: dict, tool_schema: Any = None):
        # tool_schema kept for signature compatibility; not used.
        self.name: str = config.get("name", "")  # default empty; set per-tool at call time
        self.config = config
        self.return_raw: bool = bool(config.get("return_raw", True))

        self._dataset_path = config.get("dataset_path", None)
        
        # 从缓存或新建数据
        cached_data = self._get_or_load_dataset_data(self._dataset_path)
        self._id_by_tool_call_response = cached_data["id_by_tool_call_response"]
        self._id_by_candidate_tools = cached_data["id_by_candidate_tools"]
        self._id_by_candidate_tools_name = cached_data["id_by_candidate_tools_name"]
        self._tool_by_name = cached_data["tool_by_name"]
        self._tools = cached_data["tools"]

        _raw_url = config.get("sglang_url", [])
        if _raw_url is None:
            self._sglang_url = []
        elif isinstance(_raw_url, str):
            self._sglang_url = _raw_url
        else:
            # OmegaConf ListConfig / tuple → plain list so downstream
            # isinstance(..., list) and random.choice both work.
            self._sglang_url = list(_raw_url)
        self._sglang_model = config.get("sglang_model", None)
        # self._system_instruction = config.get("system_instruction", None) or (
        #     "You are a precise tool executor that learns from examples.\n"
        #     "You will be given:\n"
        #     "- Tool call JSON Schema\n"
        #     "- Few-shot examples showing tool calls and their execution results\n"
        #     "- A new tool call with specific arguments\n"
        #     "\n"
        #     "Your task:\n"
        #     "1) Learn the OUTPUT FORMAT from the provided examples - follow the exact structure, data types, and response patterns\n"
        #     "2) Ensure FACTUAL CONSISTENCY - your output should align with the factual information demonstrated in the examples\n"
        #     "3) For the new tool call:\n"
        #     "   - Apply the learned format to the new arguments\n"
        #     "   - Maintain factual consistency with example patterns\n"
        #     "   - If arguments are similar to examples, adapt the example results appropriately\n"
        #     "   - If arguments are significantly different, generate new results following the learned format and factual patterns\n"
        #     "   - May need to fix some type or error in the examples\n"
        #     "4) Handle errors gracefully - if arguments are invalid or missing, return error messages in the same format as examples\n"
        #     "\n"
        #     "Critical constraints:\n"
        #     "- Act as a silent function executor - NO explanations, suggestions, or hints\n"
        #     "- NO guidance on how to fix errors or improve calls\n"
        #     "- NO references to examples or comparisons\n"
        #     "- Return ONLY the raw execution result as valid JSON\n"
        #     "- For errors, return minimal error information without instructional content\n"
        #     "\n"
        #     "Output requirements:\n"
        #     "- Return ONLY the execution result as valid JSON\n"
        #     "- No explanations, markdown, or code fences\n"
        #     "- Use correct JSON data types\n"
        #     "- Follow the exact output structure learned from examples\n"
        #     "- Maintain factual consistency with the example patterns\n"
        # )
        self._system_instruction = config.get("system_instruction", None) or FALLBACK_SYSTEM_INSTRUCTION

        self._temperature = config.get("temperature", 0.6)
        self._max_new_tokens = config.get("max_new_tokens", 2048)
        self._json_retry_attempts = config.get("retry_attempts", 1)
        self._top_p = config.get("top_p", 0.8)
        self._max_tokens = config.get("max_tokens", 2048)
        self._timeout = config.get("timeout", 120)
        try:
            self._semaphore_size = int(config.get("semaphore_size", 64))
        except Exception:
            self._semaphore_size = 64

        # Optional OpenAI-compatible API key. When set, the httpx fallback path adds
        # Authorization: Bearer <key> header so sglang_url can point at remote endpoints
        # like https://az.gptplus5.com/v1/chat/completions (gpt-5) without running a local
        # sglang server. None / empty string ⇒ no auth header (local sglang behavior).
        self._api_key: Optional[str] = config.get("api_key", None) or None

        # ---- Phase 0/1/2 关闭开关（默认全 True：simulator-only validation, upstream 风格） ----
        # 与 examples/drmas_trainer/config/checklist_tool_config.yaml 默认对齐:
        # validation system prompt 已经让 simulator 自己做 tool-whitelist + schema +
        # format 检查并产出 error_tool_call JSON, 在前端再跑 Phase 0/2 会把这些 case
        # 短路掉, 让 prompt 的 step 2/3 变死代码. 默认禁用让 simulator 端到端拿全权.
        # envs.py 构造 tool_cfg_raw 时不会塞这些字段(envs.py:1124-1140), 所以这些
        # __init__ 默认值就是 ChecklistEnv 路径的事实默认 — 与 yaml 路径形成对偶.
        self.disable_phase0_index_check = bool(config.get("disable_phase0_index_check", True))
        self.disable_phase0_tool_check  = bool(config.get("disable_phase0_tool_check",  True))
        self.disable_phase0_param_check = bool(config.get("disable_phase0_param_check", True))
        self.disable_cache              = bool(config.get("disable_cache",              True))
        self.disable_schema             = bool(config.get("disable_schema",             True))
        # 一开关 = 上面 5 个全 True；mcp_tool.execute 直接 NEED_SIMULATOR.
        # 默认仍 False: 上面 5 个独立默认已是 True, 不需要再用一刀切短路;
        # 若用户想绕开 simulator 直接返回 NEED_SIMULATOR 不带 name_hit/few-shot
        # 重新计算的语义, 可显式 set passthrough_tool_call=true.
        self.passthrough_tool_call      = bool(config.get("passthrough_tool_call",      False))

        # ---- Simulator prompt 注入开关 ----
        # caller's messages[0] is intentionally NOT a toggle: it's almost always
        # chat-template tool-registration boilerplate that misleads the simulator,
        # so we never inject it. Whitelist info comes from `sim_inject_candidate_tools`.
        self.sim_inject_candidate_tools         = bool(config.get("sim_inject_candidate_tools",         True))
        self.sim_inject_conversation_history    = bool(config.get("sim_inject_conversation_history",    True))
        self.sim_inject_raw_caller_message      = bool(config.get("sim_inject_raw_caller_message",      True))

        # Instance-level httpx client + semaphore: each MCPChecklistTool binds its
        # own client/semaphore to its owning env's event loop. Sharing these across
        # train/val ChecklistMultiProcessEnv instances caused cross-loop errors.
        try:
            timeout_value = float(self._timeout)
        except Exception:
            timeout_value = 120.0
        limits = httpx.Limits(
            max_connections=max(16, self._semaphore_size),
            max_keepalive_connections=max(8, self._semaphore_size // 2),
        )
        _client_headers: Dict[str, str] = {}
        if self._api_key:
            _client_headers["Authorization"] = f"Bearer {self._api_key}"
        self._client: httpx.AsyncClient = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout=timeout_value, read=timeout_value, write=timeout_value, connect=timeout_value),
            limits=limits,
            headers=_client_headers,
        )
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(self._semaphore_size)
    @classmethod
    def _get_or_load_dataset_data(cls, dataset_path: str) -> Dict[str, Any]:
        """获取或加载数据集数据，使用类级别缓存避免重复加载"""
        if dataset_path is None:
            raise ValueError("dataset_path cannot be None")
            
        # 使用线程锁确保线程安全
        with cls._cache_lock:
            if dataset_path in cls._dataset_cache:
                logger.info(f"Using cached dataset data for path: {dataset_path}")
                return cls._dataset_cache[dataset_path]
            
            logger.info(f"Loading and caching dataset data for path: {dataset_path}")
            
            # 可选：尝试从磁盘缓存加载（适合大数据集）
            disk_cache_data = cls._try_load_disk_cache(dataset_path)
            if disk_cache_data:
                logger.info(f"Loaded dataset from disk cache: {dataset_path}")
                cls._dataset_cache[dataset_path] = disk_cache_data
                return disk_cache_data
            
            # 加载数据
            data = cls._load_dataset_data(dataset_path)
            cls._dataset_cache[dataset_path] = data
            
            # 可选：保存到磁盘缓存
            cls._try_save_disk_cache(dataset_path, data)
                
            return data
    
    @staticmethod 
    def _load_dataset_data(dataset_path: str) -> Dict[str, Any]:
        """加载数据集并构建所需的数据结构"""
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        id_by_tool_call_response = {}
        id_by_candidate_tools = {}
        id_by_candidate_tools_name = {}
        tool_by_name = {}
        tools = []
        all_tools = []

        # 构建tool_call_response映射
        for item in data:
            tools = json.loads(item["extra_info"]["tools"])
            all_tools.extend(tools)
            tool_call_response_map = MCPChecklistTool._get_called_tools_and_response_static(item)
            id_by_tool_call_response[str(item["extra_info"]["original_index"])] = tool_call_response_map
        
        # 构建candidate_tools映射
        for item in data:
            tools = json.loads(item["extra_info"]["tools"])
            id_by_candidate_tools[str(item["extra_info"]["original_index"])] = {tool["function"]["name"]: tool for tool in tools}
            id_by_candidate_tools_name[str(item["extra_info"]["original_index"])] = [x["function"]["name"] for x in tools]

        # 构建tool映射(占位,保持 cache 结构兼容;execute 不使用 tool_by_name/tools)
        all_tools_name = set([x["function"]["name"] for x in all_tools])
        for name in all_tools_name:
            mcp_tool = {"name": name, "description": "", "inputSchema": {"type": "object", "properties": {}, "required": []}}
            tools.append(mcp_tool)
            tool_by_name[name] = mcp_tool

        return {
            "id_by_tool_call_response": id_by_tool_call_response,
            "id_by_candidate_tools": id_by_candidate_tools,
            "id_by_candidate_tools_name": id_by_candidate_tools_name,
            "tool_by_name": tool_by_name,
            "tools": tools
        }

    @staticmethod
    def _get_called_tools_and_response_static(item: Dict[str, Any]) -> List[Any]:
        """静态版本的get_called_tools_and_response方法，供数据加载使用"""
        if "extra_info" in item and "messages" in item["extra_info"]:
            messages = item["extra_info"]["messages"]
        else:
            raise ValueError(f"No messages found in item: {item}")
        
        results = defaultdict(list)
        if not isinstance(messages, list):
            raise ValueError(f"Unexpected messages format: {type(messages)}")

        for i in range(len(messages)):
            message = messages[i]
            if message.get("role") != "assistant":
                continue

            tool_calls = message.get("tool_calls") or []
            if not tool_calls:
                continue

            # Collect contiguous following tool messages for this assistant turn
            following_tool_msgs: List[Dict] = []
            
            for j in range(i+1, i+len(tool_calls)+1):
                assert messages[j].get("role") == "tool" or messages[j].get("role") == "observation", f"Unexpected role: {messages[j].get('role')}"
                following_tool_msgs.append(messages[j]["content"])

            for call, content in zip(tool_calls, following_tool_msgs, strict=True):
                results[call["function"]["name"]].append((json.loads(call["function"]["arguments"]), content))
        return results

    @staticmethod
    def _sanitize_text_for_tokenizer(text: Any) -> Any:
        """Replace Unicode code points that HuggingFace fast (Rust) tokenizers refuse to encode.

        Replaces (NOT drops) with U+FFFD REPLACEMENT CHARACTER so that:
        - downstream wandb char-length stats / reward judge / caller self-reflect can SEE that
          truncation happened (drop would silently shorten the string and hide the corruption);
        - cache-key lookups on tool_calls.arguments don't accidentally collide with a different
          legitimate arg string after silent shortening;
        - byte-level alignment with raw output is not required by any caller, so the small
          length change is benign.

        Covered code points (all rejected by Rust `tokenizers` encode_batch via PyO3):
        - U+D800..U+DFFF: lone UTF-16 surrogates (sglang mid-byte detokenize residue)
        - U+0000:        NUL byte
        - U+FDD0..U+FDEF: Unicode noncharacters block
        - *FFFE / *FFFF on every plane (U+FFFE/FFFF, U+1FFFE/1FFFF, ..., U+10FFFE/10FFFF)

        Accepts non-str inputs and returns them unchanged for convenience.
        """
        if not isinstance(text, str):
            return text
        replaced = False
        out_chars = []
        for ch in text:
            c = ord(ch)
            if (
                (0xD800 <= c <= 0xDFFF)
                or c == 0x00
                or (0xFDD0 <= c <= 0xFDEF)
                or ((c & 0xFFFE) == 0xFFFE)
            ):
                out_chars.append("�")
                replaced = True
            else:
                out_chars.append(ch)
        if replaced:
            try:
                logger.debug("[MCPChecklistTool] Replaced invalid Unicode code points with U+FFFD.")
            except Exception:
                pass
        return "".join(out_chars)

    @staticmethod
    def _deep_sanitize_for_tokenizer(obj: Any) -> Any:
        """Recursively apply `_sanitize_text_for_tokenizer` over arbitrary nested
        dict/list/tuple/str structures. Used for inputs that originate outside the
        LLM output entry points (e.g. dataset-supplied tool schemas, deserialized
        JSON args) and would otherwise reach `apply_chat_template` un-sanitized.
        """
        if isinstance(obj, str):
            return MCPChecklistTool._sanitize_text_for_tokenizer(obj)
        if isinstance(obj, dict):
            return {k: MCPChecklistTool._deep_sanitize_for_tokenizer(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [MCPChecklistTool._deep_sanitize_for_tokenizer(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(MCPChecklistTool._deep_sanitize_for_tokenizer(v) for v in obj)
        return obj
    
    def _format_error(self, code: str, message: str, details: Dict[str, Any] | None = None) -> str:
        """Thin wrapper around module-level `format_tool_error`. Kept for
        backward compat; new call sites should import `format_tool_error` directly."""
        return format_tool_error(code, message, details)

    def _validate_parameters_against_schema(self, original_index: str, parameters: Dict[str, Any], name: str = None) -> tuple[bool, List[str]]:
        """Lightweight validation of parameters against OpenAI-style tool schema from dataset.

        Checks:
        - required fields present
        - basic type conformity for primitive types
        - optional strict mode to disallow additional properties
        """
        errors: List[str] = []
        if name is None:
            name = self.name
        tool_schema: Dict[str, Any] = self._id_by_candidate_tools[original_index][name]

        fn = tool_schema.get("function", {}) if isinstance(tool_schema, dict) else {}
        params_schema = fn.get("parameters", {}) if isinstance(fn, dict) else {}

        if not isinstance(parameters, dict):
            return False, ["parameters must be a JSON object"]

        properties: Dict[str, Any] = params_schema.get("properties", {}) if isinstance(params_schema, dict) else {}
        required: List[str] = params_schema.get("required", []) if isinstance(params_schema, dict) else []
        strict: bool = bool(fn.get("strict", True))

        # required fields
        for key in required:
            if key not in parameters:
                errors.append(f"missing required field: {key}")

        # type checks (primitive only)
        def _matches_type(value: Any, expected: Any) -> bool:
            if isinstance(expected, list):
                return any(_matches_type(value, t) for t in expected)
            if expected == "string":
                return isinstance(value, str)
            if expected == "number":
                return (isinstance(value, (int, float)) and not isinstance(value, bool))
            if expected == "integer":
                return (isinstance(value, int) and not isinstance(value, bool))
            if expected == "boolean":
                return isinstance(value, bool)
            if expected == "null":
                return value is None
            if expected == "object":
                return isinstance(value, dict)
            if expected == "array":
                return isinstance(value, list)
            # unknown type keywords are treated as pass
            return True

        for key, value in parameters.items():
            if key not in properties:
                if strict:
                    errors.append(f"unexpected field not allowed: {key}")
                continue
            prop = properties.get(key, {})
            expected_type = prop.get("type")
            if expected_type is not None and not _matches_type(value, expected_type):
                errors.append(f"field '{key}' type mismatch: expected {expected_type}")
            # enum constraint
            if "enum" in prop:
                enum_values = prop.get("enum")
                try:
                    # allow int/float equivalence only if exactly equal (no bool)
                    if isinstance(value, bool):
                        in_enum = value in enum_values
                    else:
                        in_enum = value in enum_values
                except Exception:
                    in_enum = False
                if not in_enum:
                    errors.append(f"field '{key}' not in enum: {enum_values}")

        return len(errors) == 0, errors
    
    @staticmethod
    def _get_disk_cache_path(dataset_path: str) -> str:
        """生成磁盘缓存文件路径"""
        cache_dir = os.getenv("MCP_CACHE_DIR", "/tmp/mcp_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        path_hash = hashlib.md5(dataset_path.encode()).hexdigest()
        return os.path.join(cache_dir, f"dataset_{path_hash}.pkl")
    
    @staticmethod
    def _try_load_disk_cache(dataset_path: str) -> Dict[str, Any]:
        """尝试从磁盘缓存加载数据"""
        try:
            cache_path = MCPChecklistTool._get_disk_cache_path(dataset_path)
            if not os.path.exists(cache_path):
                return None
            
            # 检查缓存是否过期
            cache_mtime = os.path.getmtime(cache_path)
            dataset_mtime = os.path.getmtime(dataset_path)
            if cache_mtime < dataset_mtime:
                logger.info(f"Disk cache expired for {dataset_path}")
                return None
            
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load disk cache for {dataset_path}: {e}")
            return None
    
    @staticmethod  
    def _try_save_disk_cache(dataset_path: str, data: Dict[str, Any]):
        """尝试保存数据到磁盘缓存"""
        try:
            cache_path = MCPChecklistTool._get_disk_cache_path(dataset_path)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved dataset to disk cache: {dataset_path}")
        except Exception as e:
            logger.warning(f"Failed to save disk cache for {dataset_path}: {e}")
    async def execute(
        self,
        name: str,
        parameters: dict,
        original_index: str,
        mode: str = "httpx",
        **kwargs,
    ):
        """Three-phase tool execution matching upstream.

        Returns:
            ToolResponse or NEED_SIMULATOR sentinel, float (reward, always 0.0), dict (metrics).
        """
        original_index = str(original_index)
        # Aligned with upstream sglang_rollout.py:929-939 _make_error_response: when a
        # tool is unknown to this sample (missing_in_map / missing_in_kwargs in
        # upstream terms), emit caller-visible message byte-identical to upstream so the
        # caller LLM's RL learning signal matches. metrics["success"]=True
        # mirrors upstream (envs.py still detects "error_tool_call" key and terminates).
        # code/details preserved for debug.
        _TOOL_NOT_FOUND_MSG = (
            f"tool name '{name}' is not found. Please check your function call and use provided tools."
        )

        # -------- Passthrough short-circuit (≡ disable_phase0_* + disable_cache + disable_schema) --------
        if self.passthrough_tool_call:
            tcr_map = self._id_by_tool_call_response.get(original_index, {})
            return NEED_SIMULATOR, 0.0, {
                "success": True, "source": "need_simulator_passthrough",
                "name_hit": name in tcr_map,
                "args_hit": False,
                "few_shot_count": sum(len(v) for v in tcr_map.values()),
            }

        # -------- Phase 0: index/tool/param 闸门（每段独立可关） --------
        if not self.disable_phase0_index_check:
            if original_index not in self._id_by_candidate_tools_name:
                msg = self._format_error(
                    "INDEX_NOT_FOUND",
                    _TOOL_NOT_FOUND_MSG,
                    {"original_index": original_index},
                )
                logger.warning(f"[MCPTool] {msg}")
                return ToolResponse(text=msg), 0.0, {"success": True}
        if not self.disable_phase0_tool_check:
            candidate_names = self._id_by_candidate_tools_name.get(original_index, [])
            if name not in candidate_names:
                msg = self._format_error(
                    "TOOL_NOT_AVAILABLE",
                    _TOOL_NOT_FOUND_MSG,
                    {"tool": name, "original_index": original_index},
                )
                logger.warning(f"[MCPTool] {msg}")
                return ToolResponse(text=msg), 0.0, {"success": True}
        if not self.disable_phase0_index_check:
            if original_index not in self._id_by_tool_call_response:
                msg = self._format_error(
                    "INDEX_NOT_FOUND",
                    _TOOL_NOT_FOUND_MSG,
                    {"original_index": original_index},
                )
                logger.warning(f"[MCPTool] {msg}")
                return ToolResponse(text=msg), 0.0, {"success": True}

        if not self.disable_phase0_param_check:
            if not name or parameters is None or not isinstance(parameters, dict):
                msg = self._format_error(
                    "INVALID_PARAMETERS",
                    "'parameters' is missing, empty, or not a JSON object.",
                    {"tool": name, "parameters_type": type(parameters).__name__},
                )
                logger.warning(f"[MCPTool] {msg}")
                return ToolResponse(text=msg), 0.0, {"success": False}

        # -------- Phase 1: cache lookup --------
        tool_call_response_map = self._id_by_tool_call_response.get(original_index, {})
        # wandb hit-rate / few-shot signals — emitted in EVERY return below so envs.py can
        # distinguish (a) full hit (b) name hit only (c) no hit, and read few-shot size.
        # few_shot_count = total (args, content) pairs available for this sample (sums over
        # tool names); same map feeds build_few_shot_examples_lines for httpx/simulator paths.
        name_hit = name in tool_call_response_map
        few_shot_count = sum(len(v) for v in tool_call_response_map.values())
        def _canonicalize_parameters(parameters: Dict[str, Any]) -> Any:
            return json.dumps(parameters, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
        if not self.disable_cache and name_hit and isinstance(parameters, dict):
            tool_call_response = tool_call_response_map[name]
            for args, content in tool_call_response:
                if _canonicalize_parameters(args) == _canonicalize_parameters(parameters):
                    logger.info(f"Found cached match for tool {name} with arguments {parameters}")
                    safe_text = (
                        self._sanitize_text_for_tokenizer(content)
                        if isinstance(content, str)
                        else json.dumps(content, ensure_ascii=True)
                    )
                    return ToolResponse(text=safe_text), 0.0, {
                        "success": True, "source": "cache",
                        "name_hit": True, "args_hit": True, "few_shot_count": few_shot_count,
                    }

        # -------- Phase 2: schema validation --------
        if not self.disable_schema and isinstance(parameters, dict):
            sample_schemas = self._id_by_candidate_tools.get(original_index, {})
            if name in sample_schemas:
                ok, validation_errors = self._validate_parameters_against_schema(original_index, parameters, name=name)
                if not ok:
                    msg = self._format_error(
                        "SCHEMA_VALIDATION_FAILED",
                        "parameters do not conform to the tool schema",
                        {"errors": validation_errors, "tool": name, "original_index": original_index},
                    )
                    logger.info(f"[MCPTool] Schema validation failed: {validation_errors}")
                    return ToolResponse(text=msg), 0.0, {
                        "success": True, "source": "schema_error",
                        "name_hit": name_hit, "args_hit": False, "few_shot_count": few_shot_count,
                    }

        # -------- Phase 3: fallback --------
        if mode == "agent":
            # Defer to Tool Simulator Agent; ChecklistEnv handles this sentinel.
            return NEED_SIMULATOR, 0.0, {
                "success": True, "source": "need_simulator",
                "name_hit": name_hit, "args_hit": False, "few_shot_count": few_shot_count,
            }

        # mode == "httpx": call sglang fallback
        _schema_dict = self._id_by_candidate_tools.get(original_index, {}).get(name, {})
        schema_str = self._sanitize_text_for_tokenizer(
            json.dumps(_schema_dict, ensure_ascii=True, indent=0)
        )

        # Build few-shot examples + user prompt via shared module-level builders
        # (same path is used by Tool Simulator Agent — guarantees byte-identical
        # user prompt across httpx and simulator modes).
        examples_lines = build_few_shot_examples_lines(
            tool_call_response_map, self._sanitize_text_for_tokenizer
        )
        user_prompt = build_single_call_user_prompt(
            examples_lines=examples_lines,
            schema_str=schema_str,
            tool_name=name,
            arguments_obj=parameters,
            sanitize_fn=self._sanitize_text_for_tokenizer,
        )

        base_messages = [
            {"role": "system", "content": self._system_instruction},
            {"role": "user", "content": user_prompt},
        ]

        async def _single_attempt(attempt_idx: int) -> "tuple[int, tuple]":
            """One HTTP attempt to the tool simulator endpoint.

            Returns (attempt_idx, (text, finish_reason, api_failed)) tuple where:
              - text != None, api_failed=False: success path — caller parses JSON
              - text == None, api_failed=True : remote API layer failure (502/timeout/
                connect/ReadError) — caller should mark trajectory for masking and
                exponential-backoff retry
              - text == None, api_failed=False: other failure (very rare; e.g. unexpected
                exception inside the try block) — caller short-backoff retry without
                marking api_failed
            """
            payload = {
                "model": self._sglang_model,
                "messages": base_messages,
                "temperature": self._temperature,
                "max_new_tokens": self._max_new_tokens,
                "max_tokens": self._max_tokens,
                "top_p": self._top_p,
                # "json_schema": {
                #     "type": ["object", "array"]   # 允许对象或数组
                # },
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "tool_execution_response",
                        "schema": {
                            "type": "object",
                            "required": ["analysis", "execution_result"],
                            "additionalProperties": False,
                            "properties": {
                                "analysis": {
                                    "type": "string",
                                },
                                "execution_result": {
                                    "anyOf": [
                                        {
                                            "type": "object",
                                            "minProperties": 1,
                                            "additionalProperties": True
                                        },
                                        {
                                            "type": "array",
                                            "items": {}
                                        }
                                    ]
                                }
                            }
                        }
                    }
                },

                "sampling_params": {
                    "temperature": self._temperature,
                    "max_new_tokens": self._max_new_tokens,
                    "top_p": self._top_p,
                    "max_tokens": self._max_tokens,
                },


            }
            try:
                # 确保共享资源已初始化（惰性补充 + 健康检查）
                try:
                    timeout_value = float(self._timeout)
                except Exception:
                    timeout_value = 120.0
                limits = httpx.Limits(
                    max_connections=max(16, self._semaphore_size),
                    max_keepalive_connections=max(8, self._semaphore_size // 2),
                )
                # Preserve Authorization header on rebuilds (when api_key is set, e.g. gpt-5)
                _client_headers: Dict[str, str] = {}
                if self._api_key:
                    _client_headers["Authorization"] = f"Bearer {self._api_key}"
                client = self._client
                # 如果 client 缺失或已关闭，则重建
                if client is None or getattr(client, "is_closed", False):
                    if client is not None:
                        try:
                            await client.aclose()
                        except Exception:
                            pass
                    self._client = httpx.AsyncClient(
                        timeout=httpx.Timeout(timeout=timeout_value, read=timeout_value, write=timeout_value, connect=timeout_value),
                        limits=limits,
                        headers=_client_headers,
                    )
                    client = self._client

                selected_url = random.choice(self._sglang_url) if isinstance(self._sglang_url, list) and self._sglang_url else self._sglang_url
                # Network errors (ReadError/HTTPError/Timeout/ConnectError/etc.) are caught
                # at the outer try, marked api_failed=True, and the outer for-loop applies
                # exponential backoff before retrying. The previous single-shot 5s sleep +
                # client rebuild was insufficient for typical 502 recovery windows (5-30s).
                async with self._semaphore:
                    resp = await client.post(selected_url, json=payload)
                resp.raise_for_status()
                data = resp.json()
                choice = data.get("choices", [{}])[0]
                text = choice.get("message", {}).get("content", "")
                text = MCPChecklistTool._sanitize_text_for_tokenizer(text)
                finish_reason = choice.get("finish_reason", "unknown")
                _dump_sim_httpx(payload, text, finish_reason, None)
                return attempt_idx, (text, finish_reason, False, None)   # success
            except (httpx.HTTPStatusError, httpx.TimeoutException, httpx.ConnectError,
                    httpx.ReadError, httpx.HTTPError, httpx.RemoteProtocolError, RuntimeError) as e:  # type: ignore[attr-defined]
                # 远端基础设施层失败（502/503/504/timeout/connect/closed）— mark api_failed.
                # Force client rebuild on next attempt: closed sockets often produce
                # cascading failures on the same client.
                try:
                    if self._client is not None:
                        await self._client.aclose()
                except Exception:
                    pass
                self._client = None
                _dump_sim_httpx(payload, None, None, f"{type(e).__name__}: {e}")
                return attempt_idx, (None, None, True, f"{type(e).__name__}: {e}")
            except Exception as e:
                # 其他未预期异常（罕见，比如 attribute 错或 sanitize 异常）— 不算 API 故障，
                # 不让训练 mask 这条 trajectory，但也不消耗指数退避配额。
                _dump_sim_httpx(payload, None, None, f"{type(e).__name__}: {e}")
                return attempt_idx, (None, None, False, f"{type(e).__name__}: {e}")

        # 串行重试 + 指数退避 — mirrors checklist_reward._post_with_retries (judge 端)
        # 让 tool API 调用获得跟 judge 同等的 502/timeout 保护。
        #
        # 失败分两类，只影响退避节奏：
        #   - api_failed=True (远端 502/timeout/connect): 指数退避 1/2/4/8/16s (max 30s)
        #   - api_failed=False (response 解析坏 / 模型调错工具名 / 其他罕见异常):
        #     短退避 0.5s
        api_failed_seen = False
        fallback_text = None
        error_message = None
        try:
            for attempt_idx in range(self._json_retry_attempts):
                try:
                    _, (text, finish_reason, api_failed, attempt_err) = await _single_attempt(attempt_idx)
                    if api_failed:
                        api_failed_seen = True
                    if attempt_err is not None:
                        # 内层 _single_attempt 已经捕获并打包了异常信息（502/timeout/...）
                        error_message = attempt_err
                    if text is None:
                        # 失败 — 退避后重试（如果还有 retry 配额）
                        if attempt_idx + 1 < self._json_retry_attempts:
                            backoff = min(2 ** attempt_idx, 30) if api_failed else 0.5
                            await asyncio.sleep(backoff)
                        continue
                    # text != None — 尝试解析 JSON
                    parsed = json.loads(text)
                    tool_response = parsed['execution_result']
                    normalized = json.dumps(tool_response, ensure_ascii=True)
                    return ToolResponse(text=normalized), 0.0, {
                        "success": True,
                        "source": "httpx",
                        "name_hit": name_hit,
                        "args_hit": False,
                        "few_shot_count": few_shot_count,
                    }
                except json.JSONDecodeError as e:
                    error_message = f"{type(e).__name__}: {e}"
                    if attempt_idx + 1 < self._json_retry_attempts:
                        await asyncio.sleep(0.5)
                    continue
                except Exception as e:
                    error_message = f"{type(e).__name__}: {e}"
                    continue
        except Exception as e:
            logger.warning(f"Error during serial retry execution: {e}")

        # 所有尝试都失败了，返回错误信息
        if not fallback_text:
            fallback_text = json.dumps({"error": error_message or "unknown"}, ensure_ascii=True)

        logger.warning(
            f"Tool {name} execution failed after {self._json_retry_attempts} attempts; "
            f"api_failed={api_failed_seen}, last_error={error_message}"
        )
        return ToolResponse(text=fallback_text), 0.0, {
            "success": False,
            "source": "httpx",
            "name_hit": name_hit,
            "args_hit": False,
            "few_shot_count": few_shot_count,
        }
        

       

        




        