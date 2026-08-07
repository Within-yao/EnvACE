#!/bin/bash
# EnvACE Checklist joint training launcher — SHARE variant
#
# Trains the Tool Caller Agent and the Tool Simulator Agent together.
# model_sharing=True → Caller and Simulator share one actor (one FSDP shard, one worker group).
#
# Prerequisites:
#   1. A Qwen3-30B-A3B sglang judge already serving on $JUDGE_HOST:$JUDGE_PORT.
#   2. MCPChecklistTool dataset (json) at TOOL_DATASET_PATH, readable from every training node.
#   3. Caller / Simulator init checkpoints at REFERENCE_MODEL_PATH / SIMULATOR_MODEL_PATH.
#   4. Tool responses come from the co-trained Tool Simulator Agent (TOOL_RETURN_MODE=agent).
#   5. A Ray cluster is already up (ray start --head / --address=...).

set -x

MODE=${1:-train}
if [ "$MODE" == "eval" ] || [ "$MODE" == "evaluation" ]; then
    echo "Running in evaluation mode"
    VAL_ONLY=True
else
    echo "Running in training mode"
    VAL_ONLY=False
fi

# Resolve to the repo root (…/EnvACE) regardless of caller cwd.
ENVACE_ROOT=${ENVACE_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
PROJECT_DIR=${PROJECT_DIR:-"$ENVACE_ROOT"}

# Auto-load .env (project root) so JUDGE_API_KEY etc. don't need to be exported manually.
# Variables already set in the environment take precedence (set -a only exports new ones).
if [ -f "$ENVACE_ROOT/.env" ]; then
    set -a
    # shellcheck disable=SC1090,SC1091
    source "$ENVACE_ROOT/.env"
    set +a
fi
DATA_PATH=${DATA_PATH:-"$PROJECT_DIR/data/rl"}
# Caller and Simulator both start from the raw base model, so the two differ only
# by the RL signal (no SFT init on either side — keeps the ablation clean).
REFERENCE_MODEL_PATH=${REFERENCE_MODEL_PATH:-"$PROJECT_DIR/models/Qwen/Qwen3-8B"}
SIMULATOR_MODEL_PATH=${SIMULATOR_MODEL_PATH:-"$PROJECT_DIR/models/Qwen/Qwen3-8B"}

# ---- Caller training mode switch ----
# Because the two agents share weights, agent.frozen_agent_ids freezes the whole
# worker group -- CALLER_MODE=frozen is therefore only meaningful in the noshare
# variant.
# CALLER_MODE=train (default): both Caller and Simulator are trained.
# CALLER_MODE=frozen: Caller only does rollout, no gradient update; only Simulator is trained.
CALLER_MODE=${CALLER_MODE:-"train"}
if [ "$CALLER_MODE" = "frozen" ]; then
    FROZEN_AGENT_IDS='["Tool Caller Agent"]'
else
    FROZEN_AGENT_IDS='[]'
fi

# ---- Judge backend ----
# Joint training uses the dedicated in-cluster Qwen3-30B-A3B sglang judge.
# Client-side concurrency / timeout / retry mirror upstream traj_n48.sh:65-67.
# JUDGE_TIMEOUT is deliberately 20min rather than 60s: a PPO step cannot close
# until every trajectory's reward is in, so a slow request is far cheaper than a
# cancelled-and-resent one queueing behind the same sglang server.
# Set JUDGE_BACKEND=openai to fall back to a remote OpenAI-compatible API.
JUDGE_BACKEND=${JUDGE_BACKEND:-"sglang"}
if [ "$JUDGE_BACKEND" = "openai" ]; then
    JUDGE_URL_LIST=${JUDGE_URL_LIST:-"[\"https://az.gptplus5.com/v1/chat/completions\"]"}
    LLM_AS_A_JUDGE_NAME=${LLM_AS_A_JUDGE_NAME:-"gpt-4.1"}
    JUDGE_API_KEY=${JUDGE_API_KEY:?need JUDGE_API_KEY (set in $ENVACE_ROOT/.env or export before running)}
    JUDGE_SEMAPHORE=${JUDGE_SEMAPHORE:-8}
else
    : "${JUDGE_HOST:?need JUDGE_HOST=<sglang judge host> when JUDGE_BACKEND=sglang}"
    JUDGE_PORT=${JUDGE_PORT:-10001}
    JUDGE_URL_LIST=${JUDGE_URL_LIST:-"[\"http://${JUDGE_HOST}:${JUDGE_PORT}/v1/chat/completions\"]"}
    LLM_AS_A_JUDGE_NAME=${LLM_AS_A_JUDGE_NAME:-"Qwen/Qwen3-30B-A3B"}
    JUDGE_API_KEY=""
    JUDGE_SEMAPHORE=${JUDGE_SEMAPHORE:-500}
fi
# Judge / tool call timeout (seconds) and retry count for the HTTP calls.
# Defaults tuned for remote OpenAI-compatible APIs whose under-load latency was
# measured at 14-27s per call (vs <2s at low load) and occasionally hangs / 502s:
#   - timeout=120s leaves ~4x headroom over the observed 27s tail; tolerates
#     gpt-5 long-tail latency and lets connections that are merely slow (not
#     dead) finish without burning a retry slot
#   - retry_times=5 with exponential backoff (1+2+4+8+16=31s) absorbs transient 502s
# TOOL_RETRY / TOOL_TIMEOUT mirror JUDGE_RETRY / JUDGE_TIMEOUT since tool calls
# hit the same gateway and share failure modes; previously tool_timeout was
# hard-coded to 120 with only JUDGE_TIMEOUT parameterized (asymmetric).
JUDGE_TIMEOUT=${JUDGE_TIMEOUT:-120}
JUDGE_RETRY=${JUDGE_RETRY:-5}
TOOL_RETRY=${TOOL_RETRY:-5}
TOOL_TIMEOUT=${TOOL_TIMEOUT:-$JUDGE_TIMEOUT}

TOOL_DATASET_PATH=${TOOL_DATASET_PATH:-"$DATA_PATH/checklist_annotated_rl_selected_by_stats_forverl.json"}
TRAIN_DATA=${TRAIN_DATA:-"$DATA_PATH/checklist_annotated_rl_selected_by_stats_forverl_train.parquet"}
VAL_DATA=${VAL_DATA:-"$DATA_PATH/checklist_annotated_rl_selected_by_stats_forverl_val_small100.parquet"}

###################### Algorithm Configurations #################
algorithm=grpo
group_size=4
val_group_size=4
group_by_agent_id=True  # per-agent baseline (role-wise GRPO)

##################### Agent Configurations #####################
agent_ids='["Tool Caller Agent","Tool Simulator Agent"]'
model_ids="[\"${REFERENCE_MODEL_PATH}\",\"${SIMULATOR_MODEL_PATH}\"]"
model_sharing=${model_sharing:-True}
orchestra_type=checklist

# Per-agent parameter override (only supports actor_rollout_ref)
actor_optim_lr='[1e-6,1e-6]'
actor_ppo_micro_batch_size_per_gpu='[1,1]'

##################### Data / Training Configurations #####################
train_data_size=${train_data_size:-16}
val_data_size=${val_data_size:-16}
max_prompt_length=${max_prompt_length:-4000}              # 只用于 dataset 初始过滤 (filter_overlong_prompts)
max_response_length=${max_response_length:-8000}
rollout_max_prompt_length=${rollout_max_prompt_length:-12000}     # = sglang context - tool_max_new_tokens
rollout_truncation=left
kl_loss_coef=0.0001
tensor_model_parallel_size=1       # 对齐 run_math.sh / run_search.sh
                                   # 上游的 TP=4 是 4 卡下 8B 塞不下的被迫选择;
                                   # H20 96GB 单卡能装 8B FSDP shard, TP=1 让
                                   # 节点内 rollout replicas 从 4 个翻倍到 8 个
num_gpus_per_node=${num_gpus_per_node:-8}
trainer_nnodes=${trainer_nnodes:-2}    # 默认 2 节点 16 卡; wrapper 可覆盖为 1
max_num_checklist=1

##################### Checklist Env Configurations #####################
# "agent" (default): cache-miss tool responses come from the co-trained Tool
#   Simulator Agent, so no external sglang is consulted on the tool path.
# "httpx": call external sglang for tool simulation; tool_sglang_url AND
#   tool_sglang_model must be set explicitly (no fallback to judge server).
TOOL_RETURN_MODE=${TOOL_RETURN_MODE:-"agent"}
max_assistant_turns=30
max_user_turns=30
max_parallel_calls=20
# rollout_loop max micro-steps (state-machine transitions per episode).
# Per plan §五 (B′): AWAIT_SIMULATOR fires once per cache-miss tool_call,
# so each turn budget = 1 caller + N sim calls (was 1+1 batched).
# Budget: 30 turns × (1 caller + ~5 sim) ≈ 180; 200 leaves buffer.
max_micro_steps=200

# X.2 caller-side incremental token state (upstream-aligned).
# Default true (upstream-aligned incremental token accumulation).
# When true, ChecklistEnv maintains caller_input_ids accumulator and ToolCallerAgent
# skips per-turn apply_chat_template re-rendering. Mirrors upstream schemas.py:385-475.
# Set USE_CALLER_INCREMENTAL=false to fall back to the legacy full-render path.
USE_CALLER_INCREMENTAL=${USE_CALLER_INCREMENTAL:-"true"}

# X.2 step 12: total budget for caller prompt + response (upstream sglang_rollout.py:976-980 alignment).
# Used only when use_caller_incremental=True. When caller_input_ids hits this limit,
# episode terminates with error_code=LENGTH_EXCEEDED instead of left-truncating.
# Typical sglang context: 20000 (Qwen3-8B) or 32768 (longer ctx models).
max_model_len=${max_model_len:-20000}

# NOTE: simulator response char-truncation removed (was 256 char middle).
# Aligned with upstream sglang_rollout.py (rollout.mode=sync, what upstream checklist
# actually uses) which does NOT truncate single tool responses. The 256/middle
# default was inherited from upstream tool_agent_loop.py:457-464, but that path is
# never reached when rollout.mode=sync. rollout_max_prompt_length=12000
# left-truncation is the global safety net for caller prompt length.

experiment_name=${experiment_name:-"drmas-checklist-bs${train_data_size}-n${group_size}-c${max_num_checklist}-${TOOL_RETURN_MODE}-share"}

# Persist experiment_name so pipeline/run.sh step7 can locate the latest checkpoint
# without re-deriving the formula. Atomic via mv to avoid partial reads.
mkdir -p "$ENVACE_ROOT/pipeline"
echo "$experiment_name" > "$ENVACE_ROOT/pipeline/.last_experiment.tmp" \
    && mv "$ENVACE_ROOT/pipeline/.last_experiment.tmp" "$ENVACE_ROOT/pipeline/.last_experiment"

# Sim httpx I/O dump dir — for caller_only mode (where there's no Tool Simulator
# Agent rollout dump). Each Ray worker writes to {step}_{hostname}.jsonl with
# fcntl.flock + threading.Lock for cross-process / cross-thread safety.
# See agent_system/environments/env_package/checklist/mcp_tool.py:_dump_sim_httpx
export SIM_HTTPX_DUMP_DIR=${SIM_HTTPX_DUMP_DIR:-"$PROJECT_DIR/checkpoints/drmas-checklist/${experiment_name}/sim_httpx_dumps"}
mkdir -p "$SIM_HTTPX_DUMP_DIR"
echo "[run_checklist.sh] SIM_HTTPX_DUMP_DIR=$SIM_HTTPX_DUMP_DIR"

export ENABLE_CHECKLIST=1

# Force the EnvACE env Python (avoid PATH/conda-activate quirks in bash subshells).
# Override via ENVACE_PY=... bash ... if needed.
ENVACE_PY=${ENVACE_PY:-"$(command -v python3)"}

echo "[run_checklist.sh] before launch:"
echo "  ENVACE_PY        = $ENVACE_PY"
echo "  which python3   = $(which python3)"
echo "  CONDA_PREFIX    = $CONDA_PREFIX"
echo "  ENVACE_PY verl   = $($ENVACE_PY -c 'import verl; print(verl.__file__)' 2>&1 | tail -1)"
echo "  ENVACE_PY ray    = $($ENVACE_PY -c 'import ray; print(ray.__version__, ray.__file__)' 2>&1 | tail -1)"

"$ENVACE_PY" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$algorithm \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    algorithm.group_by_agent_id=$group_by_agent_id \
    data.train_files=$TRAIN_DATA \
    data.val_files=$VAL_DATA \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.prompt_key=messages \
    data.truncation='error' \
    +data.rollout_max_prompt_length=$rollout_max_prompt_length \
    +data.rollout_truncation=$rollout_truncation \
    +data.apply_chat_template_kwargs.enable_thinking=True \
    data.return_raw_chat=True \
    data.shuffle=True \
    data.custom_cls.path=${ENVACE_ROOT}/agent_system/dataset/checklist_dataset.py \
    data.custom_cls.name=ChecklistDataset \
    +data.max_num_checklist=${max_num_checklist} \
    reward_model.reward_manager=checklist \
    actor_rollout_ref.model.path=null \
    actor_rollout_ref.actor.optim.lr=null \
    +agent.agent_specific_parameters.actor.optim.lr=$actor_optim_lr \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_adaptive_ppo_mini_batch_size=True \
    actor_rollout_ref.actor.ppo_mini_update_num=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=null \
    +agent.agent_specific_parameters.actor.ppo_micro_batch_size_per_gpu=$actor_ppo_micro_batch_size_per_gpu \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss:-False} \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$tensor_model_parallel_size \
    actor_rollout_ref.rollout.name=${rollout_name:-sglang} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_memory_utilization:-0.5} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k:--1} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p:-0.95} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature:-0.6} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    env.env_name=checklist \
    env.seed=0 \
    env.max_steps=$max_micro_steps \
    env.rollout.n=$group_size \
    env.rollout.val_n=$val_group_size \
    +env.checklist.tool_return_mode=$TOOL_RETURN_MODE \
    +env.checklist.max_assistant_turns=$max_assistant_turns \
    +env.checklist.max_user_turns=$max_user_turns \
    +env.checklist.max_parallel_calls=$max_parallel_calls \
    +env.checklist.dataset_path=$TOOL_DATASET_PATH \
    +env.checklist.judge_backend=$JUDGE_BACKEND \
    +env.checklist.sglang_url=$JUDGE_URL_LIST \
    +env.checklist.sglang_model=$LLM_AS_A_JUDGE_NAME \
    +env.checklist.judge_api_key=$JUDGE_API_KEY \
    +env.checklist.retry_times=$JUDGE_RETRY \
    +env.checklist.semaphore_size=$JUDGE_SEMAPHORE \
    +env.checklist.timeout=$JUDGE_TIMEOUT \
    +env.checklist.temperature=0.6 \
    +env.checklist.top_p=0.8 \
    +env.checklist.max_new_tokens=6000 \
    +env.checklist.max_tokens=6000 \
    +env.checklist.tool_temperature=0.6 \
    +env.checklist.tool_top_p=0.8 \
    +env.checklist.tool_max_new_tokens=${tool_max_new_tokens:-8000} \
    +env.checklist.tool_max_tokens=${tool_max_tokens:-8000} \
    +env.checklist.tool_retry_attempts=$TOOL_RETRY \
    +env.checklist.tool_semaphore_size=8 \
    +env.checklist.tool_timeout=$TOOL_TIMEOUT \
    +env.checklist.use_caller_incremental=$USE_CALLER_INCREMENTAL \
    +env.checklist.max_model_len=$max_model_len \
    +env.checklist.max_response_length=$max_response_length \
    agent.agent_ids="$agent_ids" \
    agent.model_ids="$model_ids" \
    agent.model_sharing=$model_sharing \
    agent.frozen_agent_ids="$FROZEN_AGENT_IDS" \
    agent.orchestra_type=$orchestra_type \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='drmas-checklist' \
    trainer.experiment_name="$experiment_name" \
    trainer.n_gpus_per_node=$num_gpus_per_node \
    trainer.nnodes=$trainer_nnodes \
    trainer.save_freq=${save_freq:-50} \
    trainer.test_freq=${test_freq:-10} \
    trainer.val_only=$VAL_ONLY \
    trainer.val_before_train=False \
    trainer.total_epochs=1 \
    trainer.default_local_dir=${CKPT_DIR:-"$PROJECT_DIR/checkpoints/drmas-checklist/${experiment_name}"} \
    trainer.rollout_data_dir=${ROLLOUT_DUMP_DIR:-"$PROJECT_DIR/checkpoints/drmas-checklist/${experiment_name}/rollout_dumps"} \
    $@

