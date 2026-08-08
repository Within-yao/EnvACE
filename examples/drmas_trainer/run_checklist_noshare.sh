#!/bin/bash
# EnvACE Checklist joint training launcher — NOSHARE variant
#
# Trains the Tool Caller Agent and the Tool Simulator Agent together.
# model_sharing=False → Caller and Simulator get independent actors (two FSDP shards, two worker groups).
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

ENVACE_ROOT=${ENVACE_ROOT:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"}
PROJECT_DIR=${PROJECT_DIR:-"$ENVACE_ROOT"}

if [ -f "$ENVACE_ROOT/.env" ]; then
    set -a
    source "$ENVACE_ROOT/.env"
    set +a
fi
DATA_PATH=${DATA_PATH:-"$PROJECT_DIR/data/rl"}
REFERENCE_MODEL_PATH=${REFERENCE_MODEL_PATH:-"$PROJECT_DIR/models/Qwen/Qwen3-8B"}
SIMULATOR_MODEL_PATH=${SIMULATOR_MODEL_PATH:-"$PROJECT_DIR/models/Qwen/Qwen3-8B"}

CALLER_MODE=${CALLER_MODE:-"train"}
if [ "$CALLER_MODE" = "frozen" ]; then
    FROZEN_AGENT_IDS='["Tool Caller Agent"]'
else
    FROZEN_AGENT_IDS='[]'
fi

JUDGE_BACKEND=${JUDGE_BACKEND:-"sglang"}
if [ "$JUDGE_BACKEND" = "openai" ]; then
    JUDGE_URL_LIST=${JUDGE_URL_LIST:?need JUDGE_URL_LIST='["<endpoint>/v1/chat/completions"]' when JUDGE_BACKEND=openai}
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
JUDGE_TIMEOUT=${JUDGE_TIMEOUT:-120}
JUDGE_RETRY=${JUDGE_RETRY:-5}
TOOL_RETRY=${TOOL_RETRY:-5}
TOOL_TIMEOUT=${TOOL_TIMEOUT:-$JUDGE_TIMEOUT}

TOOL_DATASET_PATH=${TOOL_DATASET_PATH:-"$DATA_PATH/checklist_annotated_rl_selected_by_stats_forverl.json"}
TRAIN_DATA=${TRAIN_DATA:-"$DATA_PATH/checklist_annotated_rl_selected_by_stats_forverl_train.parquet"}
VAL_DATA=${VAL_DATA:-"$DATA_PATH/checklist_annotated_rl_selected_by_stats_forverl_val_small100.parquet"}

algorithm=grpo
group_size=4
val_group_size=4
group_by_agent_id=True

agent_ids='["Tool Caller Agent","Tool Simulator Agent"]'
model_ids="[\"${REFERENCE_MODEL_PATH}\",\"${SIMULATOR_MODEL_PATH}\"]"
model_sharing=${model_sharing:-False}
orchestra_type=checklist

actor_optim_lr='[1e-6,1e-6]'
actor_ppo_micro_batch_size_per_gpu='[1,1]'

train_data_size=${train_data_size:-16}
val_data_size=${val_data_size:-16}
max_prompt_length=${max_prompt_length:-4000}
max_response_length=${max_response_length:-8000}
rollout_max_prompt_length=${rollout_max_prompt_length:-12000}
rollout_truncation=left
kl_loss_coef=0.0001
tensor_model_parallel_size=1
num_gpus_per_node=${num_gpus_per_node:-8}
trainer_nnodes=${trainer_nnodes:-2}
max_num_checklist=1

TOOL_RETURN_MODE=${TOOL_RETURN_MODE:-"agent"}
max_assistant_turns=30
max_user_turns=30
max_parallel_calls=20
max_micro_steps=200

USE_CALLER_INCREMENTAL=${USE_CALLER_INCREMENTAL:-"true"}

max_model_len=${max_model_len:-20000}


experiment_name=${experiment_name:-"drmas-checklist-bs${train_data_size}-n${group_size}-c${max_num_checklist}-${TOOL_RETURN_MODE}-noshare"}

mkdir -p "$ENVACE_ROOT/pipeline"
echo "$experiment_name" > "$ENVACE_ROOT/pipeline/.last_experiment.tmp" \
    && mv "$ENVACE_ROOT/pipeline/.last_experiment.tmp" "$ENVACE_ROOT/pipeline/.last_experiment"

export SIM_HTTPX_DUMP_DIR=${SIM_HTTPX_DUMP_DIR:-"$PROJECT_DIR/checkpoints/drmas-checklist/${experiment_name}/sim_httpx_dumps"}
mkdir -p "$SIM_HTTPX_DUMP_DIR"
echo "[run_checklist.sh] SIM_HTTPX_DUMP_DIR=$SIM_HTTPX_DUMP_DIR"

export ENABLE_CHECKLIST=1

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

