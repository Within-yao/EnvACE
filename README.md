<h1 align="center">EnvACE: Internalizing Environment Dynamics via World Rehearsal for Agentic RL</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2608.06197">
    <img src="https://img.shields.io/static/v1?label=arXiv&message=Paper&color=red" alt="arXiv Paper">
  </a>
  <a href="https://github.com/Within-yao/EnvACE">
    <img src="https://img.shields.io/badge/code-EnvACE-black?logo=github" alt="Code">
  </a>
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License">
  </a>
</p>

`EnvACE` is an **agentic RL** framework that removes the external environment from the training loop. A single shared policy alternates between two roles:

- **Act** — generate the next tool call / environment-facing action, conditioned on the interaction history.
- **Rehearse** — play the role of the environment and generate the response induced by that action.

Subsequent decisions are conditioned on the policy's *own* rehearsed responses, so the trajectory unfolds without querying a real environment or a separate simulator. Both roles are trained jointly with a **role-wise GRPO** objective (separate baseline per role, shared parameters), so the policy internalizes environment dynamics as an implicit world model.

At inference, that internalized world model enables **test-time scaling by private rehearsal**: run N imagined trajectories in parallel or sequentially, summarize them into a rehearsal memory, then use that memory to guide one committed execution in the real environment.

<p align="center"><em>Real Environment · External Simulator · <b>EnvACE World Rehearsal</b></em></p>

## 🚀 News

- [2026-08-07] We released the code and paper for EnvACE.

## Highlights (from the paper)

- **World-rehearsal training.** Environment response generation is a role of the acting policy, not a separate module.
- **Role-wise GRPO.** Per-role advantage baselines, shared policy parameters, jointly optimized end-to-end from task-success reward.
- **Test-time rehearsal.** Two modes — Parallel (independent attempts) and Sequential (each attempt sees prior rehearsals + revisions) — condensed into a rehearsal memory before a single external execution.
- **Reference results.** EnvACE-8B: **Overall 32.91** across BFCL-v4 / τ²-Bench / VitaBench; **TF1 46.78** on FinMCP-Bench — outperforming Simulator-8B, TOUCAN-7B, EnvScaler-8B, AWM-8B/14B, ScaleEnv-8B under the same open-source scale.

## Installation

```bash
conda create -n envace python==3.12 -y
conda activate envace

# EnvACE-specific pinned versions (torch 2.6.0 + cu124, sglang 0.4.6.post5,
# ray 2.49.2, transformers 4.53.2 — validated end-to-end).
pip install -r requirements_envace.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir
pip install -e .
```

The pinned versions matter: `ray` and `sglang` must match across every node in the cluster, otherwise `ray start` complains with `Version mismatch: cluster X vs client Y` and the trainer refuses to connect.

If you want to develop against other backends (npu, vLLM etc.), see the upstream `requirements.txt` / `requirements-npu.txt` / `requirements_sglang.txt` files — they're broader in scope.

## Data & model layout (not tracked in git)

```bash
# From repo root:
ln -s <path>/data   data      # contains data/rl/checklist_annotated_rl_selected_by_stats_forverl{.json,_train.parquet,_val*.parquet}
ln -s <path>/models models    # contains models/Qwen/Qwen3-{4B,8B}
cp <path>/.env .env           # JUDGE_API_KEY, TOOL_API_KEY, optional WANDB_API_KEY
```

Sizes are ~424 MB for the tool cache JSON, ~102 MB for the train parquet, ~8 GB for Qwen3-4B, ~16 GB for Qwen3-8B. Any path can be overridden via `TRAIN_DATA=... VAL_DATA=... TOOL_DATASET_PATH=... REFERENCE_MODEL_PATH=... SIMULATOR_MODEL_PATH=...`.

## Quick start (single-node smoke, 4B share)

Once the prerequisites are in place:

```bash
# Start a Qwen3-30B-A3B sglang judge on some node.
python -m sglang.launch_server \
  --model-path <A3B_path> --served-model-name Qwen/Qwen3-30B-A3B \
  --host 0.0.0.0 --port 10001 --tp 8 --mem-fraction-static 0.88

# Start Ray (`ray start --head` / `--address=<head>:6379`), then:
QWEN4B=$PWD/models/Qwen/Qwen3-4B
ENVACE_PY=$(command -v python3) \
JUDGE_HOST=<judge_ip> \
REFERENCE_MODEL_PATH=$QWEN4B SIMULATOR_MODEL_PATH=$QWEN4B \
use_kl_loss=True rollout_gpu_memory_utilization=0.6 \
val_temperature=1 val_top_p=1 test_freq=25 \
trainer_nnodes=2 num_gpus_per_node=8 train_data_size=16 \
max_model_len=16000 rollout_max_prompt_length=8000 max_response_length=6000 \
experiment_name=envace-checklist-share-4b \
bash examples/drmas_trainer/run_checklist_share.sh 2>&1 | tee /tmp/share_4b.log
```

`train_data_size` must be divisible by `trainer_nnodes × num_gpus_per_node` (16 GPUs ⇒ 16, 32, …). Per-step wall time is ~8m30s at 16 × H20; the full 470-step reference run is ~66 h.

## Citation

```bibtex
@article{envace2026,
  title  = {EnvACE: Internalizing Environment Dynamics via World Rehearsal for Agentic Reinforcement Learning},
  author = {Xu, Zishan and Yao, Zhiyuan and Chen, Yuxin and Guo, Yifu and Lu, Zhengxi and
            Lu, Yuquan and Huang, Jinyang and Xu, Yan and Wang, Yasheng and Zhang, Weinan and
            Zeng, Xingshan and Liu, Weiwen},
  journal = {arXiv preprint arXiv:2608.06197},
  year   = {2026}
}
```

## Acknowledgement

Code builds on [Dr. MAS](https://github.com/langfengQ/DrMAS) (multi-agent RL orchestration), [verl-agent](https://github.com/langfengQ/verl-agent), and [verl](https://github.com/volcengine/verl). The A3B judge is [Qwen3-30B-A3B](https://huggingface.co/Qwen/Qwen3-30B-A3B); the actor backbone is [Qwen3-4B / 8B](https://huggingface.co/Qwen). Related environment-scaling baselines we compare against: Simulator, TOUCAN, EnvScaler, AWM, ScaleEnv (see the paper for citations).
