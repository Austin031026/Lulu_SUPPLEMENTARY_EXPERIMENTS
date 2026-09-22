#!/usr/bin/env bash
#SBATCH --job-name=lulu_grpo
#SBATCH --account=self_distil_llms
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=/mnt/vast/home/wy21quko/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/vast/home/wy21quko/slurm_logs/%x-%j.err

set -euo pipefail

# Override these with `sbatch --export=ALL,GRPO_CODE=...,GRPO_MODEL=...,GRPO_OUTPUT_DIR=...`.
CLUSTER_ROOT="${CLUSTER_ROOT:-/mnt/vast/home/wy21quko}"
GRPO_CODE="${GRPO_CODE:-$CLUSTER_ROOT/code/Rona_Lulu/Lulu}"
GRPO_MODEL="${GRPO_MODEL:-Qwen/Qwen3-1.7B}"
GRPO_TRAIN_DATA="${GRPO_TRAIN_DATA:-$GRPO_CODE/data/grpo/train_prompts.parquet}"
GRPO_OUTPUT_DIR="${GRPO_OUTPUT_DIR:-$CLUSTER_ROOT/results/grpo_qwen3_1p7b_full_thinking_seed42}"
PYTHON_BIN="${PYTHON_BIN:-$CLUSTER_ROOT/envs/lulu/bin/python}"

export GRPO_CODE GRPO_MODEL GRPO_TRAIN_DATA GRPO_OUTPUT_DIR PYTHON_BIN
export HF_HOME="${HF_HOME:-$CLUSTER_ROOT/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

mkdir -p "$CLUSTER_ROOT/slurm_logs" "$GRPO_OUTPUT_DIR" "$HF_HOME"
test -d "$GRPO_CODE"
test -f "$GRPO_TRAIN_DATA"
cd "$GRPO_CODE"

echo "code=$GRPO_CODE"
echo "model=$GRPO_MODEL"
echo "train_data=$GRPO_TRAIN_DATA"
echo "output=$GRPO_OUTPUT_DIR"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(); print("visible GPUs:", torch.cuda.device_count())'

# `auto` preserves the GPU allocation supplied by Slurm. The update runs on the
# first allocated GPU; rollout collection uses every allocated GPU.
GRPO_GPUS=auto bash runs/train_grpo.sh
