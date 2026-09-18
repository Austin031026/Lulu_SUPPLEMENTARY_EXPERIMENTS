#!/bin/bash
#SBATCH --job-name=lulu_ren_opd
#SBATCH --account=self_distil_llms
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/vast/home/wy21quko/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/vast/home/wy21quko/slurm_logs/%x-%j.err

set -euo pipefail

ROOT=/mnt/vast/home/wy21quko
CODE=$ROOT/code/Rona_Lulu/Lulu
DATA=$ROOT/data/lulu_dapo/train.jsonl
OUT=$ROOT/results/ren_opd_qwen3_1p7b_teacher32b_seed42

source "$ROOT/envs/lulu/bin/activate"
source "$CODE/runs/env_43.sh"

cd "$CODE"

mkdir -p \
  "$ROOT/slurm_logs" \
  "$ROOT/results" \
  "$ROOT/hf_cache" \
  "$ROOT/hf_cache/datasets"

test -f "$DATA"

echo "============================================================"
echo "LuLu ReN-OPD production"
echo "============================================================"
echo "date:                 $(date)"
echo "hostname:             $(hostname)"
echo "user:                 $USER"
echo "job id:               ${SLURM_JOB_ID:-unset}"
echo "node list:            ${SLURM_NODELIST:-unset}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "python:               $PYTHON"
echo "data:                 $DATA"
echo "output:               $OUT"
echo "HF_HOME:              $HF_HOME"
echo "============================================================"

"$PYTHON" - <<'PY'
import torch
import transformers
import peft

print("torch:", torch.__version__)
print("torch CUDA:", torch.version.cuda)
print("transformers:", transformers.__version__)
print("peft:", peft.__version__)
print("CUDA available:", torch.cuda.is_available())
print("visible GPUs:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 8
PY

nvidia-smi

RESUME_ARGS=()

if [[ -e "$OUT/checkpoints/latest" ]]; then
    echo "Existing checkpoint detected:"
    readlink -f "$OUT/checkpoints/latest" || true
    RESUME_ARGS+=(--resume)
else
    echo "No checkpoint detected. Starting fresh."
fi

"$PYTHON" -u scripts/train_lulu.py \
  --train-data "$DATA" \
  --output-dir "$OUT" \
  --model Qwen/Qwen3-1.7B \
  --teacher-model Qwen/Qwen3-32B \
  --method ren_opd \
  --backend persistent \
  --gpus 0,1,2,3,4,5,6,7 \
  --student-gpus 0,1,2,3,4 \
  --hindsight-gpus 5 \
  --teacher-gpus 6,7 \
  --global-batch-prompts 64 \
  --rollout-batch-size 4 \
  --score-batch-size 1 \
  --train-micro-batch-size 1 \
  --logit-chunk-size 32 \
  --rounds 100 \
  --max-new-tokens 8192 \
  --max-prompt-tokens 4096 \
  --max-sequence-tokens 16384 \
  --top-k 32 \
  --learning-rate 1e-5 \
  --weight-decay 0.0 \
  --max-grad-norm 1.0 \
  --lora-rank 16 \
  --lora-alpha 32 \
  --dtype bfloat16 \
  --save-every 20 \
  --seed 42 \
  "${RESUME_ARGS[@]}"

echo "============================================================"
echo "Training finished successfully"
echo "finished: $(date)"
echo "output:   $OUT"
echo "============================================================"