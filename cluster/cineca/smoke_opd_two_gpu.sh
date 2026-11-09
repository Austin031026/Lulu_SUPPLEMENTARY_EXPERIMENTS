#!/usr/bin/env bash
set -euo pipefail
: "${ARCQ_ROOT:?Set ARCQ_ROOT to the dedicated project workspace}"
source "$ARCQ_ROOT/env/arcq_env.sh"

CODE="$ARCQ_ROOT/tmp/opd_opsd_validation"
DATA="$ARCQ_ROOT/data/diagnostic/two_gpu_smoke/train.jsonl"
OUTPUT="$ARCQ_ROOT/experiments/smoke_opd_2gpu_20261109_a"
cd "$CODE"

python -u scripts/train_lulu.py \
  --model "$ARCQ_ROOT/models/Qwen3-1.7B" \
  --teacher-model "$ARCQ_ROOT/models/Qwen3-32B" \
  --train-data "$DATA" --output-dir "$OUTPUT" \
  --method vanilla_opd --backend staged --gpus 0,1 \
  --teacher-gpus-per-worker 2 --teacher-tp-mode eager-local \
  --rounds 1 --global-batch-prompts 2 --rollouts-per-prompt 1 \
  --update-passes 1 --rollout-backend hf --rollout-batch-size 1 \
  --score-batch-size 1 --train-micro-batch-size 1 \
  --max-new-tokens 64 --max-prompt-tokens 512 --max-sequence-tokens 1024 \
  --temperature 0.6 --top-p 0.95 --rollout-top-k 20 \
  --lora-rank 0 --master-weights-fp32 --dtype bfloat16 \
  --gradient-checkpointing --seed 42

test -f "$OUTPUT/checkpoints/round_0001/lulu_state.json"
printf 'OPD_SMOKE_PASS=%s\n' "$OUTPUT"
