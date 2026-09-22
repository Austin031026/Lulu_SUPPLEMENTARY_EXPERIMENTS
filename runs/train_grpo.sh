#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$RUN_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"

args=(
  --model "${GRPO_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
  --train-data "${GRPO_TRAIN_DATA:-$ROOT_DIR/data/grpo/train_prompts.parquet}"
  --output-dir "${GRPO_OUTPUT_DIR:-${LULU_OUTPUT_ROOT:-$ROOT_DIR/outputs}/grpo}"
  --gpus "${GRPO_GPUS:-auto}"
  --global-batch-prompts "${GRPO_GLOBAL_BATCH_PROMPTS:-48}"
  --group-size "${GRPO_GROUP_SIZE:-8}"
  --global-epochs "${GRPO_GLOBAL_EPOCHS:-1.0}"
  --max-response-tokens "${GRPO_MAX_RESPONSE_TOKENS:-8192}"
  --max-prompt-tokens "${GRPO_MAX_PROMPT_TOKENS:-4096}"
  --lr "${GRPO_LR:-1e-6}"
  --ppo-epochs "${GRPO_PPO_EPOCHS:-1}"
  --dtype "${GRPO_DTYPE:-bfloat16}"
  --seed "${GRPO_SEED:-42}"
)
[[ -z "${GRPO_MAX_STEPS:-}" ]] || args+=(--max-steps "$GRPO_MAX_STEPS")
[[ "${GRPO_PLAN_ONLY:-0}" != 1 ]] || args+=(--plan-only)

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec "$PYTHON_BIN" -u "$ROOT_DIR/scripts/run_grpo.py" "${args[@]}" "$@"
