#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$RUN_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PYTHON:-python}}"
MODEL_FAMILY="${GRPO_MODEL_FAMILY:-qwen}"
if [[ -n "${GRPO_PROMPT_MODE:-}" ]]; then
  PROMPT_MODE="$GRPO_PROMPT_MODE"
elif [[ "$MODEL_FAMILY" == nemotron ]]; then
  PROMPT_MODE=nemotron-thinking
else
  PROMPT_MODE=qwen3-thinking
fi
if [[ "$MODEL_FAMILY" == nemotron ]]; then
  DEFAULT_MODEL=/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/.cache/huggingface/hub/models--nvidia--Llama-3.1-Nemotron-Nano-4B-v1.1/snapshots/d552708a9d575fa8d4a690b988fd870d65279f98
  OPTIMIZER_DEVICE="${GRPO_OPTIMIZER_DEVICE:-cpu}"
  ACTIVATION_OFFLOAD="${GRPO_ACTIVATION_OFFLOAD:-1}"
  LOGPROB_CHUNK_SIZE="${GRPO_LOGPROB_CHUNK_SIZE:-16}"
else
  DEFAULT_MODEL=/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Rona_Lulu/Lulu_outputs/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
  OPTIMIZER_DEVICE="${GRPO_OPTIMIZER_DEVICE:-cuda}"
  ACTIVATION_OFFLOAD="${GRPO_ACTIVATION_OFFLOAD:-0}"
  LOGPROB_CHUNK_SIZE="${GRPO_LOGPROB_CHUNK_SIZE:-0}"
fi

args=(
  --model "${GRPO_MODEL:-$DEFAULT_MODEL}"
  --train-data "${GRPO_TRAIN_DATA:-${DAPO_TRAIN:-/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/data/dapo_pool2048_s42/train.jsonl}}"
  --expected-train-data-sha256 "${GRPO_TRAIN_DATA_SHA256:-7d8e0ef07b341b90a71958e4513bacdc3e086e152f55d47245c7dd9871bc5cc4}"
  --output-dir "${GRPO_OUTPUT_DIR:-${LULU_OUTPUT_ROOT:-$ROOT_DIR/outputs}/grpo}"
  --gpus "${GRPO_GPUS:-auto}"
  --tuning-mode "${GRPO_TUNING_MODE:-full}"
  --model-family "$MODEL_FAMILY"
  --prompt-mode "$PROMPT_MODE"
  --global-batch-prompts "${GRPO_GLOBAL_BATCH_PROMPTS:-32}"
  --group-size "${GRPO_GROUP_SIZE:-8}"
  --global-epochs "${GRPO_GLOBAL_EPOCHS:-0.125}"
  --max-response-tokens "${GRPO_MAX_RESPONSE_TOKENS:-8192}"
  --max-prompt-tokens "${GRPO_MAX_PROMPT_TOKENS:-4096}"
  --lr "${GRPO_LR:-1e-6}"
  --ppo-epochs "${GRPO_PPO_EPOCHS:-1}"
  --optimizer-device "$OPTIMIZER_DEVICE"
  --logprob-chunk-size "$LOGPROB_CHUNK_SIZE"
  --dtype "${GRPO_DTYPE:-bfloat16}"
  --seed "${GRPO_SEED:-42}"
)
[[ "$ACTIVATION_OFFLOAD" != 1 ]] || args+=(--activation-offload)
[[ -z "${GRPO_MAX_STEPS:-}" ]] || args+=(--max-steps "$GRPO_MAX_STEPS")
[[ "${GRPO_PLAN_ONLY:-0}" != 1 ]] || args+=(--plan-only)

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
exec "$PYTHON_BIN" -u "$ROOT_DIR/scripts/run_grpo.py" "${args[@]}" "$@"
