#!/usr/bin/env bash
# Example: DATA_MANIFEST=/path/manifest.json CHECKPOINT=/path/student bash runs/eval_lulu.sh
set -euo pipefail
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$RUN_DIR/.." && pwd)"
export MODEL="${MODEL:-Qwen/Qwen3-1.7B}"

args=(--model "$MODEL" --model-family "${MODEL_FAMILY:-qwen}" --backend "${EVAL_BACKEND:-auto}"
      --decoding "${DECODING:-auto}" --seed "${EVAL_SEED:-42}"
      --output-dir "${OUTPUT_DIR:-${LULU_OUTPUT_ROOT:-$ROOT_DIR/outputs}/evaluation}"
      --gpus "${GPUS:-auto}" --batch-size "${BATCH_SIZE:-8}"
      --max-response-tokens "${MAX_RESPONSE_TOKENS:-8192}"
      --max-prompt-tokens "${MAX_PROMPT_TOKENS:-4096}"
      --max-examples "${MAX_EXAMPLES:-199}" --split "${SPLIT:-full}")
[[ -z "${DATA_MANIFEST:-}" ]] || args+=(--data-manifest "$DATA_MANIFEST")
[[ -z "${EVAL_DATA:-}" ]] || args+=(--benchmark "${BENCHMARK_NAME:-custom}=$EVAL_DATA")
[[ -z "${BENCHMARKS:-}" ]] || args+=(--benchmarks "$BENCHMARKS")
[[ -z "${CHECKPOINT:-}" ]] || args+=(--checkpoint "${CHECKPOINT_NAME:-lulu}=$CHECKPOINT")
[[ "${INCLUDE_BASE:-1}" != 1 ]] || args+=(--include-base)
[[ "${THINKING:-1}" != 0 ]] || args+=(--no-thinking)
[[ "${STORE_TEXT:-0}" != 1 ]] || args+=(--store-text)
[[ -z "${LCB_REPO:-}" ]] || args+=(--lcb-repo "$LCB_REPO")
[[ "${DRY_RUN:-0}" != 1 ]] || args+=(--dry-run)
exec "${PYTHON_BIN:-${PYTHON:-python}}" -u "$ROOT_DIR/scripts/evaluate_lulu.py" "${args[@]}" "$@"
