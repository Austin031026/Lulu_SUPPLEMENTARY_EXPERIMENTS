#!/usr/bin/env bash
# Evaluate a GRPO adapter with Lulu's existing held-out benchmark pipeline.
set -euo pipefail

: "${GRPO_CHECKPOINT:?Set GRPO_CHECKPOINT to global_step_N/model (full) or global_step_N/adapter (LoRA)}"
RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export CHECKPOINT="$GRPO_CHECKPOINT"
export CHECKPOINT_NAME="${CHECKPOINT_NAME:-grpo}"
export MODEL_FAMILY="${GRPO_MODEL_FAMILY:-qwen}"
export INCLUDE_BASE="${INCLUDE_BASE:-0}"
export THINKING="${THINKING:-1}"
export OUTPUT_DIR="${GRPO_EVAL_OUTPUT_DIR:-${LULU_OUTPUT_ROOT:-$(cd "$RUN_DIR/.." && pwd)/outputs}/grpo_evaluation}"
exec bash "$RUN_DIR/eval_lulu.sh" "$@"
