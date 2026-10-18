#!/usr/bin/env bash
# ArcQ Experiment 1. Source $ARCQ_ROOT/env/arcq_env.sh before running.
set -uo pipefail

if [[ -z "${ARCQ_ROOT:-}" ]]; then
  echo "ERROR: source env/arcq_env.sh first" >&2
  return 1 2>/dev/null || exit 1
fi

NUM_QUESTIONS="${NUM_QUESTIONS:-1024}"
POST_MODEL="${ARCQ_EXP1_POST_MODEL:-$ARCQ_ROOT/experiments/exp3_cost/suite_20261017T170851Z/runs/vanilla_opd/train/checkpoints/round_000004}"
PYTHON_BIN="${PYTHON_BIN:-$ARCQ_ROOT/envs/lulu_probe_qwen3/bin/python}"
CODE_ROOT="$ARCQ_ROOT/code/Lulu_SUPPLEMENTARY_EXPERIMENTS"

"$PYTHON_BIN" -u "$CODE_ROOT/scripts/run_arcq_exp1.py" \
  --arcq-root "$ARCQ_ROOT" \
  --post-model "$POST_MODEL" \
  --num-questions "$NUM_QUESTIONS" \
  --gpus "${GPUS:-0,1,2,3,4,5,6,7}" \
  --run

STATUS=$?
echo "ARCQ_EXP1_EXIT_STATUS=$STATUS"
echo "TERMINAL_STILL_OPEN=YES"
