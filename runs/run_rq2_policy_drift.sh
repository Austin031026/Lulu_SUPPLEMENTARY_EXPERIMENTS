#!/usr/bin/env bash
set -euo pipefail
: "${EXPERIMENT_DIR:?set EXPERIMENT_DIR to a completed RQ2.3 teacher sweep}"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
exec "$PYTHON_BIN" -u Lulu/scripts/run_rq2_policy_drift.py --experiment-dir "$EXPERIMENT_DIR" --gpu "$GPU" --run "$@"
