#!/usr/bin/env bash
set -euo pipefail
: "${EVALUATION_DIR:?set EVALUATION_DIR to a completed 32k evaluation directory}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BUDGETS="${BUDGETS:-2048,4096,8192,16384,32768}"
exec "$PYTHON_BIN" -u Lulu/scripts/analyze_inference_budget.py --evaluation-dir "$EVALUATION_DIR" --output-dir "$OUTPUT_DIR" --budgets "$BUDGETS" "$@"
