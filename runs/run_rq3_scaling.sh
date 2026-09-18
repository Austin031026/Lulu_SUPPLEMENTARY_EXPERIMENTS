#!/usr/bin/env bash
set -euo pipefail
: "${SOURCE_EXPERIMENT:?set SOURCE_EXPERIMENT}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
args=(--source-experiment "$SOURCE_EXPERIMENT" --output-dir "$OUTPUT_DIR" --batches "${BATCHES:-64,128,256,512}" --run)
if [[ "${INCLUDE_VANILLA:-0}" == "1" ]]; then args+=(--include-vanilla); fi
exec "$PYTHON_BIN" -u Lulu/scripts/run_rq3_scaling.py "${args[@]}" "$@"
