#!/usr/bin/env bash
set -euo pipefail
: "${SOURCE_EXPERIMENT:?set SOURCE_EXPERIMENT}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"
exec "$PYTHON_BIN" -u Lulu/scripts/run_rq2_allocation.py --source-experiment "$SOURCE_EXPERIMENT" --output-dir "$OUTPUT_DIR" --run "$@"
