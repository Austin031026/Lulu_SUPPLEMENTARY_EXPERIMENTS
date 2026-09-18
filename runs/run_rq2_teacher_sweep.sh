#!/usr/bin/env bash
set -euo pipefail
: "${SOURCE_EXPERIMENT:?set SOURCE_EXPERIMENT}"
: "${OUTPUT_DIR:?set OUTPUT_DIR}"
: "${TEACHER_4B:?set TEACHER_4B}"; : "${TEACHER_8B:?set TEACHER_8B}"; : "${TEACHER_14B:?set TEACHER_14B}"; : "${TEACHER_32B:?set TEACHER_32B}"
PYTHON_BIN="${PYTHON_BIN:-python}"
exec "$PYTHON_BIN" -u Lulu/scripts/run_rq2_teacher_sweep.py \
  --source-experiment "$SOURCE_EXPERIMENT" --output-dir "$OUTPUT_DIR" \
  --teacher qwen4b="$TEACHER_4B" --teacher qwen8b="$TEACHER_8B" \
  --teacher qwen14b="$TEACHER_14B" --teacher qwen32b="$TEACHER_32B" \
  --teacher-eval --run "$@"
