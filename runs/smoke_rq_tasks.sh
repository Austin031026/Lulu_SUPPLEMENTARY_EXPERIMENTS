#!/usr/bin/env bash
set -euo pipefail
PYTHON_BIN="${PYTHON_BIN:-python}"
cd "$(dirname "$0")/.."
"$PYTHON_BIN" -m pytest -q tests/test_allocation_controls.py tests/test_rq_task_plans.py tests/test_rq_analysis_tasks.py tests/test_rq_summaries.py tests/test_rq_cli_smoke.py
