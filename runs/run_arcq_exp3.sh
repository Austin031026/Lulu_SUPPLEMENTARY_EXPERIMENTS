#!/usr/bin/env bash
# Run with bash: failure exits this script, never the caller's terminal.
set -euo pipefail
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
ARCQ_ROOT="${ARCQ_ROOT:-$(cd "$CODE_ROOT/../.." && pwd -P)}"
test -f "$ARCQ_ROOT/env/arcq_env.sh"
source "$ARCQ_ROOT/env/arcq_env.sh"
PYTHON_BIN="$ARCQ_ROOT/envs/lulu_probe_qwen3/bin/python"
test -x "$PYTHON_BIN"
unset PYTHONPATH TRANSFORMERS_CACHE
export PYTHONUNBUFFERED=1
exec "$PYTHON_BIN" -u "$CODE_ROOT/scripts/run_arcq_exp3.py" --arcq-root "$ARCQ_ROOT" --run "$@"
