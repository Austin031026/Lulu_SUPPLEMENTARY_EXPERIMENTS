#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"
require_target
load_agent

if [[ -z "${CINECA_ACCOUNT:-}" ]]; then
  echo "CINECA_ACCOUNT is empty. Run scripts/probe_cluster.sh and set the discovered project account." >&2
  exit 2
fi

MODE="${1:---test-only}"
case "$MODE" in
  --test-only) SBATCH_MODE=(--test-only) ;;
  --submit) SBATCH_MODE=() ;;
  *) echo "Usage: $0 [--test-only|--submit]" >&2; exit 2 ;;
esac

REMOTE_DIR="lulu_cluster_smoke"
ssh "${SSH_ARGS[@]}" cineca-gpu "mkdir -p '$REMOTE_DIR'"
scp "${SSH_ARGS[@]}" "$CINECA_DIR/slurm/gpu_smoke.sbatch" "cineca-gpu:$REMOTE_DIR/gpu_smoke.sbatch"

ssh "${SSH_ARGS[@]}" cineca-gpu \
  sbatch "${SBATCH_MODE[@]}" \
  --account="$CINECA_ACCOUNT" \
  --partition="${CINECA_GPU_PARTITION:-boost_usr_prod}" \
  --qos="${CINECA_GPU_QOS:-boost_qos_dbg}" \
  --gres="gpu:${CINECA_SMOKE_GPUS:-1}" \
  --time="${CINECA_SMOKE_TIME:-00:05:00}" \
  "$REMOTE_DIR/gpu_smoke.sbatch"
