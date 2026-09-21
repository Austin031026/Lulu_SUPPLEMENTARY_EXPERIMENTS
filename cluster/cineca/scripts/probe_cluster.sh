#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/common.sh"
require_target
load_agent

ssh "${SSH_ARGS[@]}" cineca-gpu 'bash -s' <<'REMOTE'
set -u
echo "== identity =="
hostname
id
echo "== storage =="
for name in HOME WORK FAST SCRATCH; do
  eval "value=\${$name:-}"
  printf '%-8s %s\n' "$name" "$value"
done
echo "== Slurm associations =="
sacctmgr -n -P show assoc where user="$USER" format=Cluster,Account,Partition,QOS 2>/dev/null || true
echo "== GPU partitions =="
sinfo -h -o '%P|%a|%l|%D|%G' 2>/dev/null | grep -E 'gpu|boost|h100|a100' || true
echo "== budget =="
command -v saldo >/dev/null 2>&1 && saldo -b || true
REMOTE
