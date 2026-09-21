#!/usr/bin/env bash
set -euo pipefail

username="bliu0001"
account="AIFAC_S07_051"
host="login.leonardo.cineca.it"
known_hosts="$(mktemp)"
parts_dir="$(mktemp -d)"
trap 'rm -f "$known_hosts"; rm -rf "$parts_dir"' EXIT

addresses=(
  login01-ext.leonardo.cineca.it
  login02-ext.leonardo.cineca.it
  login05-ext.leonardo.cineca.it
  login07-ext.leonardo.cineca.it
)

pids=()
for i in "${!addresses[@]}"; do
  address="${addresses[$i]}"
  ssh-keyscan -T 15 -t rsa,ecdsa,ed25519 "$address" \
    > "$parts_dir/$i" 2>/dev/null &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid" || true
done
for i in "${!addresses[@]}"; do
  address="${addresses[$i]}"
  sed "s/$address/login*.leonardo.cineca.it/g" "$parts_dir/$i" >> "$known_hosts"
done
if [[ ! -s "$known_hosts" ]]; then
  echo "Unable to retrieve Leonardo host keys" >&2
  exit 1
fi

ssh_args=(
  -o BatchMode=yes
  -o ConnectTimeout=20
  -o StrictHostKeyChecking=yes
  -o "UserKnownHostsFile=$known_hosts"
  "$username@$host"
)

job_id="$(ssh "${ssh_args[@]}" bash -s -- "$account" <<'REMOTE'
set -euo pipefail
account="$1"
mkdir -p "$HOME/lulu_cluster_smoke"
sbatch --parsable \
  --job-name=lulu-4gpu-probe \
  --account="$account" \
  --partition=boost_usr_prod \
  --qos=boost_qos_dbg \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --gres=gpu:4 \
  --time=00:05:00 \
  --output="$HOME/lulu_cluster_smoke/%x-%j.out" \
  --error="$HOME/lulu_cluster_smoke/%x-%j.err" \
  --wrap='echo "started=$(date -Is)"; hostname; echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"; nvidia-smi -L; sleep 60'
REMOTE
)"
job_id="${job_id%%;*}"
if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
  echo "Unexpected sbatch response: $job_id" >&2
  exit 1
fi

echo "JOB_ID=$job_id"
submitted_epoch="$(date +%s)"

for attempt in {1..18}; do
  queue="$(ssh "${ssh_args[@]}" "squeue -h -j $job_id -o '%T|%R|%S|%M'" || true)"
  if [[ -n "$queue" ]]; then
    echo "QUEUE=$queue"
    state="${queue%%|*}"
    if [[ "$state" == "RUNNING" || "$state" == "COMPLETING" ]]; then
      now="$(date +%s)"
      echo "OBSERVED_WAIT_SECONDS=$((now - submitted_epoch))"
      break
    fi
    estimate="$(ssh "${ssh_args[@]}" "squeue --start -h -j $job_id -o '%S|%Y|%R'" || true)"
    [[ -n "$estimate" ]] && echo "START_ESTIMATE=$estimate"
  else
    accounting="$(ssh "${ssh_args[@]}" "sacct -n -X -j $job_id -o State,Start,Elapsed,ExitCode -P" || true)"
    echo "ACCOUNTING=$accounting"
    break
  fi
  sleep 10
done

ssh "${ssh_args[@]}" "squeue -h -j $job_id -o 'FINAL_QUEUE=%T|%R|%S|%M'; sacct -n -X -j $job_id -o JobID,State,Submit,Start,Elapsed,AllocTRES,ExitCode -P"
echo "LOG_PATH=\$HOME/lulu_cluster_smoke/lulu-4gpu-probe-$job_id.out"
