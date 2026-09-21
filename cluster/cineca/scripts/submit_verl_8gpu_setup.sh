#!/usr/bin/env bash
set -euo pipefail

username="bliu0001"
host="login.leonardo.cineca.it"
remote_root="/leonardo_work/AIFAC_S07_051/lulu_verl"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sbatch_file="$script_dir/../slurm/verl_8gpu_setup.sbatch"
known_hosts="$(mktemp)"
trap 'rm -f "$known_hosts"' EXIT

addresses=(
  login01-ext.leonardo.cineca.it
  login02-ext.leonardo.cineca.it
  login05-ext.leonardo.cineca.it
  login07-ext.leonardo.cineca.it
)

for address in "${addresses[@]}"; do
  ssh-keyscan -T 15 -t rsa,ecdsa,ed25519 "$address" 2>/dev/null |
    sed "s/$address/login*.leonardo.cineca.it/g" >> "$known_hosts" || true
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

ssh "${ssh_args[@]}" "mkdir -p '$remote_root/logs' '$remote_root/slurm'"
scp \
  -o BatchMode=yes \
  -o ConnectTimeout=20 \
  -o StrictHostKeyChecking=yes \
  -o "UserKnownHostsFile=$known_hosts" \
  "$sbatch_file" \
  "$username@$host:$remote_root/slurm/verl_8gpu_setup.sbatch"

response="$(ssh "${ssh_args[@]}" \
  "sbatch --parsable '$remote_root/slurm/verl_8gpu_setup.sbatch'")"
job_id="${response%%;*}"

if [[ ! "$job_id" =~ ^[0-9]+$ ]]; then
  echo "Unexpected sbatch response: $response" >&2
  exit 1
fi

echo "JOB_ID=$job_id"
echo "REMOTE_ROOT=$remote_root"
echo "STDOUT=$remote_root/logs/lulu-verl-8gpu-setup-$job_id.out"
echo "STDERR=$remote_root/logs/lulu-verl-8gpu-setup-$job_id.err"

for _ in {1..12}; do
  state="$(ssh "${ssh_args[@]}" \
    "squeue -h -j '$job_id' -o '%T|%R|%S|%M'" || true)"
  if [[ -n "$state" ]]; then
    echo "QUEUE=$state"
    [[ "${state%%|*}" == "RUNNING" ]] && exit 0
  else
    ssh "${ssh_args[@]}" \
      "sacct -n -X -j '$job_id' --format=JobIDRaw,State,Reason,Submit,Start,Elapsed -P"
    exit 0
  fi
  sleep 5
done
