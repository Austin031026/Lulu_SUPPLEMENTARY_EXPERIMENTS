#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]] || [[ ! "$1" =~ ^[0-9]+$ ]]; then
  echo "Usage: $0 JOB_ID" >&2
  exit 2
fi

job_id="$1"
username="bliu0001"
host="login.leonardo.cineca.it"
remote_root="/leonardo_work/AIFAC_S07_051/lulu_verl"
known_hosts="$(mktemp)"
trap 'rm -f "$known_hosts"' EXIT

for address in \
  login01-ext.leonardo.cineca.it \
  login02-ext.leonardo.cineca.it \
  login05-ext.leonardo.cineca.it \
  login07-ext.leonardo.cineca.it; do
  ssh-keyscan -T 15 -t rsa,ecdsa,ed25519 "$address" 2>/dev/null |
    sed "s/$address/login*.leonardo.cineca.it/g" >> "$known_hosts" || true
done

ssh_args=(
  -o BatchMode=yes
  -o ConnectTimeout=20
  -o StrictHostKeyChecking=yes
  -o "UserKnownHostsFile=$known_hosts"
  "$username@$host"
)

ssh "${ssh_args[@]}" bash -s -- "$job_id" "$remote_root" <<'REMOTE'
set -euo pipefail
job_id="$1"
root="$2"

echo "== squeue =="
squeue -j "$job_id" -o '%.18i %.10T %.24R %.20S %.10M %.10l %.6D %b' || true
echo "== estimated start =="
squeue --start -j "$job_id" -o '%.18i %.10T %.24R %.20S' || true
echo "== accounting =="
sacct -X -j "$job_id" --format=JobIDRaw,State,Reason,Submit,Start,End,Elapsed,AllocTRES -P || true

stdout="$root/logs/lulu-verl-8gpu-setup-$job_id.out"
stderr="$root/logs/lulu-verl-8gpu-setup-$job_id.err"
if [[ -f "$stdout" ]]; then
  echo "== stdout tail =="
  tail -n 80 "$stdout"
fi
if [[ -s "$stderr" ]]; then
  echo "== stderr tail =="
  tail -n 80 "$stderr"
fi
REMOTE
