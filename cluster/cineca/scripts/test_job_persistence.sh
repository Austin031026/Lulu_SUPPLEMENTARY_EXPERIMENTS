#!/usr/bin/env bash
set -euo pipefail

username="bliu0001"
account="AIFAC_S07_051"
host="login.leonardo.cineca.it"
old_job="58223404"
experiment_dir="/leonardo_work/AIFAC_S07_051/lulu_persistence_probe_20260920"
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
  ssh-keyscan -T 15 -t rsa,ecdsa,ed25519 "$address" > "$parts_dir/$i" 2>/dev/null &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid" || true; done
for i in "${!addresses[@]}"; do
  address="${addresses[$i]}"
  sed "s/$address/login*.leonardo.cineca.it/g" "$parts_dir/$i" >> "$known_hosts"
done
[[ -s "$known_hosts" ]] || { echo "Unable to retrieve Leonardo host keys" >&2; exit 1; }

ssh_args=(
  -o BatchMode=yes
  -o ConnectTimeout=20
  -o StrictHostKeyChecking=yes
  -o "UserKnownHostsFile=$known_hosts"
  "$username@$host"
)

echo "== previous job =="
ssh "${ssh_args[@]}" "sacct -n -X -j $old_job -o JobID,State,Submit,Start,End,Elapsed,AllocTRES,ExitCode -P"

echo "== prepare persistent directory on login node =="
ssh "${ssh_args[@]}" bash -s -- "$experiment_dir" <<'REMOTE'
set -euo pipefail
experiment_dir="$1"
mkdir -p "$experiment_dir"
printf 'created_on_login=%s\nlogin_host=%s\n' "$(date -Is)" "$(hostname)" > "$experiment_dir/from_login.txt"
if command -v python3 >/dev/null 2>&1; then
  python3 -m venv "$experiment_dir/python_venv"
  "$experiment_dir/python_venv/bin/python" -c 'import sys; print(sys.executable)' > "$experiment_dir/venv_created_on_login.txt"
fi
ls -la "$experiment_dir"
REMOTE

job_id="$(ssh "${ssh_args[@]}" bash -s -- "$account" "$experiment_dir" "$old_job" <<'REMOTE'
set -euo pipefail
account="$1"
experiment_dir="$2"
old_job="$3"
mkdir -p "$HOME/lulu_cluster_smoke"
sbatch --parsable \
  --job-name=lulu-persistence-probe \
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
  --export="ALL,EXPERIMENT_DIR=$experiment_dir,OLD_JOB=$old_job" \
  --wrap='set -euo pipefail; test -f "$HOME/lulu_cluster_smoke/lulu-4gpu-probe-$OLD_JOB.out"; test -f "$EXPERIMENT_DIR/from_login.txt"; cat "$EXPERIMENT_DIR/from_login.txt"; if test -x "$EXPERIMENT_DIR/python_venv/bin/python"; then "$EXPERIMENT_DIR/python_venv/bin/python" -c "import sys; print(\"persistent_python=\" + sys.executable)"; fi; nvidia-smi -L | tee "$EXPERIMENT_DIR/gpus_from_job.txt"; printf "created_by_job=%s\ncompute_host=%s\njob_id=%s\n" "$(date -Is)" "$(hostname)" "$SLURM_JOB_ID" > "$EXPERIMENT_DIR/from_job_$SLURM_JOB_ID.txt"'
REMOTE
)"
job_id="${job_id%%;*}"
[[ "$job_id" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch response: $job_id" >&2; exit 1; }
echo "JOB_ID=$job_id"
submitted_epoch="$(date +%s)"

for attempt in {1..30}; do
  state="$(ssh "${ssh_args[@]}" "sacct -n -X -j $job_id -o State -P | head -n 1" | tr -d '[:space:]' || true)"
  queue="$(ssh "${ssh_args[@]}" "squeue -h -j $job_id -o '%T|%R|%S|%M'" || true)"
  [[ -n "$queue" ]] && echo "QUEUE=$queue"
  if [[ "$state" =~ ^(COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY) ]]; then
    now="$(date +%s)"
    echo "FINISHED_AFTER_SECONDS=$((now - submitted_epoch))"
    break
  fi
  sleep 10
done

echo "== second job accounting =="
ssh "${ssh_args[@]}" "sacct -n -X -j $job_id -o JobID,State,Submit,Start,End,Elapsed,AllocTRES,ExitCode -P"
echo "== second job log =="
ssh "${ssh_args[@]}" "cat \"\$HOME/lulu_cluster_smoke/lulu-persistence-probe-$job_id.out\""
echo "== persistent directory after job =="
ssh "${ssh_args[@]}" "ls -la '$experiment_dir'; cat '$experiment_dir/from_job_$job_id.txt'; cat '$experiment_dir/gpus_from_job.txt'"
