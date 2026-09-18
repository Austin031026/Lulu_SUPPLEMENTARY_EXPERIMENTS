#!/bin/bash
#SBATCH --job-name=lulu_gpu_test
#SBATCH --account=self_distil_llms
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --output=/mnt/vast/home/wy21quko/slurm_logs/%x-%j.out
#SBATCH --error=/mnt/vast/home/wy21quko/slurm_logs/%x-%j.err

set -euo pipefail

ROOT=/mnt/vast/home/wy21quko
PYTHON=$ROOT/envs/lulu/bin/python

mkdir -p "$ROOT/slurm_logs"

source "$ROOT/envs/lulu/bin/activate"

echo "date=$(date)"
echo "host=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

nvidia-smi

"$PYTHON" - <<'PY'
import torch

print("torch:", torch.__version__)
print("CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("visible GPUs:", torch.cuda.device_count())

for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i))

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 8
PY