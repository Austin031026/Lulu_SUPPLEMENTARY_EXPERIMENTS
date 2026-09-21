#!/bin/bash
#SBATCH --job-name=lulu-gpu-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=lulu-gpu-smoke-%j.out
#SBATCH --error=lulu-gpu-smoke-%j.err

set -euo pipefail

echo "date=$(date -Is)"
echo "host=$(hostname)"
echo "job_id=${SLURM_JOB_ID:-unset}"
echo "visible_gpus=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

if command -v python3 >/dev/null 2>&1; then
  python3 - <<'PY'
try:
    import torch
except ImportError:
    print("torch is not installed in the default environment; nvidia-smi succeeded")
else:
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    print("visible_gpu_count", torch.cuda.device_count())
PY
fi

