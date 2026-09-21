#!/usr/bin/env bash
#SBATCH --job-name=lulu-verl-8gpu-setup
#SBATCH --account=AIFAC_S07_051
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --time=02:00:00
#SBATCH --output=/leonardo_work/AIFAC_S07_051/lulu_verl/logs/%x-%j.out
#SBATCH --error=/leonardo_work/AIFAC_S07_051/lulu_verl/logs/%x-%j.err

set -euo pipefail

root="/leonardo_work/AIFAC_S07_051/lulu_verl"
image="$root/images/verl-uv-cu130.sif"
image_source="docker://verlai/verl:uv.cu130"

mkdir -p \
  "$root/images" \
  "$root/cache/apptainer" \
  "$root/tmp/apptainer" \
  "$root/cache/huggingface" \
  "$root/data" \
  "$root/models" \
  "$root/checkpoints" \
  "$root/logs" \
  "$root/src"

export APPTAINER_CACHEDIR="$root/cache/apptainer"
export APPTAINER_TMPDIR="$root/tmp/apptainer"

if command -v apptainer >/dev/null 2>&1; then
  runtime="$(command -v apptainer)"
elif command -v singularity >/dev/null 2>&1; then
  runtime="$(command -v singularity)"
else
  for candidate in apptainer singularity; do
    module load "$candidate" >/dev/null 2>&1 || true
    if command -v "$candidate" >/dev/null 2>&1; then
      runtime="$(command -v "$candidate")"
      break
    fi
  done
fi

if [[ -z "${runtime:-}" ]]; then
  echo "Neither Apptainer nor Singularity is available" >&2
  exit 1
fi

echo "job_id=$SLURM_JOB_ID"
echo "nodes=$SLURM_JOB_NODELIST"
echo "runtime=$($runtime --version)"
echo "image_source=$image_source"
echo "started=$(date -Is)"

# The batch script itself runs once, so only one process pulls/converts the
# shared image.  A temporary name prevents later jobs from seeing a partial SIF.
if [[ ! -s "$image" ]]; then
  partial="$image.partial.$SLURM_JOB_ID"
  rm -f "$partial"
  "$runtime" pull "$partial" "$image_source"
  mv "$partial" "$image"
fi

echo "image=$(ls -lh "$image")"

# Validate all four A100s on each of the two allocated Leonardo nodes, then
# validate the official verl image with the host NVIDIA devices passed through.
srun --nodes=2 --ntasks-per-node=1 bash -lc '
  set -euo pipefail
  root="/leonardo_work/AIFAC_S07_051/lulu_verl"
  image="$root/images/verl-uv-cu130.sif"
  runtime="$(command -v apptainer || command -v singularity)"
  echo "===== host=$(hostname) ====="
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader
  "$runtime" exec --nv \
    --bind "$root:/workspace/lulu_verl" \
    --env "HF_HOME=/workspace/lulu_verl/cache/huggingface" \
    "$image" \
    bash -lc '\''
      set -e
      echo "container_host=$(hostname)"
      python3 - <<"PY"
import sys
import torch
print(f"python={sys.version.split()[0]}")
print(f"torch={torch.__version__}")
print(f"torch_cuda_build={torch.version.cuda}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"gpu_count={torch.cuda.device_count()}")
for index in range(torch.cuda.device_count()):
    print(f"gpu[{index}]={torch.cuda.get_device_name(index)}")
assert torch.cuda.is_available(), "CUDA is not available inside the container"
assert torch.cuda.device_count() == 4, "Expected four GPUs on each node"
PY
      command -v ray
      command -v uv
    '\''
'

echo "completed=$(date -Is)"
