#!/usr/bin/env bash
# Source this file in a Leonardo Bash shell to activate the project environment.
_arcq_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ARCQ_ROOT="$(cd "$_arcq_env_dir/.." && pwd)"
unset _arcq_env_dir

export HF_HOME="$ARCQ_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$ARCQ_ROOT/cache/torch"
export TRITON_CACHE_DIR="$ARCQ_ROOT/cache/triton"
export VLLM_CACHE_ROOT="$ARCQ_ROOT/cache/vllm"
export PIP_CACHE_DIR="$ARCQ_ROOT/cache/pip"
export PYTHONNOUSERSITE=1

source "$ARCQ_ROOT/env/miniforge3/etc/profile.d/conda.sh"
conda activate "$ARCQ_ROOT/envs/lulu_py311"
