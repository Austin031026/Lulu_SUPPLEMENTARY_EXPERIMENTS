#!/bin/bash

export ROOT=/mnt/vast/home/wy21quko

export HF_HOME=$ROOT/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HUGGINGFACE_HUB_CACHE=$HF_HUB_CACHE
export HF_DATASETS_CACHE=$HF_HOME/datasets

export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

export NCCL_DEBUG=WARN

export PYTHON=$ROOT/envs/lulu/bin/python