# GRPO baseline

This directory contains the project's custom GRPO baseline ported from
`grpo_standalone_20260922`. It is an independent baseline and does not change
the Lulu/ReN training path.

## Implementation

The default baseline fully fine-tunes Qwen3-1.7B, with eight sampled responses per prompt,
binary parsed-answer rewards, population-standardized within-prompt
advantages, and the clipped PPO surrogate. Homogeneous reward groups receive
zero advantage. It has no reference model and no KL penalty. Rollouts are
collected independently on all visible GPUs; each global batch is then updated
on the first visible GPU. Sampling and log-probabilities use the full vocabulary.
Temperature and top-p are fixed to 1.0 because the
historical rollout and update calculations only agree under that setting.

The production schedule is rollout-compute matched to a 2,048-question,
one-rollout-per-question baseline: GRPO deterministically selects 256 prompt
exposures from the fixed 2,048-question pool (`global_epochs=0.125`), samples
eight responses per prompt, and therefore produces exactly 2,048 rollouts.
With 32 prompt groups per global batch this is eight rollout/update steps, each
containing 256 responses. Both methods retain the same 8,192-token response cap.

The production input is the fixed 2,048-question DAPO pool at
`/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/data/dapo_pool2048_s42/train.jsonl`.
Its expected SHA256 is
`7d8e0ef07b341b90a71958e4513bacdc3e086e152f55d47245c7dd9871bc5cc4`.
The trainer reads its raw messages and applies the Qwen3 chat template with
`enable_thinking=True` at runtime. The old DeepSeek-tokenized Parquet is not a
production input. `GRPO_PROMPT_MODE=pretokenized` remains only as an explicit
legacy/test escape hatch.

The same objective also supports the Nemotron/LLaMA student family without
changing the Qwen path. Set `GRPO_MODEL_FAMILY=nemotron`; raw DAPO messages are
then rendered with NVIDIA's `detailed thinking on` protocol. For a 4B
full-parameter student on 80GB GPUs, use the CPU Adam update, activation
offload, and chunked full-vocabulary log-probability path shown below. Rollout
collection still uses every selected GPU, while the globally normalized update
runs on the first selected GPU.

## Local or interactive cluster run

Install the baseline dependencies once in the cluster environment:

```bash
python -m pip install -e '.[grpo]'
```

```bash
GRPO_MODEL=/path/to/Qwen3-1.7B \
GRPO_TRAIN_DATA=/path/to/dapo_pool2048_s42/train.jsonl \
GRPO_OUTPUT_DIR=/path/to/results/grpo \
GRPO_GPUS=0,1,2,3,4,5,6,7 \
bash runs/train_grpo.sh
```

Nemotron Nano 4B, eight-GPU rollout with the memory-safe full update:

```bash
GRPO_MODEL=/path/to/Llama-3.1-Nemotron-Nano-4B-v1.1 \
GRPO_MODEL_FAMILY=nemotron \
GRPO_TRAIN_DATA=/path/to/dapo_pool2048_s42/train.jsonl \
GRPO_OUTPUT_DIR=/path/to/results/grpo_nemotron_4b \
GRPO_GPUS=0,1,2,3,4,5,6,7 \
GRPO_OPTIMIZER_DEVICE=cpu \
GRPO_ACTIVATION_OFFLOAD=1 \
GRPO_LOGPROB_CHUNK_SIZE=16 \
bash runs/train_grpo.sh
```

Those three memory controls are also the automatic defaults when
`GRPO_MODEL_FAMILY=nemotron`; they are written explicitly above to make the
production configuration auditable. The Qwen family continues to default to
CUDA Adam, no activation offload, and an unchunked log-probability calculation.

Run once with `GRPO_PLAN_ONLY=1`, then with `GRPO_MAX_STEPS=1`, before removing
the step cap for the eight-step production run. Qwen remains the default model
family and retains its original CUDA-Adam/non-offloaded behavior.

Set `GRPO_PLAN_ONLY=1` to validate the data, reward parser, and schedule without
loading a model. Set `GRPO_MAX_STEPS=1` for a one-step GPU smoke test. Reusing
the same output directory resumes complete steps and restores optimizer state;
changing configuration requires a new output directory.

For Slurm, edit the `#SBATCH` account/log directives if necessary, then either
edit the defaults or override all paths at submission time:

```bash
sbatch --export=ALL,\
GRPO_CODE=/cluster/path/Lulu,\
GRPO_MODEL=/cluster/models/Qwen3-1.7B,\
GRPO_TRAIN_DATA=/cluster/data/dapo_pool2048_s42/train.jsonl,\
GRPO_OUTPUT_DIR=/cluster/results/grpo \
runs/grpo_8gpu.slurm
```

`GRPO_GPUS=auto` preserves `CUDA_VISIBLE_DEVICES` assigned by Slurm. All local
and cluster paths are configurable; no macOS download path is embedded in the
runtime files.

## Evaluation

Use the repository's existing evaluator and a prepared evaluation manifest:

```bash
GRPO_CHECKPOINT=/path/to/results/grpo/global_step_8/model \
DATA_MANIFEST=/path/to/evaluation/manifest.json \
BENCHMARKS=math500,aime25,olympiadbench,mmlu_pro,gpqa_diamond \
GPUS=auto \
bash runs/eval_grpo.sh
```

Full checkpoints are self-contained model directories. Set
`GRPO_TUNING_MODE=lora` only for an explicit LoRA ablation; those adapter
checkpoints may need `--adapter-base-model /path/to/base/model` after moving
between clusters. LiveCodeBench additionally requires the official evaluator
checkout through `LCB_REPO`.
