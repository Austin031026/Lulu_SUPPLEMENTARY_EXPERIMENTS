# ArcQ Experiment 3 cost protocol

Experiment 3 compares `vanilla_opd`, HinD (`ren_balanced`), and `opsd` on the
same Qwen3-1.7B Student, Qwen3-32B Teacher, fixed training schedule, hardware,
and five-benchmark Pass@1 evaluation protocol.

The ordinary OPD run is shared with Experiment 1. It must not be trained a
second time. Experiment 1 consumes the immutable Base model and the final OPD
checkpoint produced here.

The telemetry manifest records the exact command, working directory, Git
commit/dirty state, GPU inventory, and timestamps. The training `run_config.json`
separately records resolved model/data paths and the training-data hash.

## Required measurements

- total wall-clock seconds and GPU-hours;
- number and identity of allocated GPUs;
- generated response tokens;
- rollout plus target-scoring pipeline time;
- Student update time;
- peak sampled device memory, plus Student PyTorch peak allocated and reserved memory;
- final per-benchmark and macro accuracy.

`pipeline_seconds` in the existing training metrics covers rollout collection
and target scoring. It must be reported as **rollout + scoring**, not as pure
generation time. Teacher and hindsight service work seconds are retained as
additional diagnostics. Because vLLM and PyTorch can coexist on a device, peak
memory is sampled at device level through `nvidia-smi`. Student workers also
report PyTorch allocator peaks. These are intentionally separate: allocator
figures cover Student PyTorch processes, while device memory also includes
vLLM, NCCL, Teacher/Hindsight services, and other device allocations.

## Monitoring one training command

Run the existing training command through:

```bash
python scripts/run_with_cost_telemetry.py \
  --output-dir RUN_ROOT/telemetry \
  --label vanilla_opd \
  --gpus 0,1,2,3,4,5,6,7 \
  --poll-seconds 2 \
  -- ORIGINAL_TRAINING_COMMAND
```

Use a fresh telemetry directory for every attempt. A failed attempt remains a
failed cost record and must not be silently combined with a successful retry.

## Final aggregation

Each run root must contain `train/`, `telemetry/`, and `evaluation/`. Aggregate
the three completed runs with:

```bash
python scripts/summarize_arcq_exp3_cost.py \
  --run vanilla_opd=/path/to/opd \
  --run hind=/path/to/hind \
  --run opsd=/path/to/opsd \
  --output-dir /path/to/exp3_cost/analysis
```

The command writes `summary.json`, `summary.tsv`, and `REPORT.md`.
