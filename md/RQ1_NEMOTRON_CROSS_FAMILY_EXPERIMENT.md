# RQ1.2 Nemotron cross-family experiment

This is a non-Qwen family baseline comparison motivated by
[`Lulu_SUPPLEMENTARY_EXPERIMENTS_README.md`](../Lulu_SUPPLEMENTARY_EXPERIMENTS_README.md#rq12-student-scale--model-family).
It replaces the *candidate* Gemma pair. Per the experiment owner's scope,
**do not train a Nemotron ReN arm**: run only No-Reasoning (Control + Ref.),
Vanilla OPD and OPSD, plus a single Base evaluation. Therefore these results
cannot establish that ReN itself transfers to the Nemotron family. Both models
must run with reasoning **on** using the Nemotron system message
`detailed thinking on`.

## Cluster model inputs

The snapshot paths below were reported from the cluster cache. They are runtime
inputs, **not constants in Python code**. Pass them through `--model` and
`--teacher-model`, as in the existing baseline commands.

```bash
STUDENT_MODEL=/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/.cache/huggingface/hub/models--nvidia--Llama-3.1-Nemotron-Nano-4B-v1.1/snapshots/d552708a9d575fa8d4a690b988fd870d65279f98
TEACHER_MODEL=/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/.cache/huggingface/hub/models--nvidia--Llama-3_3-Nemotron-Super-49B-v1/snapshots/387156d8d6868c19f3472fa607aa9bfc4f662333
TRAIN_DATA=/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/data/dapo_pool2048_s42/train.jsonl
DATA_MANIFEST=/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/data/rq1_eval_v1/manifest.json
EXP_ROOT=/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/nemotron_rq1_three_baselines_pool2048_b256_r4_s42

test -f "$STUDENT_MODEL/config.json" && test -f "$TEACHER_MODEL/config.json"
test -f "$TRAIN_DATA" && test -f "$DATA_MANIFEST"
```

The reported Student snapshot has two valid safetensors shards; the Teacher
snapshot has 21. The exact Student–Teacher token-ID mapping was separately
reported as verified; the training preflight must continue to enforce that
invariant rather than infer it from equal vocabulary sizes.
The pool and evaluation-manifest paths above are the previously reported
Qwen3 baseline inputs; recheck their availability and hashes on the target
cluster before claiming the matched comparison.

## Intended run

- Model pair: Nemotron Nano 4B Student / Nemotron Super 49B Teacher.
- Eight-GPU layout for Teacher-using arms: Student `0,1,2`; Hindsight `3`;
  Teacher `4,5,6,7` with **true TP=4**. OPSD does not use the external Teacher.
- Three independent arm starts from the same Base Student and 2048-question
  pool; 4 rounds × 256 prompts, one rollout and one global update per round,
  with the common `0.5 L_control + 0.1 L_ref` backbone.
- Evaluate Base and the three Round-4 checkpoints on identical benchmark rows,
  prompts, seeds, sampling and answer parser.
- Round-4 checkpoints: `$EXP_ROOT/arms/{control_ref,vanilla_opd,opsd}/train/checkpoints/round_000004`.
- Joint Base + three-baseline evaluation: `$EXP_ROOT/evaluation/summary.json`,
  with per-example results under `$EXP_ROOT/evaluation/<model>/<benchmark>/`.

## Implementation status

The dedicated entry is `scripts/run_nemotron_baselines.py`. It accepts all
model/data paths at launch, creates a reviewable plan by default, runs CPU-side
interface checks with `--check`, tests one optimizer update with `--smoke`,
and performs the formal three-arm training/evaluation only with `--run`.
The first TP=4 Teacher load, scoring pass, Student backward and evaluation
remain **unverified on the cluster**. Do not treat a generated plan or a
successful CPU-side preflight as a completed experiment.

After assigning a verified 2048-pool `train.jsonl`, a five-benchmark evaluation
`manifest.json`, and a fresh output directory, use the model variables above:

```bash
python scripts/run_nemotron_baselines.py \
  --model "$STUDENT_MODEL" --teacher-model "$TEACHER_MODEL" \
  --train-data "$TRAIN_DATA" --data-manifest "$DATA_MANIFEST" \
  --output-dir "$EXP_ROOT" --gpus 0,1,2,3,4,5,6,7 --check

# Each smoke uses 8 prompts and one complete optimizer update in its own
# output subdirectory; it does not replace the four-round formal run.
for ARM in control_ref vanilla_opd opsd; do
  python scripts/run_nemotron_baselines.py \
    --model "$STUDENT_MODEL" --teacher-model "$TEACHER_MODEL" \
    --train-data "$TRAIN_DATA" --data-manifest "$DATA_MANIFEST" \
    --output-dir "$EXP_ROOT" --gpus 0,1,2,3,4,5,6,7 \
    --smoke --smoke-arm "$ARM" || break
done

# Only after --check and all three smoke runs succeed:
python scripts/run_nemotron_baselines.py \
  --model "$STUDENT_MODEL" --teacher-model "$TEACHER_MODEL" \
  --train-data "$TRAIN_DATA" --data-manifest "$DATA_MANIFEST" \
  --output-dir "$EXP_ROOT" --gpus 0,1,2,3,4,5,6,7 --run
```
