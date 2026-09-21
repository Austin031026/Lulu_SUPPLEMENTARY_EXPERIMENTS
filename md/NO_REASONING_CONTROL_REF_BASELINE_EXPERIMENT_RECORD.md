# No-Reasoning (Control + Reference) Baseline — Experiment Record

## 1. Current status

Last verified: 2026-09-19, Asia/Shanghai.

| Stage | Status | Evidence |
|---|---|---|
| Repository checkout | Verified | `main` at `ca1d0e50e8449e5d6f5ce124cc3d83ff69b820b2`; cluster worktree matched `origin/main` when checked |
| Six-GPU role parsing | Verified by `--dry-run` | 3 Student + 1 H/reference + TP2 Teacher |
| Student Base files | Verified | Original Hugging Face Qwen3-1.7B snapshot; two weight shards; no `lulu_state.json` and no `optimizer.pt` |
| Teacher files | Verified | Qwen3-32B snapshot; 17 weight shards plus index |
| Raw DAPO interface | Verified, but **not the run input** | Loader read 17,176 source records; this is the source population, not the frozen matched pool |
| 2,048-question training pool | Generated | New-repository script produced `Feng_J/data/dapo_pool2048_s42/train.jsonl`, seed 42, SHA256 `7d8e0ef07b341b90a71958e4513bacdc3e086e152f55d47245c7dd9871bc5cc4` |
| Student/Teacher tokenizer compatibility | Verified | Both tokenizers have 151,669 explicit entries and their complete token-to-ID mappings are equal |
| Formal training | **Not yet confirmed as started** | Only configuration/path checks and a six-GPU dry-run have been reported; no training PID/log/checkpoint evidence has been recorded here |
| Evaluation | **Not run** | Evaluation must wait for the committed Round-4 checkpoint |
| Paper result table | **Not available** | No evaluation summary or `analysis/main_results.csv` exists yet |

The earlier command that pointed directly at the 17,176-row source JSONL is
superseded and its output directory must be removed.  The new launch uses the
generated 2,048-question pool and a distinct experiment directory containing
`pool2048` in its name.

## 2. Scientific identity

This is the RQ1 backbone control named:

```text
No-Reasoning (Control + Reference)
internal arm name: control_ref
reasoning_ablation: none
```

It is not a literature baseline and it is not a no-thinking model.  The Student
still samples ordinary Qwen3 thinking trajectories.  The ablation only sets the
applied Teacher weight on reasoning positions to zero.

The optimized objective is:

```text
L_reasoning = 0
L_total = 0.5 L_control + 0.1 L_ref
```

`L_control` is the answer-blind external-Teacher KL on generated control/final
answer/stop positions.  `L_ref` is the reverse KL from the live Student to the
frozen initial Student and covers all generated response tokens.

## 3. Frozen cluster inputs

### Code

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_SUPPLEMENTARY_EXPERIMENTS
```

### Student Base — original, untrained Qwen3-1.7B

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Rona_Lulu/Lulu_outputs/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
```

Verified configuration:

| Field | Value |
|---|---:|
| `model_type` | `qwen3` |
| architecture | `Qwen3ForCausalLM` |
| hidden size | 2,048 |
| hidden layers | 28 |
| configured vocabulary width | 151,936 |
| explicit tokenizer entries | 151,669 |
| LuLu training state present | No |
| optimizer state present | No |

The Hugging Face snapshot is used instead of any `step_*`, `round_*`, adapter or
previously trained checkpoint.

### Frozen external Teacher — Qwen3-32B

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Rona_Lulu/Lulu_outputs/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
```

The snapshot has 17 safetensor shards and an index.  Its tokenizer mapping was
verified to be exactly equal to the Student tokenizer mapping.

### Training data used by this run

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/data/dapo_pool2048_s42/train.jsonl
```

Generated with the new repository's `scripts/prepare_decisive_pool.py` from the
17,176-row prepared DAPO source using size 2,048 and seed 42.  Recorded output
SHA256: `7d8e0ef07b341b90a71958e4513bacdc3e086e152f55d47245c7dd9871bc5cc4`.

## 4. Hardware and placement

Verified allocation: six idle NVIDIA A100-SXM4-80GB GPUs.

| Physical GPU | Resident role | Rollouts per round |
|---:|---|---:|
| 0 | Student DDP + vLLM rank 0 | 86 |
| 1 | Student DDP + vLLM rank 1 | 85 |
| 2 | Student DDP + vLLM rank 2 | 85 |
| 3 | synchronized Hindsight Student / frozen initial reference service | 0 |
| 4–5 | one Qwen3-32B Teacher in TP2 eager-local mode | 0 |

Total per round: 256 prompts and 256 rollouts.  Total across four rounds: 1,024
prompts and 1,024 rollouts.  Each prompt receives one rollout.

## 5. Frozen training parameters

| Category | Parameter | Value |
|---|---|---:|
| method | method / backend | `ren_balanced` / `persistent` |
| ablation | reasoning ablation | `none` |
| parameters | LoRA rank | 0, full-parameter training |
| precision | master / compute | FP32 master / BF16 compute |
| optimizer | AdamW learning rate | `1e-6` |
| optimizer | weight decay | 0 |
| schedule | rounds / updates per round | 4 / 1 |
| global data | prompts per round | 256 |
| sampling | rollouts per prompt | 1 |
| rollout | backend / batch per Student | vLLM / 16 |
| rollout | vLLM memory fraction | 0.42 |
| rollout | vLLM maximum sequences | 32 |
| rollout | temperature / top-p / top-k | 0.6 / 0.95 / 20 |
| scoring | score batch | 2 |
| update | train microbatch | 1 |
| logits | position chunk | 128 |
| context | prompt / generated / total | 4,096 / 8,192 / 16,384 |
| objective | control coefficient | 0.5 |
| objective | reference coefficient | 0.1 |
| objective | applied reasoning coefficient | 0 |
| gradients | maximum norm | 1 |
| gradients | exact component diagnostics | updates 1 and 4, including cosines |
| checkpoints | explicitly retained | Round 4; latest is committed each round |
| runtime | worker timeout | 7,200 seconds for the reduced Student-worker topology |
| reproducibility | seed | 42 |

Thinking is enabled in the training prompt construction.  The terms
"No-Reasoning" and `reasoning_ablation=none` refer to the removed reasoning
Teacher loss, not to disabling Qwen thinking generation.

## 6. Reserved production output

```text
Experiment root:
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/no_reasoning_control_ref_qwen3_1p7b_pool2048_b256_r4_s42

Training output:
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/no_reasoning_control_ref_qwen3_1p7b_pool2048_b256_r4_s42/training

Training log:
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/no_reasoning_control_ref_qwen3_1p7b_pool2048_b256_r4_s42/logs/train.log

PID file:
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/no_reasoning_control_ref_qwen3_1p7b_pool2048_b256_r4_s42/logs/train.pid

Expected final checkpoint:
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/no_reasoning_control_ref_qwen3_1p7b_pool2048_b256_r4_s42/training/checkpoints/round_000004
```

The output directory must be new for the first launch.  A later resume must use
the identical scientific configuration and the same training-data bytes.

### Checkpoint write and retention behavior

The source Base model is read from the pre-existing cluster cache and is never
modified in place:

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Rona_Lulu/Lulu_outputs/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
```

However, the current implementation does create a second, full managed copy of
those initial Student weights at `checkpoints/round_000000`.  This is not the
input model path being overwritten: it is an experiment-local checkpoint.  The
`ren_balanced` reference service explicitly loads this managed Round-0 copy as
the frozen policy for `L_ref`; there is currently no CLI switch that suppresses
this initial copy.

The persistent backend writes checkpoints under:

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/no_reasoning_control_ref_qwen3_1p7b_pool2048_b256_r4_s42/training/checkpoints
```

The configured `--save-every 20` does **not** mean that this four-round run
waits for Round 20 before saving.  Every completed round first writes a full,
recoverable checkpoint and atomically moves `checkpoints/latest` to it.
`save_every` only decides whether an older checkpoint remains archived after a
newer checkpoint has been committed.

With this command's `--rounds 4 --save-every 20 --retain-checkpoints 4`, the
checkpoint lifecycle is:

| Point in the run | Committed checkpoint | What remains after pruning |
|---|---|---|
| Student initialization | `round_000000` | `round_000000`; also the frozen initial reference |
| Round 1 completed | `round_000001` | `round_000000`, `round_000001`, and `latest -> round_000001` |
| Round 2 completed | `round_000002` | `round_000000`, `round_000002`, and `latest -> round_000002`; Round 1 is pruned |
| Round 3 completed | `round_000003` | `round_000000`, `round_000003`, and `latest -> round_000003`; Round 2 is pruned |
| Round 4 completed | `round_000004` | `round_000000`, `round_000004`, and `latest -> round_000004`; Round 3 is pruned |

Therefore, after a normal four-round completion, the durable checkpoint tree is
expected to be:

```text
training/checkpoints/round_000000/
training/checkpoints/round_000004/
training/checkpoints/latest -> round_000004
training/latest.json
```

`round_000000` is deliberately retained by the automatic checkpoint manager:
it is the original Qwen3-1.7B Student snapshot as loaded for this run, plus the
newly created optimizer state and LuLu metadata.  It is also the frozen
reference policy used by `L_ref`.  `round_000004` is retained explicitly and is
the checkpoint to evaluate.

This run has four strict on-policy rounds.  Each round consumes 256 prompts,
generates one rollout per prompt and performs one global optimizer step after
gradient accumulation (`--update-passes 1`).  Therefore the planned run has
four global optimizer steps in total, and checkpointing is round-based:

```text
Round 1 / optimizer step 1 -> round_000001
Round 2 / optimizer step 2 -> round_000002
Round 3 / optimizer step 3 -> round_000003
Round 4 / optimizer step 4 -> round_000004
```

There is no meaningful "save every 10/20 optimizer steps" interval in this
four-step experiment.  The code writes after every round/global step, while the
retention flags decide which older writes survive:

| Desired durable history | Flags |
|---|---|
| Initial plus final only | `--save-every 20 --retain-checkpoints 4` (current command) |
| Initial, Round 2 and final | `--save-every 2 --retain-checkpoints 4` |
| Initial and every completed round | `--save-every 1 --retain-checkpoints 4` |

The current command is therefore already configured for the smallest supported
managed history: the mandatory initial reference copy plus the final Round-4
checkpoint.  It does not retain Rounds 1–3 after the next successful round is
committed.

### Supplementary README retention requirement for this RQ1 arm

The specified `Lulu_SUPPLEMENTARY_EXPERIMENTS_README.md` states:
**“RQ1/RQ2.1: Round4 is sufficient after final evaluation and summaries are
complete.”**

Therefore this No-Reasoning `control_ref` arm does not need durable trained
weights from Rounds 1, 2 or 3.  During execution the manager still saves the
current round for crash recovery, but the next successful commit prunes the
previous non-retained round.  Before evaluation is complete, keep the manager's
automatic `round_000000` plus `round_000004`.  After final evaluation and the
RQ1 summaries have been verified, this README requires only
`round_000004`; the experiment's Base can be loaded again from the immutable
Hugging Face Base path.  Removing the automatically retained Round-0 directory
would be a separate manual cleanup action and must not be done during training
or while resume may still be needed.

Even when intermediate model weights are pruned, the supplementary README says
to preserve every round's `metrics`, `diagnostics` and `rollouts` artifacts for
paper analysis.  “Only Round 4 is needed” applies to heavyweight model
checkpoint directories, not to those per-round analytical artifacts.

An intermediate Round 1/2/3 checkpoint is still fully recoverable while it is
the current `latest`; it is deleted only after the next round has been saved and
published successfully.  Thus, if the process stops after Round 2, for example,
`latest` remains on `round_000002` and `--resume` continues from Round 2 rather
than starting again.

Each managed checkpoint directory contains:

```text
model*.safetensors*       full Student weights
config.json               model configuration
tokenizer*                tokenizer files
optimizer.pt              optimizer state for exact resume
lulu_state.json           completed_rounds, completed_updates, method and retention metadata
```

Checkpoint publication is crash-safe: files are first written into a temporary
directory, the directory is renamed into place, and the `latest` symlink is
atomically replaced only after the checkpoint is complete.  An interrupted new
write therefore does not invalidate the previously committed `latest`.

The associated per-round artifacts are outside the checkpoint directory:

```text
training/metrics/round_0000.json ... round_0003.json
training/rollouts/round_0000/ ... round_0003/
training/diagnostics/round_0000/ ... round_0003/
training/run_config.json
training/runtime_plan.json
```

The four-digit metrics/rollout indices are zero-based pipeline rounds; the
six-digit checkpoint index is the number of completed rounds.  For example,
`metrics/round_0000.json` corresponds to the update that produces
`checkpoints/round_000001`.

The direct `scripts/train_lulu.py` command stops after training and optional
in-training validation.  Because this launch sets `--validation-every 0`, it
does not run validation, and it does not automatically invoke the separate
RQ1 benchmark evaluation.  The evaluation stage remains a distinct command to
run against `checkpoints/round_000004` after training completes.

## 7. Production training command

Run from the new repository root with the `lulu_probe_qwen3` environment active:

```bash
cd /pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_SUPPLEMENTARY_EXPERIMENTS

export HF_HOME=/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Rona_Lulu/Lulu_outputs/.cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONUNBUFFERED=1
unset TRANSFORMERS_CACHE
unset LULU_BATCH_PROFILE

STUDENT_MODEL=/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Rona_Lulu/Lulu_outputs/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
TEACHER_MODEL=/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Rona_Lulu/Lulu_outputs/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/9216db5781bf21249d130ec9da846c4624c16137
TRAIN_DATA=/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/data/dapo_pool2048_s42/train.jsonl

EXP_ROOT=/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/no_reasoning_control_ref_qwen3_1p7b_pool2048_b256_r4_s42
OUTPUT_DIR="$EXP_ROOT/training"
TRAIN_LOG="$EXP_ROOT/logs/train.log"
PID_FILE="$EXP_ROOT/logs/train.pid"

if [ -e "$OUTPUT_DIR" ]; then
  echo "Refusing to overwrite existing training output: $OUTPUT_DIR"
  exit 1
fi
mkdir -p "$EXP_ROOT/logs"

nohup python -u scripts/train_lulu.py \
  --model "$STUDENT_MODEL" \
  --teacher-model "$TEACHER_MODEL" \
  --train-data "$TRAIN_DATA" \
  --output-dir "$OUTPUT_DIR" \
  --method ren_balanced --backend persistent --reasoning-ablation none \
  --gpus 0,1,2,3,4,5 --student-gpus 0,1,2 --hindsight-gpus 3 --teacher-gpus 4,5 \
  --teacher-gpus-per-worker 2 --teacher-tp-mode eager-local \
  --rounds 4 --global-batch-prompts 256 --rollouts-per-prompt 1 --update-passes 1 \
  --rollout-backend vllm --rollout-batch-size 16 --rollout-top-k 20 \
  --rollout-vllm-memory 0.42 --rollout-vllm-max-seqs 32 \
  --score-batch-size 2 --train-micro-batch-size 1 --logit-chunk-size 128 \
  --max-new-tokens 8192 --max-prompt-tokens 4096 --max-sequence-tokens 16384 \
  --temperature 0.6 --top-p 0.95 --top-k 32 \
  --learning-rate 1e-6 --weight-decay 0 --max-grad-norm 1 \
  --control-loss-coef 0.5 --reference-kl-coef 0.1 \
  --lora-rank 0 --master-weights-fp32 --gradient-checkpointing --match-causal-update \
  --gradient-norm-every 4 --gradient-cosines --reasoning-diagnostic-split 0 \
  --validation-every 0 --save-every 20 --retain-checkpoints 4 \
  --worker-timeout 7200 --seed 42 \
  > "$TRAIN_LOG" 2>&1 < /dev/null &

echo $! > "$PID_FILE"
```

## 8. Evaluation status and required follow-up

This experiment is currently a **training-only pending run**.  No evaluation
has been launched or completed.  Evaluation cannot be considered complete until
the final checkpoint's `lulu_state.json` confirms four completed rounds.

The intended matched RQ1 evaluation is thinking-mode sampled generation with:

```text
benchmarks = Math500, AIME25, OlympiadBench, MMLU-Pro, GPQA Diamond, DAPO dev128
maximum examples per benchmark = 199
decoding = Qwen thinking sampling
temperature / top-p / top-k = 0.6 / 0.95 / 20
maximum response tokens = 32768
maximum model length = 40960
context safety margin = 128
scoring = only the final answer after a closed </think>
stored artifacts = response text and token IDs
```

After training, evaluation must at minimum include the original Base and the
new `control_ref` Round-4 checkpoint under the same questions and seeds.  The
full paper RQ1 table additionally includes Vanilla OPD, matched OPSD and the
completed ReN checkpoint.  Until those evaluation artifacts exist, this record
must not report accuracy, Pass@1, benchmark deltas or a final paper conclusion.

### 8.1 Frozen 512-question OlympiadBench evaluation override

For the 2026-09-19 Base-versus-No-Reasoning evaluation, the locally
materialized RQ1 manifest is frozen at:

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/data/rq1_eval_v1/manifest.json
```

The manifest contains 500 MATH500, 30 AIME25, **512 OlympiadBench**, 1,400
MMLU-Pro and 198 GPQA-Diamond rows. The current Hugging Face
`OE_TO_maths_en_COMP` source yields 580 valid single-answer rows after the
schema filters, so preparation deterministically retains the first 512 rows.
The GPQA source is the public 198-row `bdytx5/gpqa_gpqa_diamond` mirror.

This expanded evaluation uses `--max-examples 512`. Therefore the actual
per-model counts are MATH500 500, AIME25 30, OlympiadBench 512, MMLU-Pro 512
and GPQA-Diamond 198. This is a documented override of the paper README's
199-row preliminary cap and must not be mixed with an older 199-cap evaluation
in a paired table. Fresh Base and `control_ref_round4` generations are produced
together with identical rows, decoding and per-question seeds. Later RQ1 arms
must reuse this exact manifest and 512 cap when compared with this run.

## 9. Completion evidence to append later

When training is actually launched or finishes, append rather than overwrite:

1. launch time, PID and the first successful resident runtime plan;
2. committed checkpoint paths for Rounds 0–4;
3. per-round objective, `control_penalty`, `reference_penalty`, generated-token
   totals, rollout lengths, cap rate and applied reasoning weight sum;
4. confirmation that applied reasoning weight remains zero;
5. final checkpoint state and training exit status;
6. evaluation command, evaluation summary path and benchmark results;
7. grouped RQ1 metrics and deltas from the fresh Base.
