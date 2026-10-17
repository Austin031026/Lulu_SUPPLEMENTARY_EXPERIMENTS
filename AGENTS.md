# ArcQ new-cluster operating rules

## Scope and default cluster

- Unless the user explicitly says to switch to the old cluster, all setup,
  training, evaluation, analysis, and recovery instructions must target the new
  cluster only.
- The old cluster is read-only historical context. Do not tell the user to run
  new jobs there, and do not copy, delete, or mutate old-cluster artifacts
  without an explicit request.
- Do not embed an old absolute path in new code or commands. In particular,
  paths containing `Rona_Lulu`, the old `Feng_J/Lulu_outputs`, or the old
  `/mnt/vast/...` workspace are not valid new-cluster runtime paths.
- The new-cluster workspace is identified by `ARCQ_ROOT`. Every command must
  derive paths from this variable or from the repository location; never guess
  a cluster-specific absolute prefix.

## Required workspace layout

The repository checkout lives at:

```text
$ARCQ_ROOT/code/Lulu_SUPPLEMENTARY_EXPERIMENTS
```

All generated data stays outside the Git checkout:

```text
$ARCQ_ROOT/
  code/Lulu_SUPPLEMENTARY_EXPERIMENTS/  # version-controlled source
  env/                                  # activation and path configuration
  cache/huggingface/                    # Hugging Face hub/dataset cache
  cache/torch/                          # PyTorch cache
  cache/triton/                         # Triton compilation cache
  cache/vllm/                           # vLLM cache
  models/                               # stable links to immutable base snapshots
  data/source/                          # immutable downloaded/source data
  data/prepared/                        # generated training/evaluation datasets
  data/diagnostic/                      # fixed prefixes for Experiment 1
  experiments/exp1_predictiveness/      # Experiment 1 outputs and checkpoints
  experiments/exp2_low_kl_matched/      # Experiment 2 Low-KL arm
  experiments/exp2_entropy_matched/     # Experiment 2 entropy arm
  experiments/exp3_cost/                # Experiment 3 cost reports
  releases/ICLR_rebuttle_data/          # read-only published evaluation archive
  manifests/                            # hashes, model revisions, run inventory
  tmp/                                  # disposable temporary files
```

- A trained checkpoint belongs under its experiment directory, for example
  `$ARCQ_ROOT/experiments/exp1_predictiveness/train/checkpoints/`. Do not put
  trained checkpoints in `models/` or inside the source checkout.
- `models/` contains stable symlinks or manifests for pretrained Student and
  Teacher snapshots. The actual Hugging Face content-addressed files live in
  `cache/huggingface/hub/`.
- Evaluation output must remain inside the experiment that produced it. The
  GitHub rebuttal archive under `releases/` is an immutable reference, not a
  training-output directory.

## Path handling

- New scripts and launch commands must accept paths through CLI arguments or
  environment variables. Defaults may be relative to `ARCQ_ROOT`; they must not
  contain a personal absolute path.
- Before a cluster run, source `$ARCQ_ROOT/env/arcq_env.sh` and fail clearly if
  `ARCQ_ROOT`, the Student snapshot, the Teacher snapshot, or required data is
  missing.
- Resolve a path once at the boundary, record the resolved value in the run
  manifest, and use it consistently. Published documentation should show the
  path relative to `ARCQ_ROOT` so the workspace can be moved.
- Runtime caches must be isolated by run when concurrent jobs could otherwise
  race. Large caches and model files must never be committed to Git.

## ArcQ supplementary experiments

The source requirements are in `md/ARCQ_SUPPLEMENTARY_EXPERIMENTS.md`; the
new-cluster layout and execution inventory are in
`md/ARCQ_NEW_CLUSTER_LAYOUT.md`.

### Experiment 1: hindsight-score predictiveness

- Start from the immutable Qwen3-1.7B Student and Qwen3-32B Teacher.
- Train one ordinary OPD arm (`vanilla_opd`) on training questions disjoint from
  the diagnostic set. HinD weights must not affect this training.
- Save the pre-training and post-training checkpoints needed to compare the
  same frozen prefixes. Do not regenerate diagnostic trajectories after
  training.
- Store fixed diagnostic inputs and prefix identities under
  `data/diagnostic/`; store distributions, KL values, analysis, and uncertainty
  estimates under `experiments/exp1_predictiveness/`.

### Experiment 2: matched allocation controls

- Add exactly two new allocation arms: `low_kl_matched` and
  `entropy_matched`.
- Preserve each trajectory's HinD weight multiset, total weight, and sparsity;
  only the token-position assignment changes.
- Run Qwen3-1.7B Student with Qwen3-32B Teacher using the same submission
  training and evaluation protocol. Existing RQ2.1 results in the release
  archive are comparison data, not replacement checkpoints.

### Experiment 3: computation cost

- Collect comparable telemetry for HinD, ordinary OPD, and OPSD: total
  GPU-hours, rollout time, update time, generated tokens, peak allocated and
  reserved GPU memory, hardware identity, and final evaluation accuracy.
- Instrument new runs before launching Experiments 1 and 2 so their costs are
  not reconstructed after the fact.
- Historical logs may be reported separately when complete, but must not be
  presented as a same-hardware comparison without verification.

## Models and rerun boundary

- Student: `Qwen/Qwen3-1.7B`, pinned revision
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- Teacher: `Qwen/Qwen3-32B`, pinned revision
  `9216db5781bf21249d130ec9da846c4624c16137`.
- Old Qwen3-1.7B trained checkpoints were not preserved. Experiment 1 ordinary
  OPD and both Experiment 2 arms therefore require fresh training on the new
  cluster.
- Previously published Base/OPD/OPSD/HinD, RQ2.1, and RQ2.3 evaluation outputs
  may be reused from `releases/ICLR_rebuttle_data/` for analysis and consistency
  checks; they are not resumable training states.

## Safety and provenance

- Never delete checkpoints, source data, or release artifacts merely to free
  space. Report the exact target and ask first when deletion is necessary.
- Every production run must save its command, Git commit, model revisions,
  data manifest/hash, seed, GPU inventory, start/end timestamps, and output
  path relative to `ARCQ_ROOT`.
- Prepare and validate a run plan before allocating GPUs. Do not call a new arm
  complete until its checkpoint, evaluation output, and recomputed summary have
  all been verified.
