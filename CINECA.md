# CINECA cluster

For CINECA work, use this file as the short operating record. Keep future CINECA access, paths, allocation details, and verified commands here.

## Access

From a local Mac Terminal (the CINECA SSO browser session may avoid another OTP prompt while it remains valid):

```bash
step ssh login 'boliang.liu@campus.tu-berlin.de' --provisioner cineca-hpc
ssh bliu0001@login.leonardo.cineca.it
```

The email is the certificate identity; `bliu0001` is the Leonardo SSH username. A new SSH certificate is valid for 12 hours. Do not store passwords, OTPs, private keys, or recovery codes here.

## Leonardo workspace and GPUs

- Existing GPU allocation uses Slurm account `AIFAC_S07_051`; GPU partition: `boost_usr_prod`. The newly verified storage project is `AIFAC_S07_076` (see below). Compute account and storage project must be checked separately for each future job.
- Each Booster node has four NVIDIA A100 64 GB GPUs. A two-GPU Slurm allocation uses two GPUs on one node.
- Persistent code checkout: `$ARCQ_ROOT/code/Lulu_SUPPLEMENTARY_EXPERIMENTS`. Set `ARCQ_ROOT` from the actual workspace path in each new shell, or source `$ARCQ_ROOT/env/arcq_env.sh` when present.
- Active workspace: `ARCQ_ROOT=/leonardo_work/AIFAC_S07_076/ArcQ_Supplementary_Experiments`. This dedicated project folder now holds the verified source checkout and Conda environment. The older `/051` copy remains intact. Do not infer the root from `$WORK`, which still points to `/051` in the login shell. Activate the new environment with `source /leonardo_work/AIFAC_S07_076/ArcQ_Supplementary_Experiments/env/arcq_env.sh`.
- Job `63892248` was verified `RUNNING` on `lrdn0665` on 2026-11-08, with `AllocTRES=...gres/gpu=2`. This is historical; check `squeue` for current state.
- Source files and experiment outputs should be in persistent storage, not in a node-local temporary directory.

## Code provenance

Target branch: `codex/nemotron-grpo-4b` in `https://github.com/Austin031026/Lulu_SUPPLEMENTARY_EXPERIMENTS.git`.
Expected commit: `ffbcc2a95fd34b002a821f8764c250af3c5f1818`.

Before updating an existing checkout, inspect `git status --short` and preserve any changes. Verify `git rev-parse HEAD` and `git ls-remote origin refs/heads/codex/nemotron-grpo-4b` both equal the expected commit. The training entrypoints are `scripts/run_two_gpu_opd_opisd.py` and `scripts/run_two_gpu_hind.py`.

On 2026-11-08 the checkout did not previously exist; it was cloned into `/051`. Both Git checks returned the expected commit, and both training files were visible from the active two-GPU node `lrdn0665`. On 2026-11-09 a clean copy at the same commit was placed in `/076`. This verifies code availability only; it does not verify training readiness or model assets.

## Two-GPU training readiness (2026-11-08)

- Job `63892248` on `lrdn0665` exposed two `NVIDIA A100-SXM-64GB` GPUs, each with 65,536 MiB total; driver `535.274.02`.
- `scripts/run_two_gpu_opd_opisd.py` rejects GPUs below 75 GiB and requires at least 73 GiB free per GPU when `--run` is used. Its 32B Teacher single-GPU load is still untested. These checks cannot pass on the current 64GB node.
- At this 2026-11-08 inspection the model links and 2048-question pool had not yet been prepared. Their later status is recorded below.
- The former `/051` environment remains intact. The active `/076` environment is recorded below.
- The existing `/051` project directory had about 978GB free when checked. See the separate `/076` capacity below before downloading large model assets.

## Storage audit (2026-11-09)

- `cinQuota` reports `/leonardo_work/AIFAC_S07_051` at **9.047T used / 10T quota** (11,969,898 files). `df -h "$WORK"` reports the same work filesystem at **10T total, 9.1T used, 977G available, 91% full**. These figures apply to the shared project account, not just this user's new experiment directory. `cindata` reports roughly 90.5% account use; its sample was about 100 minutes old at inspection.
- `du -sh "$ARCQ_ROOT"` reports only **1.1G** for `ArcQ_Supplementary_Experiments`. Thus the work quota pressure comes from other content in the shared project area. We have not attributed that content to a particular person or directory and have not removed any of it. For capacity checks prefer `cinQuota` / `cindata` over recursive `du` on Lustre.
- `$HOME` is a separate **50G per-user** area; `cinQuota` showed 22.46G used.
- Check current values with `cinQuota`, `cindata`, and `df -h /leonardo_work/AIFAC_S07_076`. `df` on a project directory reflects that project's quota view; `df` on `/051` and `/076` produced different used/free values despite the same Lustre mount.

### Dedicated project storage: `/leonardo_work/AIFAC_S07_076`

- Verified on 2026-11-09 as `bliu0001`: the account belongs to group `AIFAC_S07_076`; the project directory is readable and traversable. A temporary file was successfully created in the directory and immediately removed, confirming write access. Existing top-level directories are owned by `wye00000`; none was changed.
- `cinQuota`: **933.2G used / 10T quota**, about 160,200 files. `df -h /leonardo_work/AIFAC_S07_076`: **10T total, 934G used, 9.1T available, 10% used**. Inode use on the mount was 14%. This is the project storage budget, shared among its members; it is not a personal reservation.
- The existing two-GPU allocation's compute node `lrdn0665` also passed read and temporary-file write checks on `/076` via `srun --jobid=63892248 --overlap`; the probe file was removed. The dedicated `ArcQ_Supplementary_Experiments` subfolder now exists. Do not write into or alter existing `Rona_*` directories.

## Conda environment in `/076` (2026-11-09)

- Verified Miniforge3 26.9.4-1 installer SHA256 `571b410272cc434f28bc6c5c49ca8ffd4a4d1bcba9f57f1b6baccdb46446cfcc`; installed in `env/miniforge3`. Python 3.11 environment is `envs/lulu_py311`.
- Installed PyTorch `2.7.0+cu118` (CUDA 11.8), Transformers `4.52.4`, PEFT `0.20.0`, NumPy `1.26.4`, datasets `3.6.0`, PyArrow `18.1.0`, and the repository's `[eval,grpo,test]` extras. The newest datasets/PyArrow combination initially failed imports; these two compatible versions were pinned in the environment. `python -m pip check` reports no broken requirements; the training CLI `--help` exits successfully.
- In the existing allocation `63892248` on `lrdn0665`, this environment detected two A100-SXM-64GB GPUs: `torch.cuda.is_available() == True`, `torch.cuda.device_count() == 2`. This checks CUDA visibility, not model loading or training.
- Activate in Bash: `source /leonardo_work/AIFAC_S07_076/ArcQ_Supplementary_Experiments/env/arcq_env.sh`. The script derives `ARCQ_ROOT` from its location and keeps Hugging Face, Torch, Triton, vLLM, and pip caches inside the project folder.
- The optional `vllm` extra is not installed. Pinned Qwen3-1.7B and Qwen3-32B snapshots were subsequently downloaded into the project cache and linked under `models/`. OPD/OPSD short staged tests are recorded below; HinD remains untested. A new project 2048-question pool was prepared later, as recorded below.
- Rechecked inside allocation `63892248` on 2026-11-09: active Python is `envs/lulu_py311/bin/python`; PyTorch `2.7.0+cu118`, Transformers `4.52.4`, PEFT `0.20.0`, two visible A100-SXM-64GB GPUs, and `pip check` clean. `importlib.util.find_spec('vllm')` is false and `pip show vllm` finds no package. The deployed 64GB launcher chooses `--backend staged --rollout-backend hf`; `lulu.training.validate_args` explicitly disallows `--backend staged --rollout-backend vllm`. The short OPD/OPSD runs prove that vLLM is not needed for this HF path, but no full-length throughput or 200-step ETA has been established.
- Performance gate: `staged` starts a new rollout and Teacher process each round. The 64-token OPD smoke took about 2m16s just to load the 32B Teacher; if that load time repeats 200 times, it alone sums to roughly 7.5 hours, excluding long rollout, scoring, updates, and checkpoint I/O. This is an extrapolation, not a measured full-run ETA. Do not describe the present 200-step path as throughput-validated. A vLLM rollout requires a compatible CUDA build plus a new staged rollout phase that releases the vLLM engine before HF scoring/Student update; merely installing the Python package does not switch this launcher to vLLM.
- A newly submitted future GPU job `63899681` was canceled at the user's request. No replacement timed job was scheduled. Use the existing allocation for subsequent tests while it remains active.

## Two A100 64GB execution plan

- The original 80GB resident launcher could not run on Leonardo's 64GB cards. The design uses alternating phases on both cards: current Student collects trajectories; Student exits; Qwen3-32B Teacher scores with true TP2; Teacher exits; Student updates from cached Teacher hidden states. The existing staged backend supplies phase ordering and hidden-state target reconstruction; its previous Teacher phase used `device_map='auto'` layer dispatch.
- OPSD uses the Student's hindsight view and needs no external 32B Teacher. The staged backend requires HF rollout, so its runtime differs from the resident vLLM recipe.
- Local implementation update (2026-11-09): `scripts/run_two_gpu_opd_opisd.py` selects the staged backend for two 64GB A100s. OPD uses eager-local Qwen3 Teacher TP2 through `lulu/training.py`; OPSD skips the external Teacher asset. The staged path uses HF rollout and four-digit checkpoint names. The earlier one-round GPU checks are recorded below; the checkpoint cadence has passed a focused CPU test in the isolated cluster checkout.
- HinD remains a blocker on 64GB: its `ren_balanced` objective currently requires the persistent backend and a separate single-GPU Teacher. The staged TP change above does not make HinD runnable on two 64GB cards. Do not start its GPU smoke until a valid sequential implementation exists.

### Two-GPU OPD/OPSD smoke results (2026-11-09)

- The local staged changes were copied into an isolated validation checkout at `$ARCQ_ROOT/tmp/opd_opsd_validation`; the verified Git checkout under `$ARCQ_ROOT/code/` was not overwritten. Targeted OPD/OPSD/TP CPU checks: 6 passed, 2 GPU tests skipped on login node. Nine unrelated pre-existing ReN method test cases failed in the broader suite; their failure was not treated as an OPD/OPSD pass.
- Both tests used existing Slurm allocation `63892248` on `lrdn0665`, two A100-SXM-64GB GPUs, pinned Qwen3-1.7B Student, a separate two-question diagnostic JSONL at `data/diagnostic/two_gpu_smoke/train.jsonl`, one round, two prompts, one rollout each, and a 64-token response cap. The smoke data is not the historical 2048-question pool.
- **OPSD passed:** both Student workers collected one trajectory; the update reported `objective_loss=0.2410411536693573`, 126 reasoning tokens, 2 supervised trajectories, and produced `experiments/smoke_opsd_2gpu_20261109_a/checkpoints/round_0001/lulu_state.json`. Log: `logs/smoke-opsd-2gpu.log`.
- **OPD passed:** Qwen3-32B loaded in eager-local TP2 across the same two GPUs and scored both trajectories; the Student update reported `objective_loss=0.8421602249145508`, 126 reasoning tokens, 2 supervised trajectories, and produced `experiments/smoke_opd_2gpu_20261109_a/checkpoints/round_0001/lulu_state.json`. Log: `logs/smoke-opd-2gpu.log`. The 17-shard Teacher load took about 2m16s in this test.
- This demonstrates the short staged execution path and checkpoint creation on 2×64GB. It does not establish that 8192-token responses, 16 prompts/round, 200 rounds, or the production launcher with the frozen pool fit GPU memory or finish within the job time limit.

## OPD / OPSD 200-step plan and checkpoint recovery (2026-11-09)

- `scripts/run_two_gpu_opd_opisd.py` plans **200 rounds × one optimizer update per round** for each arm, with 16 training questions and one sampled rollout per question per round. The fixed 2048-question pool is sampled by the training scheduler; 3200 prompt uses do not mean 3200 distinct questions.
- Both arms use the same Qwen3-1.7B Student. The Student **samples** its own reasoning trajectory with `do_sample=True`, temperature `0.6`, top-p `0.95`, rollout top-k `20`. On 64GB A100s this is HF generation; the historical resident recipe uses vLLM with the same decoding settings, so the generated tokens are not guaranteed bitwise identical.
- OPD (`vanilla_opd`) uses a fixed answer-blind Qwen3-32B external Teacher. On 2×64GB, its eager-local TP2 path computes next-token logits by teacher forcing the sampled Student trajectory; the Teacher generates no separate rollout. The target is the **full-vocabulary** Teacher softmax, with no sampling or top-k truncation during scoring.
- OPSD (`opsd`) uses the **same-round frozen Student** conditioned on the correct final answer as the hindsight target. This view also teacher-forces the sampled Student trajectory and does not generate a separate rollout. Its full-vocabulary softmax is the distillation target; no external 32B Teacher is loaded in this arm.
- If “Simte” means Qwen **thinking mode**, both causal and hindsight prompts are built with `enable_thinking=True`; OPD Teacher scores the same causal prompt and sampled thinking prefix. If it means **sampling mode**, only the causal Student rollout samples. OPD Teacher and OPSD hindsight Student perform deterministic teacher-forced scoring on the sampled prefix. This is the same role pattern described in the previous experiment README; the decoder backend differs on two 64GB GPUs.
- The trainable Student is teacher-forced over that trajectory for exact full-vocabulary forward KL against the respective frozen target. Thus “sampling” describes trajectory collection; “exact” describes conditional target scoring and the KL calculation. The method matches the repository's OPD/OPSD definitions, though the current 200×16 schedule and HF backend differ from the completed 4×256 ReN run described in `README.md`.
- The latest ReN balanced run also uses `structured_response_regions()` for reasoning/control token coverage, whereas the staged OPD/OPSD path uses the older `reasoning_token_mask()`. The roles and thinking/sampling pattern are consistent, but these two loss masks and objectives are not identical. Do not report the new OPD/OPSD runs as an exact rerun of the ReN balanced experiment.
- These two objectives do **not** include the separate fixed Base Reference KL term used in the earlier ReN balanced recipe. If “自动引用” refers to restoring a checkpoint, the latest pointer and `--resume` implement that; if it refers to an extra reference-model loss, that would change the OPD/OPSD objective and is not part of this plan.
- The two-GPU launcher now sets `--save-every 20`. The staged backend writes a recoverable Student+optimizer checkpoint after **every** successful update, atomically advances `checkpoints/latest`, and retains round 0, 20, 40, …, 200 and the current latest. It prunes only its own marked non-milestone checkpoints after a newer checkpoint is committed. `latest.json` is a readable summary; resume uses the authoritative latest symlink/checkpoint metadata. `--run --resume --output-dir <same run directory>` uses the original frozen plan and restores each incomplete arm automatically.
- These implementation changes are in the local working tree and the isolated cluster validation copy. The earlier 1-round GPU smokes verified both objectives and checkpoint creation, **not** the full 200-round production shape. Production training has not started.

### New project training pool and deterministic order (2026-11-09)

- `data/prepared/dapo_pool2048_s42/train.jsonl` was verified on CINECA: SHA256 `d62f4902220b73919392676e0538b6add33ed42f6a7f4f9a4137ef16898190fb`, 2048 rows, 2048 distinct IDs. The adjacent manifest records DAPO-Math-17k, seed 42, 16,920 eligible questions, zero train/dev overlap by ID and normalized content against the 256-question dev set, and thinking-mode prompt lengths under the configured 4096/16384 limits. The two-GPU OPD/OPSD launcher now pins this new SHA and accepts `--train-data` for an explicit path inside `ARCQ_ROOT/data/prepared`. The canonical method name is `opsd`; the earlier `opisd` spelling remains an accepted alias.
- `lulu.training.schedule` already provides the training order: a seed-42 permutation of all 2048 row indices is consumed in 16-question rounds, without repetition for rounds 1–128. Rounds 129–200 consume 1152 questions from a new permutation seeded with `42 + 1000003`. Total is 3200 prompt uses, 2048 distinct questions, with 1152 revisits. This order is deterministic given the exact JSONL bytes and config; Student rollout tokens are sampled and need not reproduce bit-for-bit across runtimes.
- The local launcher, staged training code, checkpoint manager, and focused checkpoint test were copied into the isolated validation checkout at `$ARCQ_ROOT/tmp/opd_opsd_validation`. The previous three runtime files were backed up under `$ARCQ_ROOT/tmp/opd_opsd_validation_backup_20261109_pool_update`; the verified Git checkout under `code/` was left untouched. On existing allocation `63892248` (`lrdn0665`), both-method **read-only preflight passed** with 200 rounds, 16 prompts, and the new pool. Focused remote CPU tests: 17 passed; the added four-digit staged checkpoint retention/recovery test also passed. This did not start OPD/OPSD production training.

## Historical 2048-question pool provenance (different from the new project pool)

- The source dataset is [`BytedTsinghua-SIA/DAPO-Math-17k`](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k), which the publisher says contains about 17k unique prompts repeated about 100 times. Local [`lulu/data.py`](lulu/data.py) normalizes question text, deduplicates by normalized question hash, drops conflicting gold-answer groups, and makes a seeded train/dev split. Local notes report 1,791,700 source rows, 17,188 unique groups, 12 conflicting groups removed, then 16,920 train and 256 dev.
- [`scripts/prepare_decisive_pool.py`](scripts/prepare_decisive_pool.py) selects 2048 eligible train questions by ascending `SHA256("lulu-decisive-pool-v1:42:" + normalized_question_sha256)` after excluding five exact benchmark overlaps (four MATH500, one OlympiadBench), as recorded in `md/README_before_recipe_20260918.md`. The historical pool `train.jsonl` SHA256 is `7d8e0ef07b341b90a71958e4513bacdc3e086e152f55d47245c7dd9871bc5cc4`.
- Historical documents disagree about whether the pool selection source contained 17,176 or 16,920 prepared rows. The original source manifest, exclusion audit, and frozen `train.jsonl` are not in this checkout or the new workspace. Do not claim a newly sampled 2048 rows match the historical pool until the final SHA256 matches. First locate the original pool and manifest, or reproduce from pinned dataset/source files and compare hashes.
