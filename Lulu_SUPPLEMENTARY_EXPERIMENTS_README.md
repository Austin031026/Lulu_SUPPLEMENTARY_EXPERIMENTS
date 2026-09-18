# LuLu / ReN — Paper Supplementary Experiments

> **Purpose.** This README is the execution plan for the *remaining paper experiments* after freezing the current successful `ren_balanced` recipe. It is intentionally paper-facing: every experiment is tied to one RQ, every arm has an explicit fairness criterion, and every saved quantity is listed before the run starts.
>
> **Frozen main method.** Do **not** change the ReN objective while running this suite:
>
> $$
> D_t^C=D_{\mathrm{KL}}(q_T\Vert p_C),\qquad
> D_t^H=D_{\mathrm{KL}}(q_T\Vert p_H),
> $$
> $$
> g_t=[D_t^C-D_t^H]_+,\qquad
> w_t=\frac{g_t}{1+g_t},
> $$
> $$
> L_R=\frac1B\sum_i\frac1{|R_i|}\sum_{t\in R_i}
> w_{it}D_{\mathrm{KL}}(q_{T,it}\Vert p_{\theta,it}),
> $$
> with the current stable recipe `L = L_R + 0.5 L_C + 0.1 L_ref`, 8192-token on-policy rollout, matched causal/live scoring, one optimizer update per round, and Qwen thinking sampling `T=0.6 / top-p=0.95 / top-k=20`.

---

# 0. Fairness rules shared by all supplementary experiments

## 0.1 The source experiment

All Qwen3-1.7B supplementary runs should use the successful frozen experiment as `SOURCE_EXPERIMENT`, e.g.

```bash
export SOURCE_EXPERIMENT=/pfss/mlde/workspaces/mlde_wsp_Eco_Inference/Rona_Soraka/LuLu_outputs/experiments/ren_balanced_matched_b256_c0p5_8k_r4_s42_20260918
```

The paper controllers copy the source train/eval command and only modify the scientific axis under test. This preserves model, data pool, optimizer, learning rate, decoding, context limits, GPU layout, control/reference coefficients, parser, and matched-causal path unless the RQ explicitly changes one of them.

## 0.2 What “same training questions” means

For the main ReN run and all matched RQ1/RQ2 controls:

- `global_batch_prompts = 256`;
- `rounds = 4`;
- `rollouts_per_prompt = 1`;
- therefore each arm receives **1024 prompt exposures**;
- the training pool contains 2048 questions;
- under the frozen deterministic scheduler, the first four 256-question blocks are non-overlapping, so each arm sees **1024 distinct questions**;
- the prompt IDs and per-round prompt schedule are the same across matched arms because train data, seed, scheduler, batch size, and number of rounds are unchanged.

Important distinction:

> The **questions are matched**, but after the first update the **on-policy trajectories are not expected to be identical**, because each method has its own updated Student and must roll out from its own current policy. Forcing identical future trajectories would violate the iterative on-policy comparison.

For every completed matched experiment, the post-run audit should verify:

1. 256 source IDs per round;
2. 1024 unique source IDs over four rounds;
3. identical source-ID sets and round assignment across matched arms;
4. one rollout per source ID;
5. four committed optimizer updates and no skipped update.

## 0.3 Evaluation fairness

Within one RQ experiment:

- use the same fresh Base generation whenever practical;
- use the same benchmark rows, prompt hashes, gold answers, decoding settings, and per-question seeds;
- score with the same final-only parser;
- save both text and token IDs;
- use 32768-token sampled evaluation for the primary long-horizon score;
- derive 2k/4k/8k/16k prefix audits from the saved 32k trajectories instead of regenerating them.

Do not mix a Base from an older execution path with a new method unless it is explicitly labeled as historical/non-matched.

---

# RQ1 — Overall Effectiveness

## Research question

> **Does ReN improve on-policy reasoning distillation across Student scales, model families, and reasoning domains?**

RQ1 is a *performance section*. It should not carry the mechanism story.

## RQ1.1 Main Qwen comparisons

### Methods

- **Base**: pretrained Student, no post-training.
- **No-Reasoning (Control + Ref.)**: diagnostic ablation described below.
- **Vanilla OPD**: full Teacher KL on every reasoning state (`w=1`).
- **OPSD**: privileged Student `p_H` is used directly as the distillation target.
- **ReN**: current bounded absolute ReN.

### What is “Control/Reference Only”?

The code arm is `reasoning_ablation=none`. A clearer paper label is:

> **No-Reasoning (Control + Ref.)**

It is **not** a literature baseline and it is **not** a frozen model. It uses the same on-policy rollouts and still performs the common stabilization updates, but sets the reasoning-distillation component to zero:

$$
L_R=0,\qquad L=0.5L_C+0.1L_{ref}.
$$

Therefore it answers:

> “How much of the final change comes from the shared control/reference training backbone when no reasoning Teacher supervision is used?”

It should be treated as an **ablation / backbone control**, not as the primary baseline against which novelty is claimed.

### Training question count

All three new trained arms (`control_ref`, `vanilla_opd`, `opsd`) use exactly the same training-budget design as ReN:

| Arm | Prompts / round | Rounds | Distinct training questions | Optimizer updates |
|---|---:|---:|---:|---:|
| No-Reasoning (Control + Ref.) | 256 | 4 | 1024 | 4 |
| Vanilla OPD | 256 | 4 | 1024 | 4 |
| OPSD | 256 | 4 | 1024 | 4 |
| ReN | 256 | 4 | 1024 | 4 |

The prompt IDs are matched per round; only the learned policy and therefore later on-policy trajectories can differ.

### Information to analyze

For every Student/Teacher pair save/report:

- MATH-500 accuracy;
- OlympiadBench accuracy;
- AIME25 accuracy;
- Math Avg.;
- MMLU-Pro accuracy;
- GPQA-Diamond accuracy;
- OOD Avg.;
- external micro accuracy;
- response length and hit-cap as secondary diagnostics;
- paired rescue/degradation counts vs Base;
- paired confidence intervals.

### Presentation

**Main table**. Group rows by Student/Teacher pair. Primary columns: MATH / Olympiad / AIME / Math Avg. / MMLU-Pro / GPQA-D / OOD Avg.

Do not put training-engineering diagnostics in the main table.

## RQ1.2 Student scale / model family

Planned blocks:

1. Qwen3-1.7B Student / Qwen3-32B Teacher;
2. Qwen3-4B Student / Qwen3-32B Teacher;
3. non-Qwen family once its source ReN recipe is frozen (candidate: Gemma-3-4B / Gemma-3-27B if tokenizer/reasoning-region requirements pass).

The existing RQ controllers are source-experiment driven. For another Student/family, first establish one frozen ReN source experiment, then reuse the same baseline controllers on that source.

---

# RQ2 — Privileged Supervision Allocation

## Research question

> **Does privileged self-distillation provide useful information for allocating external-Teacher supervision?**

This is the mechanism section. It should establish three progressively stronger claims:

1. **placement matters**;
2. **hindsight adds information beyond causal mismatch**;
3. **selective allocation becomes especially useful as Teacher–Student gap increases**.

---

## RQ2.1 — Does state-specific allocation matter beyond the amount of supervision?

### Methods

- No-Reasoning (Control + Ref.)
- Vanilla OPD
- **Uniform-Matched**
- **Shuffled-ReN**
- **ReN**

Optional in the same table: Causal-Matched (its conceptual role belongs mainly to RQ2.2).

### Exact controls

**Uniform-Matched**

For each trajectory, replace all reasoning weights with that trajectory's mean ReN weight. This preserves the trajectory's total reasoning-weight budget but removes state localization.

**Shuffled-ReN**

Deterministically permute the exact ReN weight multiset within each trajectory. This preserves mean, variance, sparsity, top-k mass, and total weight budget, while breaking the association between a weight and the state that received it.

### Training question count

Every RQ2.1 arm uses:

- 256 prompts / round;
- 4 rounds;
- 1024 distinct matched questions;
- one on-policy rollout per question;
- four optimizer updates.

Thus the RQ2.1 comparison is not confounded by seeing more questions.

### Main information to analyze

**Performance**

- Math Avg.;
- each math benchmark;
- OOD Avg.;
- external micro;
- paired deltas vs Base and vs ReN.

**Allocation invariants**

For ReN / Uniform / Shuffled, verify and save:

- per-trajectory sum of applied weights;
- mean applied weight;
- weight variance;
- top-1% / top-10% weight mass;
- token weight ESS;
- prompt reasoning-loss ESS;
- prompt-balanced retained Teacher-KL fraction.

The point is to demonstrate that any performance difference is caused by *where* supervision is allocated, not by one arm simply receiving a larger reasoning budget.

### Presentation

Use a **controlled ablation table**, not a bar chart.

Recommended columns:

| Method | Weight budget matched? | Weight shape matched? | State placement | Math Avg. | OOD Avg. | Δ Math vs Base |
|---|---|---|---|---:|---:|---:|

A short line below the table can report the invariant checks (e.g. Uniform and Shuffled preserve the intended total weight budget to numerical tolerance).

---

## RQ2.2 — Does hindsight add information beyond causal Teacher–Student mismatch?

### Metrics

Causal Teacher–Student mismatch:

$$
D_t^C=D_{KL}(q_T\Vert p_C).
$$

Hindsight response:

$$
\Delta_t=D_t^C-D_t^H.
$$

ReN weight is a monotone transform of the positive part of `Delta`, so plot signed `Delta`, not only `w`.

### Existing data

No new model run is required for the geometry plot. The source ReN run already stores per-position:

- `causal_kl = D_C`;
- `hindsight_kl = D_H`;
- `resolved_mismatch = D_C-D_H`;
- `bounded_weight`;
- response position;
- trajectory index and round.

### Analysis

**Panel A — continuous geometry**

- x-axis: `D_C`;
- y-axis: `D_C-D_H`;
- render as hexbin/density because there are many positions;
- overlay conditional median and IQR of `Delta` in causal-KL bins;
- horizontal `Delta=0` line.

Scientific question: at matched causal mismatch, does hindsight still meaningfully change transfer priority?

**Panel B — ranking consequence**

For top fractions `k ∈ {1,5,10,20,50}%`, compute overlap between:

- top-k ReN states ranked by `w`;
- top-k causal states ranked by `D_C`.

Plot overlap fraction vs `k`, with random overlap `y=k` as reference.

### Performance control: Causal-Matched

Causal-Matched preserves the exact ReN weight multiset but assigns the weights only according to causal `D_C` rank. It uses the same 1024 matched questions and four updates.

If ReN > Causal-Matched, the performance result complements the geometry plot: hindsight is not merely a reparameterization of causal Teacher difficulty.

### Presentation

A **2-panel figure**:

- (a) `D_C` vs signed hindsight response density + conditional statistics;
- (b) top-k ReN-vs-causal ranking overlap.

Report Causal-Matched performance in the RQ2.1 ablation table or as a small callout adjacent to the figure; do not make another redundant benchmark table.

---

## RQ2.3 — Does selective allocation become more valuable as the Teacher–Student gap grows?

### Teacher sweep

For fixed Qwen3-1.7B Student, planned Teachers:

- Qwen3-4B;
- Qwen3-8B;
- Qwen3-14B;
- Qwen3-32B.

For **each Teacher**, train both:

- Vanilla OPD;
- ReN.

### Training question count

Every Teacher × method arm uses the same:

- 256 prompts / round;
- 4 rounds;
- **the same 1024 distinct prompt IDs**;
- one rollout per prompt;
- four optimizer updates.

Thus the sweep changes Teacher identity and reasoning allocation, not data exposure.

Teacher standalone evaluation has no training and is used only to measure actual capability gap.

### Required saved information

For each Teacher × method × round, preserve:

- Student checkpoint for rounds 0,1,2,3,4;
- benchmark performance;
- measured Teacher standalone Math Avg.;
- `D_C`, `D_H`, signed `Delta` quantiles;
- positive fraction `P(Delta>0)`;
- mean applied ReN weight;
- top-1% / top-10% weight concentration;
- token weight ESS;
- prompt loss ESS;
- prompt-balanced fraction of Teacher KL retained by ReN;
- Teacher-scored sequence tokens / positions;
- Hindsight-scored tokens / positions;
- Teacher/Hindsight service time;
- round wall-clock;
- fixed-prefix policy drift for every retained Student checkpoint.

### Presentation: one 2×2 full-width figure

**Panel A — performance vs measured Teacher gap**

- x-axis: Teacher Math Avg. − Base Student Math Avg.;
- y-axis: post-training Student Math Avg.;
- two curves: Vanilla and ReN;
- Base Student horizontal reference.

This panel already contains Teacher capability and transfer outcome; do not waste a separate panel on Teacher capability alone.

**Panel B — compatibility distribution**

- ridgeline / violin distributions of signed `Delta=D_C-D_H` for 4B/8B/14B/32B;
- annotate `P(Delta>0)` and mean `w`.

Question: how does the structure of ReN-compatible supervision change as Teacher gap grows?

**Panel C — supervision filtering dynamics**

- rows: 4B/8B/14B/32B Teachers;
- columns: rounds 1–4;
- heatmap value: *prompt-balanced fraction of raw Teacher KL retained by ReN*;
- optional cell annotation: mean `w` or top-10% mass.

Question: how aggressively does ReN filter stronger Teachers, and how does this evolve over training?

**Panel D — actual Student policy drift**

On the same fixed Base prefixes, plot round 0–4 policy divergence from Base for Vanilla vs ReN for every Teacher:

$$
D_{KL}(p_{\theta_r}\Vert p_{\theta_0})
$$

(or TV as a secondary line/table statistic).

Question: does stronger Teacher supervision lead to larger policy movement, and does ReN alter that movement while preserving performance?

---

# RQ3 — Scaling and Efficiency

## Research question

> **How does ReN scale with post-training supervision budget, and does it improve reasoning efficiency?**

---

## RQ3.1 — How does ReN scale with on-policy supervision budget?

### Design

Keep the number of optimizer updates fixed at four. Vary only how many fresh questions contribute to each update:

| Prompts / round | Rounds | Unique questions | Optimizer updates |
|---:|---:|---:|---:|
| 64 | 4 | 256 | 4 |
| 128 | 4 | 512 | 4 |
| 256 | 4 | 1024 | 4 |
| 512 | 4 | 2048 | 4 |

Under the frozen scheduler, these prompt sets are nested prefixes of the same shuffled 2048-question pool. This is deliberate: the experiment studies *more fresh on-policy supervision per update*, not more optimizer steps.

Default run: ReN only. Optional `INCLUDE_VANILLA=1` runs a matched Vanilla curve as well.

### Required saved information

For every budget × round:

**Performance**

- held-out DAPO dev accuracy;
- final benchmark accuracies;
- Math Avg.;
- OOD Avg.;
- external micro;
- delta vs fresh Base.

**Coverage / allocation**

- number of unique prompts seen cumulatively;
- reasoning positions;
- mean applied weight;
- top-1% / top-10% mass;
- token-weight ESS and normalized ESS;
- prompt-loss ESS and normalized ESS;
- prompt-balanced Teacher-KL retention.

**Compute**

- rollout response tokens;
- Teacher/Hindsight scored sequence tokens and positions;
- Teacher/Hindsight service seconds;
- round wall-clock;
- total training wall-clock.

### Presentation: one 2×2 figure

**Panel A — learning curves**

- x-axis: cumulative unique prompts;
- y-axis: held-out dev accuracy;
- show every retained checkpoint, not only final points.

**Panel B — benchmark × budget heatmap**

Rows:

- MATH500;
- OlympiadBench;
- AIME25;
- Math Avg.;
- MMLU-Pro;
- GPQA-D;
- OOD Avg.;
- external micro.

Columns: 256 / 512 / 1024 / 2048 unique prompts.

Cells: delta accuracy vs Base.

**Panel C — effective supervision coverage**

Plot normalized token-weight ESS, normalized prompt-loss ESS, and top-10% weight mass vs supervision budget.

Question: does more on-policy data increase the breadth/effective coverage of the sparse ReN signal?

**Panel D — performance/compute frontier**

- x-axis: Teacher-scored sequence tokens (portable primary compute measure);
- y-axis: delta Math Avg. vs Base;
- annotate wall-clock near each point;
- optionally overlay Vanilla if `INCLUDE_VANILLA=1`.

---

## RQ3.2 — Does ReN improve reasoning efficiency under limited inference budgets?

### No new training required

Every paper evaluation stores the full 32k response token IDs. Re-score the **same sampled trajectory** at:

- 2k;
- 4k;
- 8k;
- 16k;
- 32k.

This is a same-trajectory prefix analysis, not five independent generations.

### Required information

Per example and budget:

- prediction / correctness;
- closed-thinking flag;
- valid-final-answer flag;
- prefix length;
- first tested budget at which thinking is closed;
- first tested budget at which a valid final answer exists;
- first tested budget at which the final answer is correct.

### Presentation: 3-panel figure

**Panel A — accuracy vs inference budget**

Base vs ReN Math Avg. at 2k/4k/8k/16k/32k.

**Panel B — benchmark × token-budget gain heatmap**

Rows: five benchmarks. Columns: 2k/4k/8k/16k/32k. Cell value: ReN − Base accuracy.

**Panel C — cumulative solve-by-budget curve**

Fraction of problems with a correct final answer by each tested token budget, Base vs ReN.

If ReN reaches the same accuracy at a smaller budget, this is evidence of inference-time reasoning efficiency. It does **not** by itself prove that the training rollout horizon can be shortened; a 4k-vs-8k training ablation would be a separate appendix experiment if needed.

---

# 1. What experiments are still required?

## Mandatory paper runs

### A. RQ1 matched baselines

New training jobs:

1. No-Reasoning (Control + Ref.)
2. Vanilla OPD
3. OPSD

ReN already exists and is evaluated in the same job.

### B. RQ2.1 allocation controls

New training jobs:

1. Uniform-Matched
2. Shuffled-ReN
3. Causal-Matched

### C. RQ2.3 Teacher sweep

For 4B / 8B / 14B / 32B Teacher:

- Vanilla;
- ReN.

Total: 8 matched training jobs, plus standalone Teacher evaluation.

### D. RQ3.1 scaling

ReN budgets:

- 256 unique questions;
- 512;
- 1024 (already represented by the successful source recipe, but the scaling controller may rerun it for a fully homogeneous matrix unless intentionally configured to reuse it);
- 2048.

Optional: matched Vanilla scaling via `INCLUDE_VANILLA=1`.

## No-training analyses

- RQ2.2 causal-vs-hindsight geometry;
- RQ2.3 fixed-prefix policy drift (Student forward only, no generation);
- RQ3.2 inference-budget rescoring (CPU only once 32k token IDs exist).

---

# 2. How to run

## Step 0 — enter the correct environment and run cheap smoke tests

```bash
cd /pfss/mlde/workspaces/mlde_wsp_Eco_Inference/Rona_Soraka

export PYTHON_BIN=/pfss/mlde/workspaces/mlde_wsp_Eco_Inference/envs/trl/bin/python

bash Lulu/runs/smoke_rq_tasks.sh
```

The cheap smoke suite is CPU-only. It does not prove real-model GPU integration; it catches plan/allocation/analysis regressions before expensive jobs.

## Step 1 — define the successful source experiment and paper output root

```bash
export SOURCE_EXPERIMENT=/pfss/mlde/workspaces/mlde_wsp_Eco_Inference/Rona_Soraka/LuLu_outputs/experiments/ren_balanced_matched_b256_c0p5_8k_r4_s42_20260918
export PAPER_EXP_ROOT=/pfss/mlde/workspaces/mlde_wsp_Eco_Inference/Rona_Soraka/LuLu_outputs/paper_supplement_20260918
mkdir -p "$PAPER_EXP_ROOT"
```

## Step 2 — prepare plans before launching GPUs

The shell wrappers immediately add `--run`. For expensive jobs, first create a plan with the Python controller **without** `--run`, inspect `experiment_plan.json`, then launch.

### RQ1 plan

```bash
$PYTHON_BIN Lulu/scripts/run_rq1_baselines.py \
  --source-experiment "$SOURCE_EXPERIMENT" \
  --output-dir "$PAPER_EXP_ROOT/rq1_baselines"

cat "$PAPER_EXP_ROOT/rq1_baselines/experiment_plan.json"
```

### RQ2.1 plan

```bash
$PYTHON_BIN Lulu/scripts/run_rq2_allocation.py \
  --source-experiment "$SOURCE_EXPERIMENT" \
  --output-dir "$PAPER_EXP_ROOT/rq2_allocation"

cat "$PAPER_EXP_ROOT/rq2_allocation/experiment_plan.json"
```

### RQ2.3 plan

```bash
export TEACHER_4B=/path/to/Qwen3-4B
export TEACHER_8B=/path/to/Qwen3-8B
export TEACHER_14B=/path/to/Qwen3-14B
export TEACHER_32B=/path/to/Qwen3-32B

$PYTHON_BIN Lulu/scripts/run_rq2_teacher_sweep.py \
  --source-experiment "$SOURCE_EXPERIMENT" \
  --output-dir "$PAPER_EXP_ROOT/rq2_teacher_sweep" \
  --teacher qwen4b="$TEACHER_4B" \
  --teacher qwen8b="$TEACHER_8B" \
  --teacher qwen14b="$TEACHER_14B" \
  --teacher qwen32b="$TEACHER_32B" \
  --teacher-eval

cat "$PAPER_EXP_ROOT/rq2_teacher_sweep/experiment_plan.json"
```

### RQ3.1 plan

```bash
$PYTHON_BIN Lulu/scripts/run_rq3_scaling.py \
  --source-experiment "$SOURCE_EXPERIMENT" \
  --output-dir "$PAPER_EXP_ROOT/rq3_scaling" \
  --batches 64,128,256,512

cat "$PAPER_EXP_ROOT/rq3_scaling/experiment_plan.json"
```

At this point explicitly verify model paths, GPU lists, train-data path, evaluation manifest, batch/round settings, control/reference coefficients, and source-experiment hashes.

## Step 3 — run RQ1 baselines

```bash
SOURCE_EXPERIMENT="$SOURCE_EXPERIMENT" \
OUTPUT_DIR="$PAPER_EXP_ROOT/rq1_baselines" \
PYTHON_BIN="$PYTHON_BIN" \
bash Lulu/runs/run_rq1_baselines.sh --detach
```

Expected new train arms:

```text
rq1_baselines/arms/
├── control_ref/train/
├── vanilla_opd/train/
└── opsd/train/
```

Expected summary:

```text
rq1_baselines/analysis/main_results.csv
```

## Step 4 — run RQ2.1 matched allocation controls

```bash
SOURCE_EXPERIMENT="$SOURCE_EXPERIMENT" \
OUTPUT_DIR="$PAPER_EXP_ROOT/rq2_allocation" \
PYTHON_BIN="$PYTHON_BIN" \
bash Lulu/runs/run_rq2_allocation.sh --detach
```

Expected arms:

```text
uniform_matched
shuffled_ren
causal_matched
```

Expected analysis:

```text
rq2_allocation/analysis/performance.csv
rq2_allocation/analysis/allocation_dynamics.csv
rq2_allocation/rq2_geometry/dc_delta_quantiles.csv
rq2_allocation/rq2_geometry/topk_overlap.csv
```

## Step 5 — run RQ2.3 Teacher-strength sweep

```bash
SOURCE_EXPERIMENT="$SOURCE_EXPERIMENT" \
OUTPUT_DIR="$PAPER_EXP_ROOT/rq2_teacher_sweep" \
TEACHER_4B="$TEACHER_4B" \
TEACHER_8B="$TEACHER_8B" \
TEACHER_14B="$TEACHER_14B" \
TEACHER_32B="$TEACHER_32B" \
PYTHON_BIN="$PYTHON_BIN" \
bash Lulu/runs/run_rq2_teacher_sweep.sh --detach
```

After the sweep is complete, run the fixed-prefix drift audit:

```bash
EXPERIMENT_DIR="$PAPER_EXP_ROOT/rq2_teacher_sweep" \
GPU=0 \
PYTHON_BIN="$PYTHON_BIN" \
bash Lulu/runs/run_rq2_policy_drift.sh
```

Expected outputs:

```text
rq2_teacher_sweep/analysis/performance.csv
rq2_teacher_sweep/analysis/teacher_capability.csv
rq2_teacher_sweep/analysis/round_dynamics.csv
rq2_teacher_sweep/analysis/resolved_distribution_quantiles.csv
rq2_teacher_sweep/analysis/policy_drift/policy_drift.csv
```

## Step 6 — run RQ3.1 supervision scaling

Recommended first run: ReN only.

```bash
SOURCE_EXPERIMENT="$SOURCE_EXPERIMENT" \
OUTPUT_DIR="$PAPER_EXP_ROOT/rq3_scaling" \
BATCHES=64,128,256,512 \
PYTHON_BIN="$PYTHON_BIN" \
bash Lulu/runs/run_rq3_scaling.sh --detach
```

If the paper needs a matched Vanilla scaling curve:

```bash
SOURCE_EXPERIMENT="$SOURCE_EXPERIMENT" \
OUTPUT_DIR="$PAPER_EXP_ROOT/rq3_scaling_with_vanilla" \
BATCHES=64,128,256,512 \
INCLUDE_VANILLA=1 \
PYTHON_BIN="$PYTHON_BIN" \
bash Lulu/runs/run_rq3_scaling.sh --detach
```

Expected analysis:

```text
analysis/learning_curve.csv
analysis/benchmark_heatmap.csv
analysis/coverage.csv
analysis/compute_frontier.csv
```

## Step 7 — RQ3.2 inference-budget analysis

For any completed 32k evaluation directory containing stored token IDs:

```bash
EVALUATION_DIR=/path/to/evaluation \
OUTPUT_DIR=/path/to/inference_budget \
PYTHON_BIN="$PYTHON_BIN" \
bash Lulu/runs/analyze_rq3_inference_budget.sh
```

This is CPU-only and does not launch generation.

---

# 3. Recommended run order

Because all major training jobs use the full GPU allocation, run them serially unless the cluster provides a second independent 8-GPU allocation.

Recommended priority:

1. `smoke_rq_tasks.sh`;
2. RQ1 matched baselines;
3. RQ2.1 allocation controls;
4. RQ2.3 Teacher sweep;
5. RQ2.3 fixed-prefix drift;
6. RQ3.1 scaling;
7. RQ3.2 CPU inference-budget analysis.

The reason for this order is scientific, not only computational:

- RQ1 first determines whether ReN beats matched standard baselines under the successful recipe;
- RQ2.1 then tests whether **where** the weight is placed matters;
- RQ2.3 is expensive and only worth interpreting after the basic mechanism controls exist;
- RQ3.1 is primarily a scaling/efficiency question and should not be used to rescue an unvalidated mechanism.

---

# 4. Information that must be preserved before deleting checkpoints

Do not prune an experiment until all required paper analyses for that RQ have been generated.

## Training artifacts

For every round preserve at least:

```text
metrics/round_xxxx.json
diagnostics/round_xxxx/position_scores.npz
diagnostics/round_xxxx/trajectory_scores.jsonl
diagnostics/round_xxxx/summary.json
rollouts/round_xxxx/*.jsonl
```

These contain the quantities needed for `D_C`, `D_H`, signed hindsight response, applied weights, concentration, ESS, prompt contribution, lengths/cap, scored tokens, and timing.

## Checkpoints

- RQ1/RQ2.1: Round4 is sufficient after final evaluation and summaries are complete.
- RQ2.3: **keep rounds 0,1,2,3,4** until fixed-prefix policy drift is finished.
- RQ3.1: **keep rounds 1,2,3,4 for every budget** until the 16-checkpoint dev learning curve and all requested post-hoc analyses are finished.

## Evaluation artifacts

Keep:

- per-example text;
- `response_token_ids`;
- prediction/reward;
- prompt hash / source index / seed / gold;
- length / cap metadata;
- evaluation plan and model/checkpoint mapping.

These are required for RQ3.2 and for paired question-level statistics without new generation.

---

# 5. Final paper output map

| Paper location | Experiment | Presentation | Primary output files |
|---|---|---|---|
| RQ1 | Base / Vanilla / OPSD / ReN across scales/families | Main table | `rq1_baselines/analysis/main_results.csv` + corresponding 4B/family runs |
| RQ2.1 | Uniform / Shuffled / Causal / ReN controls | Controlled ablation table | `rq2_allocation/analysis/performance.csv`, `allocation_dynamics.csv` |
| RQ2.2 | causal mismatch vs hindsight response | 2-panel geometry figure | `rq2_geometry/dc_delta_quantiles.csv`, `topk_overlap.csv` |
| RQ2.3 | Teacher gap sweep | 2×2 full-width figure | `performance.csv`, `teacher_capability.csv`, `round_dynamics.csv`, `resolved_distribution_quantiles.csv`, `policy_drift.csv` |
| RQ3.1 | 256/512/1024/2048 supervision budget | 2×2 scaling figure | `learning_curve.csv`, `benchmark_heatmap.csv`, `coverage.csv`, `compute_frontier.csv` |
| RQ3.2 | 2k/4k/8k/16k/32k inference budget | 3-panel efficiency figure | `prefix_scores.csv`, `budget_summary.csv`, `solve_by_budget.csv` |

---

# 6. Go/no-go checks before spending GPU time

Before each expensive run:

1. `bash Lulu/runs/smoke_rq_tasks.sh` passes;
2. generate `experiment_plan.json` without `--run`;
3. inspect model/Teacher paths and GPU allocation;
4. confirm `batch=256, rounds=4, control=0.5, reference=0.1` for RQ1/RQ2 matched runs;
5. confirm source train-data and evaluation manifest are the frozen ones;
6. confirm the output directory is new/empty or contains exactly the prepared frozen plan;
7. for RQ2.3, verify all Teacher tokenizers/output vocabularies pass the existing compatibility checks before the full sweep;
8. after training, verify source-ID schedule equality before interpreting matched-arm performance.

If any of these checks fails, do not treat that arm as a matched paper comparison.
