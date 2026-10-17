# ArcQ new-cluster workspace plan

## 1. One movable workspace

All new work is rooted at `ARCQ_ROOT`. On the new cluster, enter the personal
directory first and create `ArcQ_Supplementary_Experiments`; its absolute parent
may differ between clusters, but everything below it is stable:

```text
ArcQ_Supplementary_Experiments/             # ARCQ_ROOT
├── code/
│   └── Lulu_SUPPLEMENTARY_EXPERIMENTS/     # Git checkout only
├── env/
│   ├── arcq_env.sh                         # exported paths and cache settings
│   └── model_paths.env                     # pinned snapshot paths
├── cache/
│   ├── huggingface/{hub,datasets,assets}/
│   ├── torch/
│   ├── triton/
│   ├── vllm/
│   └── pip/
├── models/
│   ├── Qwen3-1.7B -> ../cache/.../snapshot
│   └── Qwen3-32B  -> ../cache/.../snapshot
├── data/
│   ├── source/                             # immutable source data
│   ├── prepared/                           # fixed pools and train/eval splits
│   └── diagnostic/                         # Experiment 1 frozen prefixes
├── experiments/
│   ├── exp1_predictiveness/
│   │   ├── train/checkpoints/
│   │   ├── diagnostic/pre/
│   │   ├── diagnostic/post/
│   │   ├── evaluation/
│   │   ├── analysis/
│   │   └── logs/
│   ├── exp2_low_kl_matched/
│   │   ├── train/checkpoints/
│   │   ├── evaluation/
│   │   ├── analysis/
│   │   └── logs/
│   ├── exp2_entropy_matched/
│   │   ├── train/checkpoints/
│   │   ├── evaluation/
│   │   ├── analysis/
│   │   └── logs/
│   └── exp3_cost/
│       ├── raw/
│       ├── analysis/
│       └── logs/
├── releases/
│   └── ICLR_rebuttle_data/                 # cloned, read-only results archive
├── manifests/                              # machine-readable provenance
└── tmp/                                    # safe to regenerate
```

The code checkout is deliberately separated from data, caches, checkpoints,
and results. Updating Git cannot overwrite an experiment, and moving the whole
workspace only requires changing `ARCQ_ROOT`.

## 2. Standard pretrained models

Download the two public models into the workspace-local Hugging Face cache and
pin the exact revisions:

| Role | Repository | Revision | Stable workspace path |
|---|---|---|---|
| Student | `Qwen/Qwen3-1.7B` | `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` | `models/Qwen3-1.7B` |
| Teacher | `Qwen/Qwen3-32B` | `9216db5781bf21249d130ec9da846c4624c16137` | `models/Qwen3-32B` |

The stable paths are symlinks to snapshots in `cache/huggingface/hub`. Training
commands consume the stable paths and record the pinned revision in a manifest.
No model weight is stored inside the Git checkout.

## 3. What must be trained again

| Work item | Training required | Code-level arm | Why |
|---|---:|---|---|
| Experiment 1 common update | Yes | `vanilla_opd` | The old Qwen3-1.7B OPD checkpoints were deleted; a pre/post checkpoint pair is required. |
| Experiment 2 Low-KL control | Yes | new `low_kl_matched` allocation | This allocation arm does not yet exist as a completed checkpoint. |
| Experiment 2 entropy control | Yes | new `entropy_matched` allocation | Student causal entropy must be computed and used to reorder the matched HinD weights. |
| Experiment 3 HinD cost | Possibly | submission HinD recipe, expected `ren_balanced` after recipe verification | Rerun only if old logs cannot supply comparable same-hardware telemetry. |
| Experiment 3 OPD cost | Covered by Experiment 1 if fully instrumented | `vanilla_opd` | Do not run a duplicate when Experiment 1 uses the final matched protocol. |
| Experiment 3 OPSD cost | Possibly | `opsd` | Needed for a same-hardware cost table unless a compatible new-cluster run already exists. |

The Base model requires evaluation but no training. Existing evaluation outputs
for Baseline, RQ2.1, and RQ2.3 are already available in the public release
repository and should be cloned into `releases/ICLR_rebuttle_data` for reuse.

## 4. Execution order

1. Create the directory tree and `env/arcq_env.sh` relative to the current
   personal directory.
2. Clone the source repository into `code/` and the published results archive
   into `releases/`.
3. Download and verify the two pinned Qwen snapshots; record their paths and
   hashes in `manifests/`.
4. Locate or recreate the immutable 2048-question training pool and construct a
   disjoint Experiment 1 diagnostic split. Freeze IDs and hashes before GPU use.
5. Add cost telemetry first, because every subsequent production run should
   contribute to Experiment 3.
6. Implement and unit-test the two Experiment 2 allocation rules.
7. Run the Experiment 1 OPD job and retain the required pre/post checkpoints.
8. Run both Experiment 2 arms, then evaluate all new checkpoints with the same
   benchmark protocol.
9. Recompute metrics from raw outputs and publish manifests before release.

## 5. Path migration rule

Existing scripts that contain old absolute defaults are historical code. Before
launching a new ArcQ experiment, replace those defaults with explicit CLI
arguments or `ARCQ_ROOT`-derived paths. This migration is intentionally separate
from the workspace bootstrap: no training command should be launched until its
resolved plan contains only paths under `ARCQ_ROOT` (apart from system binaries
and mounted read-only dependencies).
