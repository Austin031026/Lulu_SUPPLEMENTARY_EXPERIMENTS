# Rebato：Qwen3-1.7B Evaluation 原始文件盘点（待确认）

更新时间：2026-10-03

本清单用于确认后续要搬入 `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Rebato` 的原始 evaluation 文件。当前内容来自本地实验登记和评测代码；原 PFSS 集群尚未完成实时只读验收，因此“已登记”不等于“当前文件仍存在”。本清单不执行复制、移动或删除。

## 1. 已确认的 1.7B 算法集合

### RQ1 Pass@16 的四个算法

| 统一名称 | evaluation 内部目录名 | 说明 |
|---|---|---|
| Base | `base` | 原始 Qwen3-1.7B，不训练 |
| No-Reasoning | `control_ref_round4` | Control + Reference，Round 4 |
| Vanilla OPD | `vanilla_opd_round4` | Round 4 |
| OPSD | `opsd_round4` | Round 4 |

### 另外存在，但不属于上述四算法 Pass@16 的 1.7B 结果

| 统一名称 | 内部目录名 | 说明 |
|---|---|---|
| ReN matched | `base`、`round2`、`round4` | 2026-09-18 matched recipe；825 个外部题 + dev128，Pass@1/single generation，并带 8k prefix 重评分 |
| ReN / Qwen3-32B Teacher | `qwen32b_ren` | RQ2.3 Teacher sweep 的 Round-4 Student；约 825 题，配置与 RQ1 1,752 题评测不同 |
| Historical ReN | `base`、`round8`、`final12` | 历史 12-round 8k 与长预算复评；仅作为历史/负结果保存 |

目前没有证据表明 No-Reasoning、Vanilla OPD、OPSD 之外的 RQ2.1 allocation control 已经产生正式 evaluation 数据。

## 2. Non-coding benchmark

正式 non-coding 五集为：

| 目录名 | 展示名 | RQ1 每模型题数 | RQ2.3/旧评测常见题数 |
|---|---|---:|---:|
| `math500` | MATH-500 | 500 | 199 |
| `aime25` | AIME25 | 30 | 30 |
| `olympiadbench` | OlympiadBench | 512 | 199 |
| `mmlu_pro` | MMLU-Pro | 512 | 199 |
| `gpqa_diamond` | GPQA-Diamond | 198 | 198 |

不能把 1,752 题配置和 825 题配置直接合并。搬运时需要同时保留每次运行自己的 `eval_plan.json` 和 `summary.json`。

## 3. Non-coding Pass@16：两路源目录

Pass@16 的目标 seed 是 42–57。四个算法、五个 benchmark、每题 16 次 generation 必须按逐题键合并，不能平均 16 个 `summary.json`。

### 路径 A：原 8-GPU 正序任务

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/rq1_qwen3_1p7b_pass16_32k_five_bench_s42
```

登记采用：

```text
sample_00_seed_42/
sample_01_seed_43/
sample_02_seed_44/
sample_03_seed_45/
sample_04_seed_46/
sample_05_seed_47/
sample_06_seed_48/
sample_07_seed_49/
sample_08_seed_50/
sample_09_seed_51/
sample_10_seed_52/
```

每个 sample 预期包含：

```text
eval_plan.json
summary.json
base/{math500,aime25,olympiadbench,mmlu_pro,gpqa_diamond}/
control_ref_round4/{math500,aime25,olympiadbench,mmlu_pro,gpqa_diamond}/
vanilla_opd_round4/{math500,aime25,olympiadbench,mmlu_pro,gpqa_diamond}/
opsd_round4/{math500,aime25,olympiadbench,mmlu_pro,gpqa_diamond}/
```

每个 `<algorithm>/<benchmark>/` 中预期保存 `shard-*.jsonl` 和该 benchmark 的 `summary.json`；根目录还可能有 `worker-plan-*.json`、`worker*.log`、`dispatch.jsonl` 和 timing/状态文件，这些也属于原始评测证据。

### 路径 B：6-GPU 倒序任务

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/rq1_qwen3_1p7b_pass16_32k_five_bench_s42/reverse_6gpu
```

登记目标：

```text
sample_11_seed_53/
sample_12_seed_54/
sample_13_seed_55/
sample_14_seed_56/
sample_15_seed_57/
```

另需保留：

```text
reverse_status.json
logs/controller_*.log
logs/current_log_path.txt
```

本地最后一次状态记录称这一路当时在 seed 57、0/8 seed 时因缺少 `latex2sympy2_extended` 失败；用户现在说明两路均已运行，因此必须以集群当前目录为准重新验收 53–57，不能沿用旧状态，也不能先假定五个 seed 均完整。

### Pass@16 根级控制文件

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/rq1_qwen3_1p7b_pass16_32k_five_bench_s42/PASS16_SPLIT_MANIFEST.json
```

该文件登记纳入/排除的 seed。正式汇总还需要检查：

- seed 42–57 恰好各出现一次；
- 四个算法均完整；
- 每个算法的五个 benchmark 完整；
- `prompt_index`、`prompt_sha256`、`ground_truth` 在 16 个 seed 间一致；
- `generation_seed` 与 sample 目录匹配；
- 没有把早期 partial/retry 目录混入。

## 4. Non-coding 的单次 evaluation 目录

这些目录应作为 Pass@1、评测配置和补充证据保留，即使论文主结果最终使用 Pass@16。

### Base + No-Reasoning

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/no_reasoning_control_ref_qwen3_1p7b_pool2048_b256_r4_s42/evaluation_base_control_ref_round4_32k_max512
```

已登记子目录：

```text
base/
control_ref_round4/
summary.json
```

### Vanilla OPD

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/vanilla_opd_qwen3_1p7b_pool2048_b256_r4_s42/evaluation_vanilla_opd_round4_32k_max512
```

已登记子目录：

```text
vanilla_opd_round4/
summary.json
```

### OPSD

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/opsd_qwen3_1p7b_pool2048_b256_r4_s42/evaluation_opsd_round4_32k_max512
```

已登记子目录：

```text
opsd_round4/
summary.json
```

明确排除：Vanilla OPD 的 `.failed_20260919_135543` 目录。

### RQ2.3 32B-Teacher ReN Student

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/rq2_teacher_sweep_qwen3_1p7b_pool2048_b256_r4_s42_6gpu/evaluation_rq23_students_round4_32k_five_bench_s42
```

已登记：

```text
qwen32b_ren/
summary.json
```

它是 825 题配置，不属于 RQ1 Pass@16 的 1,752 题集合。

### 2026-09-18 matched ReN

```text
/pfss/mlde/workspaces/mlde_wsp_Eco_Inference/Rona_Soraka/LuLu_outputs/experiments/ren_balanced_matched_b256_c0p5_8k_r4_s42_20260918
```

需要集群核对实际 workspace；本地 README 使用相对路径登记。预期保留：

```text
evaluation/
prefix_8192/
summary.json（位于 evaluation 下）
comparison.json
REPORT.md
```

模型名为 `base`、`round2`、`round4`，其中论文主要候选为 Round 4。

## 5. Coding benchmark：当前只能确认格式，尚未确认实际源路径

仓库评测代码支持的 coding benchmark 是：

```text
livecodebench
```

正式运行必须使用完整 split 和官方 LiveCodeBench scorer。对每个算法，原始目录按代码应包含：

```text
<algorithm>/livecodebench/shard-*.jsonl
<algorithm>/livecodebench/summary.json
<algorithm>/livecodebench/livecodebench_custom.json
<algorithm>/livecodebench/livecodebench_custom_codegeneration_output_eval_all.json
<algorithm>/livecodebench/scored/shard-*.jsonl
```

evaluation 根目录还应有：

```text
eval_plan.json
summary.json
dispatch.jsonl
worker-plan-*.json
worker*.log
```

但是本地 README、状态文档和 git 历史都没有记录正式 1.7B LiveCodeBench run 的绝对输出目录，也没有登记 coding Pass@16 的两路 seed 目录。因此以下信息必须通过 PFSS 集群只读扫描补齐：

1. coding eval 的根目录名；
2. 实际包含哪些算法；
3. 是 Pass@1 还是 Pass@16；
4. 如果是 Pass@16，两个源根目录及 seed 分工；
5. official scorer 产物是否完整；
6. `eval_plan.json` 中的 LiveCodeBench release version、题数和 decoding 设置。

## 6. 建议的 Rebato 目标结构

为避免 benchmark 类型、算法和 seed 混淆，建议采用“模型 → 评测类型 → 算法 → benchmark → sample”的结构：

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Rebato/
└── Qwen3-1.7B/
    ├── non_coding/
    │   ├── base/
    │   │   ├── math500/
    │   │   │   ├── sample_00_seed_42/
    │   │   │   └── ...
    │   │   ├── aime25/
    │   │   ├── olympiadbench/
    │   │   ├── mmlu_pro/
    │   │   └── gpqa_diamond/
    │   ├── no_reasoning/
    │   ├── vanilla_opd/
    │   ├── opsd/
    │   └── ren/
    └── coding/
        ├── base/
        │   └── livecodebench/
        ├── no_reasoning/
        │   └── livecodebench/
        ├── vanilla_opd/
        │   └── livecodebench/
        ├── opsd/
        │   └── livecodebench/
        └── ren/
            └── livecodebench/
```

此外在 `Qwen3-1.7B/` 根部保存：

```text
MANIFEST.json
FILES_SHA256.txt
SOURCE_PATHS.tsv
PASS16_VALIDATION.json
```

建议采用复制而不是移动。先完成覆盖率与 SHA256 验收，再考虑是否清理源目录。

## 7. 确认前仍缺的信息

1. `Rebato` 是否确实位于 `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Rebato`；
2. 目标层级是否接受 `模型/评测类型/算法/benchmark/sample`；
3. ReN 是只收 2026-09-18 matched Round 4，还是把 RQ2.3 32B-ReN 和 historical ReN 分成独立算法变体；
4. coding benchmark 是否只有 LiveCodeBench；
5. 原 PFSS Determined 登录需要重新认证，才能对 coding 输出和 Pass@16 第二路做实时文件验收。

