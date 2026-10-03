# Rebuttal 实验结果总账与抢救清单

更新时间：2026-10-03

本文只登记仓库文档中已经留下明确完成证据或数值的实验。它不把“已实现”“已规划”“训练启动过”自动视为完成结果。集群路径目前仅来自既有记录，尚未在集群文件系统上重新验收。

## 1. 当前结论

| 证据级别 | 实验 | 当前可用性 |
|---|---|---|
| A：可直接整理进 rebuttal | 2026-09-18 Qwen3-1.7B matched ReN | 有完整主表、配对差值、bootstrap CI、rescue/degradation 和训练诊断 |
| A：可直接整理进 rebuttal | Qwen3-1.7B No-Reasoning（Control + Ref.） | 有五集完整评测表和 Base 对照 |
| A：可直接整理进 rebuttal | Qwen3-4B RQ1 Base / No-Reasoning / Vanilla OPD / OPSD | 有统一 1,752 题评测表、配对 CI 和逐题目录登记 |
| B：可作为历史/负结果补充 | 历史 12-round Qwen3-1.7B ReN | 有 8k、final-only、dev256 和长预算复评；不同口径不能与主结果混表 |
| B：已跑但需要从集群取回数值 | Qwen3-1.7B Vanilla OPD、OPSD | 训练和单模型 evaluation 均登记为完成，但本仓库没有复制具体 accuracy 数值 |
| B：训练完成，完整论文分析未确认 | RQ2.3 Teacher-strength sweep | 8/8 arm 训练完成；发现 32B-ReN Student evaluation 路径登记，但全 Student/Teacher 汇总和 policy drift 没有完成证据 |
| C：未完成，不能作为最终结果 | Qwen3-1.7B Pass@16 | seed 42–52 共 11 个 seed 已验收；seed 53–57 因依赖缺失没有有效生成 |
| D：只有实现或计划 | RQ2.1 allocation controls、RQ3.2 prefix analysis、Nemotron、GRPO、gate/correction audits | 本地没有足以证明正式实验完成的结果总表 |

checkpoint 已删除后，A 类实验的既有 benchmark 结论仍可保留，前提是下面第 6 节列出的评测证据仍在。需要加载模型权重的新评测、复现实验和 checkpoint 间 policy drift 则不能靠汇总表恢复。

## 2. Rebuttal 主结果：Qwen3-1.7B matched ReN

实验：`ren_balanced_matched_b256_c0p5_8k_r4_s42_20260918`。Student 为 Qwen3-1.7B，Teacher 为 Qwen3-32B；四轮、每轮 256 个 prompt，共 1,024 次 prompt exposure。评测为同一批 32k generation，并从同一 token trajectory 截取 8k prefix 重评分。

| 指标 | Base | Round 2 | Round 4 |
|---|---:|---:|---:|
| 32k MATH-500 | 86.93% | 87.94% | 91.46% |
| 32k AIME25 | 40.00% | 36.67% | 43.33% |
| 32k OlympiadBench | 63.32% | 61.31% | 65.83% |
| 32k MMLU-Pro | 62.31% | 64.82% | 63.82% |
| 32k GPQA-Diamond | 38.38% | 37.88% | 37.37% |
| **32k 外部 macro** | **58.19%** | **57.72%** | **60.36%** |
| **8k 同轨迹 prefix macro** | **42.98%** | **43.05%** | **45.03%** |
| 32k DAPO dev128 | 67.97% | 71.09% | 67.19% |
| 8k DAPO dev128 | 42.97% | 45.31% | 39.06% |

Round 4 相对 Base：

- 32k macro：`+2.17 pp`，配对题目 bootstrap 95% CI `[-0.73, +5.22] pp`；
- 8k prefix macro：`+2.05 pp`，95% CI `[-0.33, +4.23] pp`；
- 32k 共 62 个 rescue、46 个 degradation，净多答对 16 题；
- 排除任一模型 hit-cap 的题目后，仍净多答对 15 题；
- 提升主要来自 MATH，GPQA 和独立 dev 没有同步改善；CI 跨零且只有一个训练 seed。

训练侧诊断：四轮 `E[w]≈0.009`，top-10% 位置承载约 95% weight mass，control/reasoning gradient-norm 比约 25–29%；capped rollout 占 50.49%，对应 prompt-balanced reasoning-loss 份额 46.92%。这些诊断支持“监督高度集中”，但不能单独证明 gate 的因果贡献。

必须抢救的原始目录：

```text
LuLu_outputs/experiments/ren_balanced_matched_b256_c0p5_8k_r4_s42_20260918/
```

优先文件：`REPORT.md`、`comparison.json`、`experiment_plan.json`、`live_progress.json`、`evaluation/eval_plan.json`、`evaluation/summary.json`、所有逐题 JSONL、`prefix_8192/`、`analysis_postrun/`、每轮 `metrics/` 和 `diagnostics/`、冻结 `code/` 与输入/source hash。

## 3. RQ1 已完成结果

### 3.1 Qwen3-1.7B No-Reasoning（Control + Ref.）

同一 1,752 题 32k evaluation：

| Benchmark | N | Base | Control + Ref. | Delta | Rescue | Degrade |
|---|---:|---:|---:|---:|---:|---:|
| MATH-500 | 500 | 88.00% | 88.00% | +0.00 pp | 17 | 17 |
| AIME25 | 30 | 36.67% | 40.00% | +3.33 pp | 3 | 2 |
| OlympiadBench | 512 | 66.60% | 65.04% | -1.56 pp | 27 | 35 |
| MMLU-Pro | 512 | 56.05% | 56.84% | +0.78 pp | 30 | 26 |
| GPQA-Diamond | 198 | 32.32% | 36.36% | +4.04 pp | 23 | 15 |
| **Macro** | — | **55.93%** | **57.25%** | **+1.32 pp** | — | — |
| **Micro** | 1,752 | **65.24%** | **65.53%** | **+0.29 pp** | — | — |

结果根目录：

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/no_reasoning_control_ref_qwen3_1p7b_pool2048_b256_r4_s42/evaluation_base_control_ref_round4_32k_max512
```

需要保留 `summary.json`、`eval_plan.json`（如存在）、`base/`、`control_ref_round4/` 和 evaluation 日志。

### 3.2 Qwen3-1.7B Vanilla OPD 与 OPSD

仓库记录确认两者均完成 Round 4 训练和单模型 32k evaluation，且与 1.7B Base/Control 使用同一配置签名：

```text
0dfa76050bc69a91778f7156c96daf596f89eef7062a89f7b29e30cdef40c995
```

题数为 `500/30/512/512/198`。但具体 benchmark accuracy 没有复制进本地文档，因此当前不能凭记忆补表。必须从以下文件取回：

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/vanilla_opd_qwen3_1p7b_pool2048_b256_r4_s42/evaluation_vanilla_opd_round4_32k_max512/summary.json
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/opsd_qwen3_1p7b_pool2048_b256_r4_s42/evaluation_opsd_round4_32k_max512/summary.json
```

正式做配对统计前，还要逐题核对 `prompt_index`、`prompt_sha256`、`generation_seed` 和 `ground_truth`，不能只因配置签名相同就直接合并。

### 3.3 Qwen3-4B Base / No-Reasoning / Vanilla OPD / OPSD

四个模型使用相同 1,752 个 benchmark 样本，共 7,008 条 generation。

| Benchmark | N | Base | No-Reasoning | Vanilla OPD | OPSD |
|---|---:|---:|---:|---:|---:|
| MATH-500 | 500 | 92.20% | 93.40% | 91.60% | 92.60% |
| AIME25 | 30 | 56.67% | 53.33% | 56.67% | 63.33% |
| OlympiadBench | 512 | 76.17% | 77.73% | 78.52% | 76.17% |
| MMLU-Pro | 512 | 69.14% | 67.58% | 67.58% | 70.12% |
| GPQA-Diamond | 198 | 52.02% | 53.03% | 52.53% | 49.49% |

| Model | Math Avg. | OOD Avg. | Macro | Micro | Δ Macro vs Base | Paired 95% CI | Rescue / Degrade |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 75.01% | 60.58% | 69.24% | 75.63% | — | — | — |
| No-Reasoning | 74.82% | 60.30% | 69.02% | 76.03% | -0.22 pp | [-4.40, +3.95] | 76 / 69 |
| Vanilla OPD | 75.59% | 60.05% | 69.38% | 75.74% | +0.14 pp | [-4.32, +4.41] | 70 / 68 |
| OPSD | 77.37% | 59.81% | 70.34% | 75.86% | +1.10 pp | [-2.89, +5.17] | 87 / 83 |

结果根目录：

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/qwen3_4b_rq1_baselines_mem024_seq4_pool2048_b256_r4_s42/evaluation_rq1_4b_base_control_ref_vanilla_opd_opsd_round4_32k_max512
```

需要保留该目录全部内容，以及实验根目录下的 `RQ1_4B_BASELINE_EVALUATION_RESULTS.md` 和 evaluation 日志。

## 4. 历史实验：只能单独报告，不能与主表混合

### 4.1 历史 12-round bounded ReN，8k sampling

实验：`ren_balanced_abs_full_qwen1p7_teacher32b_pool2048_dev256_r12_s42`。该实验使用不同训练量、旧 parser/评估口径和 8k 独立 generation。

| 指标 | Base | Round 8 | Final 12 |
|---|---:|---:|---:|
| 外部五集 macro | 46.24% | 47.77% | 48.15% |
| Pooled accuracy | 51.64% | 52.12% | 53.94% |
| Hit-cap | 36.73% | 35.52% | 33.94% |

Round 8 相对 Base 为 `+1.54 pp`，95% CI `[-0.95,+4.31]`；Final 12 为 `+1.91 pp`，95% CI `[-0.79,+4.66]`。final-only 重评分 macro 为 `42.95/44.12/44.36%`。

同一 checkpoint 的 38,912-token 重新生成没有复现正向趋势：Base/Round 8/Final 12 macro 为 `60.30/58.69/57.22%`。这应作为预算敏感性或负结果报告，不能只选择 8k 的正向点估计。

### 4.2 早期 32-update decisive run

实验：`ren_qwen1p7_teacher32b_pool2048_s42`。中间节点统一补评记录的 macro 为：

| Base | Step 20 | Step 32 |
|---:|---:|---:|
| 46.03% | 45.99% | 44.69% |

该结果是早期配方和评测执行，后期下降主要来自 MMLU-Pro。它只能作为开发历史/负结果，不应冒充 2026-09-18 matched recipe。

## 5. 已跑但尚不能形成最终 claim 的实验

### 5.1 RQ2.3 Teacher-strength sweep

已确认完成 8 个 Round-4 Student 训练 arm：4B/8B/14B/32B Teacher × Vanilla/ReN。旧状态文档仍把全矩阵 Student evaluation、Teacher standalone evaluation、四张汇总表和 fixed-prefix policy drift 标为待完成。

较新的登记又出现了下面的 Student evaluation：

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/rq2_teacher_sweep_qwen3_1p7b_pool2048_b256_r4_s42_6gpu/evaluation_rq23_students_round4_32k_five_bench_s42/summary.json
```

以及 `qwen32b_ren/` 逐题目录，配置签名为：

```text
28e7af4ee9ff83f311dc144afdc2e0c94f7306232416d9de3463cfbe1dd0727e
```

其五集题数为 `199/30/199/199/198`，不能与 RQ1 的 1,752 题表直接混合。当前本地没有复制该 summary 的数值，也没有证据确认 9 个 Student 和 4 个 Teacher 是否全部评完。因此 rebuttal 中目前只能写“8/8 训练 arm 完成”，不能写“完整 Teacher-gap sweep 完成”。

如果 Round 0–4 checkpoint 已删除，原计划的 fixed-prefix policy drift 无法从 summary 或逐题 generation 恢复。这一点必须明确登记为不可恢复分析，除非仍有 checkpoint 副本。

### 5.2 Pass@16

seed 42–52 共 11 个 seed 已逐 seed 验收。seed 53–57 的倒序任务在 0/8 seed 时因缺少 `latex2sympy2_extended` 失败。因此当前只有 11 个有效 sample，不能称为 Pass@16，也不能把局部 summary 平均后代替逐题 OR。

### 5.3 只有计划/代码、没有正式结果证据

- RQ2.1 Uniform-Matched / Shuffled-ReN / Causal-Matched；
- RQ3.2 2k/4k/8k/16k/32k prefix curve（主 matched run 只有已汇总的 8k 与 32k）；
- Qwen3-4B matched ReN 正式复现；
- Nemotron cross-family；
- GRPO baseline；
- current gate diagnostics、shared/certified correction、semantic utility audit 的最终报告数值。

除非从集群找到对应 `REPORT.md`、`comparison.json`、逐题输出或最终 summary，否则这些不能列为“实验结果”。

## 6. checkpoint 删除后的最小证据集

对于已经完成 evaluation 的实验，按下面优先级抢救。P0 缺失时，rebuttal 数字的可审计性会明显下降。

### P0：必须保留

1. `summary.json`、`comparison.json`、`REPORT.md` 和所有 CSV；
2. 每道题的原始 JSONL/shard，包括完整 response text、`response_token_ids`、prediction、reward/correct、gold、benchmark/source index、prompt hash、generation seed、response length、hit-cap；
3. `eval_plan.json`、data/evaluation manifest、实际 benchmark 行或其不可变版本/hash；
4. parser 文件及 hash、评分脚本、冻结代码快照或 git commit；
5. `experiment_plan.json`、实际命令、模型精确 revision、tokenizer revision、随机种子和解码参数；
6. 重评分审计，例如 `rescore_audit.json`，保留旧分数和新分数，不能覆盖原始视图。

### P1：强烈建议保留

1. 每轮 `metrics/round_*.json`；
2. `diagnostics/round_*/position_scores.npz`、`trajectory_scores.jsonl`、`summary.json`；
3. training/evaluation worker logs、timing JSON、`live_progress.json`；
4. rollout JSONL、训练数据 manifest/hash、实际 prompt schedule；
5. 环境信息：依赖 lock、CUDA/PyTorch/vLLM 版本、GPU 型号。

### P2：checkpoint 已删后的替代记录

1. 保存“删除了哪些 checkpoint、何时删除、原路径、删除原因”的 ledger；
2. 若还存在 `lulu_state.json`、`config.json`、optimizer/训练状态的非权重副本，一并保留；
3. 为保留下来的每个文件生成 SHA256 manifest；
4. 将 P0/P1 复制到与训练集群独立的持久存储，并至少保留两份。

仅有 Markdown 表格不足以支撑重新评分；仅有逐题输出而没有 parser/manifest，也不足以证明评测口径。最稳妥的 rebuttal 证据单位是“逐题输出 + 冻结评测计划 + 冻结评分代码 + 汇总及审计”。

## 7. 集群回收顺序

1. 先只读检查本文件列出的所有绝对路径是否仍存在；
2. 立即复制所有 evaluation、analysis、metrics、diagnostics、rollouts、logs 和冻结配置，暂时不要再清理任何目录；
3. 对复制后的文件生成 SHA256 manifest，并记录缺失项；
4. 从 1.7B Vanilla/OPSD 与 RQ2.3 `summary.json` 回填本总账；
5. 用逐题键做去重和配对审计，再生成最终 rebuttal 表格；
6. 最后才决定哪些剩余大文件可以删除。

## 8. 本总账的本地证据来源

- `README.md` 第 16–17 节：2026-09-18 matched ReN 的评测协议和结果；
- `md/RQ1_COMPLETED_EXPERIMENTS.md`：1.7B/4B RQ1、结果路径和 Pass@16 状态；
- `md/RQ2_3_EXPERIMENT_AND_EVALUATION_STATUS.md`：RQ2.3 训练矩阵及未完成项；
- `md/historical_balanced_8k.md`：历史 12-round 结果及长预算负结果；
- `md/README_before_recipe_20260918.md`：早期 32-update run 的补评结果；
- `EXPERIMENT_RUN_PLAN.md`：论文实验取舍与依赖关系。

