# RQ1 已完成实验汇总

## 当前完成状态

| 项目 | 类型 | 训练状态 | Evaluation 状态 | 说明 |
|---|---|---|---|---|
| Qwen3-1.7B Base | 未训练对照模型 | 不需要训练 | 已完成 | 后续 Vanilla OPD 和 OPSD 可复用本次 Base 完整逐题结果 |
| No-Reasoning（Control + Ref.） | Backbone control / ablation | 已完成 Round 4 | 已完成 | `L_R=0`，`L=0.5 L_control+0.1 L_ref` |
| Vanilla OPD | RQ1 baseline | 已完成 Round 4 | 已完成 | 单模型 evaluation 已完成；与 Base 的配对比较应复用相同逐题结果 |
| OPSD | RQ1 baseline | 已完成 Round 4 | 已完成 | 单模型 evaluation 已完成；与 Base 的配对比较应复用相同逐题结果 |

## 已完成 Evaluation

本次 evaluation 比较了以下两个模型：

1. **Base**：未经后训练的 Qwen3-1.7B。
2. **Control+Ref Round 4**：完成四轮训练的 No-Reasoning（Control + Ref.）checkpoint。

| Benchmark | N | Base Acc | Control+Ref | Delta (pp) | Rescue | Degrade | RespTok | HitCap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| math500 | 500 | 88.00% | 88.00% | +0.00 | 17 | 17 | 5948.6 | 1.00% |
| aime25 | 30 | 36.67% | 40.00% | +3.33 | 3 | 2 | 17305.4 | 3.33% |
| olympiadbench | 512 | 66.60% | 65.04% | -1.56 | 27 | 35 | 11171.0 | 1.95% |
| mmlu_pro | 512 | 56.05% | 56.84% | +0.78 | 30 | 26 | 3289.8 | 0.00% |
| gpqa_diamond | 198 | 32.32% | 36.36% | +4.04 | 23 | 15 | 8039.2 | 0.51% |

| 汇总指标 | Base | Control+Ref | Delta |
|---|---:|---:|---:|
| Macro accuracy | 55.93% | 57.25% | +1.32 pp |
| Micro accuracy | 65.24% | 65.53% | +0.29 pp |
| Evaluated examples/model | 1752 | 1752 | — |

## 集群结果路径

实验根目录：

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/no_reasoning_control_ref_qwen3_1p7b_pool2048_b256_r4_s42
```

最终训练 checkpoint：

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/no_reasoning_control_ref_qwen3_1p7b_pool2048_b256_r4_s42/training/checkpoints/round_000004
```

Evaluation 目录：

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/no_reasoning_control_ref_qwen3_1p7b_pool2048_b256_r4_s42/evaluation_base_control_ref_round4_32k_max512
```

最终汇总文件：

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/no_reasoning_control_ref_qwen3_1p7b_pool2048_b256_r4_s42/evaluation_base_control_ref_round4_32k_max512/summary.json
```

Base 完整逐题结果：

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/no_reasoning_control_ref_qwen3_1p7b_pool2048_b256_r4_s42/evaluation_base_control_ref_round4_32k_max512/base
```

Control+Ref 完整逐题结果：

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/no_reasoning_control_ref_qwen3_1p7b_pool2048_b256_r4_s42/evaluation_base_control_ref_round4_32k_max512/control_ref_round4
```

Evaluation 日志：

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/no_reasoning_control_ref_qwen3_1p7b_pool2048_b256_r4_s42/logs/eval_base_control_ref_round4_32k_max512.log
```

## Base 结果复用规则

后续 Vanilla OPD 和 OPSD evaluation 不需要重新运行 Base，但必须复用上述 Base 的完整逐题结果，而不是只复用汇总 accuracy。复用时必须保持同一 evaluation manifest、题目顺序、prompt、gold answer、32k 解码设置、per-question seed、parser 和评分代码。

如果上述任一配置发生变化，则必须重新运行 Base evaluation。

## 当前结论

No-Reasoning（Control + Ref.）相对 Base 的 Macro Accuracy 提升 **+1.32 pp**，Micro Accuracy 提升 **+0.29 pp**。主要提升来自 AIME25 和 GPQA-Diamond，OlympiadBench 有所下降。该方法应标记为 backbone control / ablation，而不是文献 baseline。

## Qwen3-4B RQ1 baseline 训练状态

Qwen3-4B / Qwen3-32B 的三个新增 RQ1 训练 arm 已于 2026-09-20 完成。流水线状态为 `TRAINING_COMPLETE arms=3/3`，每个 arm 均完成 4 轮、4 次全局参数更新，并生成可评估的 `round_000004` 最终 checkpoint。原始 Qwen3-4B Base 不进行训练；ReN 使用已有结果，不在本次 baseline 流水线中重复训练。

实验根目录：

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/qwen3_4b_rq1_baselines_mem024_seq4_pool2048_b256_r4_s42
```

| 论文名称 | 内部 arm | 训练状态 | 最终 checkpoint |
|---|---|---|---|
| No-Reasoning（Control + Ref.） | `control_ref` | 已完成 Round 4 / 4 | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/qwen3_4b_rq1_baselines_mem024_seq4_pool2048_b256_r4_s42/arms/control_ref/train/checkpoints/round_000004` |
| Vanilla OPD | `vanilla_opd` | 已完成 Round 4 / 4 | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/qwen3_4b_rq1_baselines_mem024_seq4_pool2048_b256_r4_s42/arms/vanilla_opd/train/checkpoints/round_000004` |
| OPSD | `opsd` | 已完成 Round 4 / 4 | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/qwen3_4b_rq1_baselines_mem024_seq4_pool2048_b256_r4_s42/arms/opsd/train/checkpoints/round_000004` |

已核验三个最终 checkpoint 均包含 `lulu_state.json`、`config.json` 和模型 safetensor 权重。训练指标和诊断使用零基轮次编号：第 1–4 轮分别保存为 `metrics/round_0000.json`–`round_0003.json` 和 `diagnostics/round_0000/`–`round_0003/`；最终模型 checkpoint 使用提交后编号 `round_000004`。

## Qwen3-4B RQ1 baseline evaluation

Qwen3-4B Base、No-Reasoning、Vanilla OPD 和 OPSD 已完成统一的 32k thinking evaluation。四个模型使用相同 benchmark 行、prompt、gold、逐题 seed 和 final-only parser；每个模型评估 1,752 题，共生成 7,008 条结果。

| Benchmark | N | Qwen3-4B Base | No-Reasoning | Vanilla OPD | OPSD |
|---|---:|---:|---:|---:|---:|
| MATH-500 | 500 | 92.20% | 93.40% | 91.60% | 92.60% |
| AIME25 | 30 | 56.67% | 53.33% | 56.67% | 63.33% |
| OlympiadBench | 512 | 76.17% | 77.73% | 78.52% | 76.17% |
| MMLU-Pro | 512 | 69.14% | 67.58% | 67.58% | 70.12% |
| GPQA-Diamond | 198 | 52.02% | 53.03% | 52.53% | 49.49% |

| Model | Math Avg | OOD Avg | Macro Acc | Micro Acc | Delta Macro vs Base | Paired Macro 95% CI | Rescue | Degrade | Mean RespTok | HitCap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-4B Base | 75.01% | 60.58% | 69.24% | 75.63% | — | — | — | — | 6648.0 | 1.14% |
| No-Reasoning | 74.82% | 60.30% | 69.02% | 76.03% | -0.22 pp | [-4.40, +3.95] pp | 76 | 69 | 6647.1 | 1.14% |
| Vanilla OPD | 75.59% | 60.05% | 69.38% | 75.74% | +0.14 pp | [-4.32, +4.41] pp | 70 | 68 | 6698.5 | 1.26% |
| OPSD | 77.37% | 59.81% | 70.34% | 75.86% | +1.10 pp | [-2.89, +5.17] pp | 87 | 83 | 6189.2 | 1.14% |

Evaluation 根目录：

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/qwen3_4b_rq1_baselines_mem024_seq4_pool2048_b256_r4_s42/evaluation_rq1_4b_base_control_ref_vanilla_opd_opsd_round4_32k_max512
```

最终汇总文件：

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/qwen3_4b_rq1_baselines_mem024_seq4_pool2048_b256_r4_s42/evaluation_rq1_4b_base_control_ref_vanilla_opd_opsd_round4_32k_max512/summary.json
```

Evaluation plan：

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/qwen3_4b_rq1_baselines_mem024_seq4_pool2048_b256_r4_s42/evaluation_rq1_4b_base_control_ref_vanilla_opd_opsd_round4_32k_max512/eval_plan.json
```

Evaluation 主日志：

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/qwen3_4b_rq1_baselines_mem024_seq4_pool2048_b256_r4_s42/logs/eval_rq1_4b_base_control_ref_vanilla_opd_opsd_round4_32k_max512.log
```

Markdown 结果报告：

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/qwen3_4b_rq1_baselines_mem024_seq4_pool2048_b256_r4_s42/RQ1_4B_BASELINE_EVALUATION_RESULTS.md
```

逐题结果目录：

| Model | 逐题 evaluation 目录 |
|---|---|
| Qwen3-4B Base | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/qwen3_4b_rq1_baselines_mem024_seq4_pool2048_b256_r4_s42/evaluation_rq1_4b_base_control_ref_vanilla_opd_opsd_round4_32k_max512/base` |
| No-Reasoning Round 4 | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/qwen3_4b_rq1_baselines_mem024_seq4_pool2048_b256_r4_s42/evaluation_rq1_4b_base_control_ref_vanilla_opd_opsd_round4_32k_max512/control_ref_round4` |
| Vanilla OPD Round 4 | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/qwen3_4b_rq1_baselines_mem024_seq4_pool2048_b256_r4_s42/evaluation_rq1_4b_base_control_ref_vanilla_opd_opsd_round4_32k_max512/vanilla_opd_round4` |
| OPSD Round 4 | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/qwen3_4b_rq1_baselines_mem024_seq4_pool2048_b256_r4_s42/evaluation_rq1_4b_base_control_ref_vanilla_opd_opsd_round4_32k_max512/opsd_round4` |

## 快速查询索引

| Student | Method | Final checkpoint | Evaluation summary / result | 登记状态 |
|---|---|---|---|---|
| Qwen3-1.7B | Base | 不训练 | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/no_reasoning_control_ref_qwen3_1p7b_pool2048_b256_r4_s42/evaluation_base_control_ref_round4_32k_max512/summary.json` | 已登记 |
| Qwen3-1.7B | No-Reasoning | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/no_reasoning_control_ref_qwen3_1p7b_pool2048_b256_r4_s42/training/checkpoints/round_000004` | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/no_reasoning_control_ref_qwen3_1p7b_pool2048_b256_r4_s42/evaluation_base_control_ref_round4_32k_max512/summary.json` | 已登记 |
| Qwen3-1.7B | Vanilla OPD | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/vanilla_opd_qwen3_1p7b_pool2048_b256_r4_s42/training/checkpoints/round_000004` | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/vanilla_opd_qwen3_1p7b_pool2048_b256_r4_s42/evaluation_vanilla_opd_round4_32k_max512/summary.json` | 已完成；与 Base 的 eval 配置签名一致，逐题配对待审计 |
| Qwen3-1.7B | OPSD | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/opsd_qwen3_1p7b_pool2048_b256_r4_s42/training/checkpoints/round_000004` | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/opsd_qwen3_1p7b_pool2048_b256_r4_s42/evaluation_opsd_round4_32k_max512/summary.json` | 已完成；与 Base 的 eval 配置签名一致，逐题配对待审计 |
| Qwen3-1.7B | ReN（Qwen3-32B Teacher，RQ2.3） | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/rq2_teacher_sweep_qwen3_1p7b_pool2048_b256_r4_s42_6gpu/arms/qwen32b_ren/train/checkpoints/round_000004` | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/rq2_teacher_sweep_qwen3_1p7b_pool2048_b256_r4_s42_6gpu/evaluation_rq23_students_round4_32k_five_bench_s42/summary.json` | RQ2.3 已完成；评估每模型约 825 题，不能直接用于 RQ1 的 1752 题表 |
| Qwen3-4B | Base | 不训练 | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/qwen3_4b_rq1_baselines_mem024_seq4_pool2048_b256_r4_s42/evaluation_rq1_4b_base_control_ref_vanilla_opd_opsd_round4_32k_max512/summary.json` | 已登记 |
| Qwen3-4B | No-Reasoning | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/qwen3_4b_rq1_baselines_mem024_seq4_pool2048_b256_r4_s42/arms/control_ref/train/checkpoints/round_000004` | 同一 4B `summary.json` | 已登记 |
| Qwen3-4B | Vanilla OPD | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/qwen3_4b_rq1_baselines_mem024_seq4_pool2048_b256_r4_s42/arms/vanilla_opd/train/checkpoints/round_000004` | 同一 4B `summary.json` | 已登记 |
| Qwen3-4B | OPSD | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/qwen3_4b_rq1_baselines_mem024_seq4_pool2048_b256_r4_s42/arms/opsd/train/checkpoints/round_000004` | 同一 4B `summary.json` | 已登记 |
| Qwen3-4B | ReN | 未在本次 Feng_J 扫描范围内找到 | 未找到 | 需要其他 workspace 的路径或另行核实 |

### Qwen3-1.7B 单模型逐题目录

| Method | Evaluation 逐题目录 | 状态 |
|---|---|---|
| Vanilla OPD | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/vanilla_opd_qwen3_1p7b_pool2048_b256_r4_s42/evaluation_vanilla_opd_round4_32k_max512/vanilla_opd_round4` | `summary.json` 已生成 |
| OPSD | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/opsd_qwen3_1p7b_pool2048_b256_r4_s42/evaluation_opsd_round4_32k_max512/opsd_round4` | `summary.json` 已生成 |
| ReN（RQ2.3，32B Teacher） | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/rq2_teacher_sweep_qwen3_1p7b_pool2048_b256_r4_s42_6gpu/evaluation_rq23_students_round4_32k_five_bench_s42/qwen32b_ren` | 不同题数的独立评估，不与上两项混为 RQ1 配对结果 |

本次扫描中的 1.7B Base、No-Reasoning、Vanilla OPD、OPSD 的 evaluation 配置签名均为 `0dfa76050bc69a91778f7156c96daf596f89eef7062a89f7b29e30cdef40c995`，五个 benchmark 的题数均为 500/30/512/512/198。配置签名相同只说明计划参数一致；正式合并配对统计仍需逐题核对 `prompt_index`、`prompt_sha256`、`generation_seed` 与 `ground_truth`。RQ2.3 Teacher sweep 的签名为 `28e7af4ee9ff83f311dc144afdc2e0c94f7306232416d9de3463cfbe1dd0727e`，题数为 199/30/199/199/198，因此不能直接与 RQ1 的完整评估混表。已废弃的 Vanilla OPD `.failed_20260919_135543` 目录不作为有效结果。

## Qwen3-1.7B RQ1 Pass@16：双任务分流（进行中，尚未汇总）

四个模型为 Base、No-Reasoning Round 4、Vanilla OPD Round 4、OPSD Round 4。每个 seed 对五个 benchmark 的 1,752 题各生成一次；目标 seed 为 42–57，共 16 次。两个任务必须使用同一训练后 checkpoint、评估数据和解码参数；6 卡任务从 57 倒序至 50，直接复用 8 卡任务 `sample_00_seed_42/eval_plan.json` 的配置，仅更改 GPU 列表、seed 和输出目录。

| 任务 | 计划 seed | 任务根目录 | 每个 seed 的结果与日志 |
|---|---|---|---|
| 原 8 卡正序任务 | **正式采用 42–52（11 seeds，已逐 seed 验收）** | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/rq1_qwen3_1p7b_pass16_32k_five_bench_s42` | `sample_00_seed_42` 至 `sample_10_seed_52`；每个目录包含 `summary.json`、`eval_plan.json` 和四模型逐题 shards。Controller 已停止，用户复查时没有 seed-53 残留进程。 |
| 新 6 卡倒序任务 | **正式采用 53–57（5 seeds；完成状态以目录验收为准）** | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/rq1_qwen3_1p7b_pass16_32k_five_bench_s42/reverse_6gpu` | `sample_11_seed_53` 至 `sample_15_seed_57`；控制日志 `logs/controller_*.log`（当前日志路径见 `logs/current_log_path.txt`），状态 `reverse_status.json`。 |

正式分流清单为 `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/rq1_qwen3_1p7b_pass16_32k_five_bench_s42/PASS16_SPLIT_MANIFEST.json`。该清单明确只读取 8 卡 seed 42–52 和 6 卡 seed 53–57；8 卡曾启动但未完成的重复 seed 53 已列入 `excluded`，清单记录其 partial 目录应已删除，最终合并前仍需以文件系统检查确认。最终只接受两侧合计覆盖 seed 42–57 且每个模型/benchmark/题目/seed 恰好一条的逐题记录；先核对 `prompt_index`、题目与答案、`generation_seed` 和评估计划，再对 16 次结果做逐题 OR 统计 Pass@16。不能把两组局部统计取平均。每个 seed 的 `summary.json` 只是该 seed 的 pass@1 汇总，不是最终 Pass@16。

2026-09-21 01:19（用户提供的监控）：6 卡倒序控制器在 seed 57、0/8 seed、0/4 模型时因当前 Python 环境缺少 `latex2sympy2_extended` 导入失败，尚无该任务的有效生成结果。新仓库 `pyproject.toml` 的 `eval` extra 声明 `latex2sympy2-extended>=1.11.0`。需在 6 卡节点实际使用的 Python 环境中补齐依赖、验证 `lulu.benchmark_parser` 可导入，再重新启动倒序任务；失败目录若存在，须先确认其内容，不能直接混入重跑结果。
