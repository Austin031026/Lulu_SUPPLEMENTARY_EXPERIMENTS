# RQ2.3 Teacher-Strength Sweep：训练与 Evaluation 状态

更新时间：2026-09-20

## 当前状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| 8 个 Teacher × Method 训练 arm | **已完成（8/8）** | 4B/8B/14B/32B Teacher，各包含 Vanilla 与 ReN |
| 最终 Student checkpoint 验收 | **已完成** | 每个 arm 均完成 4 轮、4 次非跳过 optimizer update |
| Student benchmark evaluation | **待完成** | Base + 8 个 Round-4 Student checkpoint，使用完全一致的逐题设置 |
| Teacher standalone evaluation | **待完成** | 4B/8B/14B/32B Teacher，用于测量实际 Teacher capability gap |
| RQ2.3 汇总表 | **待完成** | `performance.csv`、`teacher_capability.csv`、`round_dynamics.csv`、`resolved_distribution_quantiles.csv` |
| Fixed-prefix policy drift | **待完成** | 无新生成；需要 Round 0–4 checkpoint，输出 `policy_drift.csv` |

## 集群实验根目录

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/rq2_teacher_sweep_qwen3_1p7b_pool2048_b256_r4_s42_6gpu
```

训练数据：

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/data/dapo_pool2048_s42/train.jsonl
```

Base Student：

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Rona_Lulu/Lulu_outputs/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
```

续跑计划和最终状态：

```text
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/rq2_teacher_sweep_qwen3_1p7b_pool2048_b256_r4_s42_6gpu/remaining_training_plan.json
/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/rq2_teacher_sweep_qwen3_1p7b_pool2048_b256_r4_s42_6gpu/remaining_training_status.json
```

## 八个最终 Student checkpoint

| Teacher | Method | 训练状态 | Evaluation 状态 | Round-4 checkpoint |
|---|---|---|---|---|
| Qwen3-4B | Vanilla | 已完成 | 待完成 | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/rq2_teacher_sweep_qwen3_1p7b_pool2048_b256_r4_s42_6gpu/arms/qwen4b_vanilla/train/checkpoints/round_000004` |
| Qwen3-4B | ReN | 已完成 | 待完成 | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/rq2_teacher_sweep_qwen3_1p7b_pool2048_b256_r4_s42_6gpu/arms/qwen4b_ren/train/checkpoints/round_000004` |
| Qwen3-8B | Vanilla | 已完成 | 待完成 | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/rq2_teacher_sweep_qwen3_1p7b_pool2048_b256_r4_s42_6gpu/arms/qwen8b_vanilla/train/checkpoints/round_000004` |
| Qwen3-8B | ReN | 已完成 | 待完成 | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/rq2_teacher_sweep_qwen3_1p7b_pool2048_b256_r4_s42_6gpu/arms/qwen8b_ren/train/checkpoints/round_000004` |
| Qwen3-14B | Vanilla | 已完成 | 待完成 | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/rq2_teacher_sweep_qwen3_1p7b_pool2048_b256_r4_s42_6gpu/arms/qwen14b_vanilla/train/checkpoints/round_000004` |
| Qwen3-14B | ReN | 已完成 | 待完成 | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/rq2_teacher_sweep_qwen3_1p7b_pool2048_b256_r4_s42_6gpu/arms/qwen14b_ren/train/checkpoints/round_000004` |
| Qwen3-32B | Vanilla | 已完成 | 待完成 | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/rq2_teacher_sweep_qwen3_1p7b_pool2048_b256_r4_s42_6gpu/arms/qwen32b_vanilla/train/checkpoints/round_000004` |
| Qwen3-32B | ReN | 已完成 | 待完成 | `/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J/Lulu_outputs/experiments/rq2_teacher_sweep_qwen3_1p7b_pool2048_b256_r4_s42_6gpu/arms/qwen32b_ren/train/checkpoints/round_000004` |

对应训练目录统一为：

```text
<实验根目录>/arms/<arm>/train
```

训练日志统一为：

```text
<实验根目录>/logs/training_<arm>.log
```

## Checkpoint 保留要求

在 fixed-prefix policy drift 完成前，八个 arm 的以下 checkpoint 均不得删除：

```text
<实验根目录>/arms/<arm>/train/checkpoints/round_000000
<实验根目录>/arms/<arm>/train/checkpoints/round_000001
<实验根目录>/arms/<arm>/train/checkpoints/round_000002
<实验根目录>/arms/<arm>/train/checkpoints/round_000003
<实验根目录>/arms/<arm>/train/checkpoints/round_000004
```

同时保留每轮的：

```text
metrics/round_xxxx.json
diagnostics/round_xxxx/position_scores.npz
diagnostics/round_xxxx/trajectory_scores.jsonl
diagnostics/round_xxxx/summary.json
rollouts/round_xxxx/*.jsonl
```

## RQ2.3 Evaluation benchmark

README 规定的五个论文主 benchmark：

| 分组 | Benchmark | 主要用途 |
|---|---|---|
| Math | MATH-500 | 数学准确率与 Math Avg. |
| Math | OlympiadBench | 数学准确率与 Math Avg. |
| Math | AIME25 | 数学准确率与 Math Avg. |
| OOD | MMLU-Pro | OOD/general reasoning 与 OOD Avg. |
| OOD | GPQA-Diamond | OOD/general reasoning 与 OOD Avg. |

RQ2.3 的论文结果严格使用上述五个 benchmark。`dapo_dev128` 不属于 README 为 RQ2.3 规定的论文 benchmark，不计入本实验的主结果或完成条件。

### Student evaluation 模型

- fresh Qwen3-1.7B Base；
- 上表中的 8 个 Round-4 Student checkpoint。

### Teacher standalone evaluation 模型

- Qwen3-4B；
- Qwen3-8B；
- Qwen3-14B；
- Qwen3-32B。

### 公平性与保存要求

- 所有模型使用相同 benchmark rows、prompt hashes、gold answers 和逐题 seed；
- 使用相同 final-only parser；
- primary evaluation 使用 32768-token sampled generation；
- 保存完整生成文本和 `response_token_ids`；
- 同时保存 prediction/reward、source index、prompt hash、seed、gold、response length 和 hit-cap；
- 后续 2k/4k/8k/16k prefix 分析必须从同一批 32k token IDs 重评分，不重新生成。

## Evaluation 输出路径规划

Student evaluation：

```text
<实验根目录>/evaluation/summary.json
```

Teacher standalone evaluation：

```text
<实验根目录>/teacher_evaluation/summary.json
```

RQ2.3 汇总：

```text
<实验根目录>/analysis/performance.csv
<实验根目录>/analysis/teacher_capability.csv
<实验根目录>/analysis/round_dynamics.csv
<实验根目录>/analysis/resolved_distribution_quantiles.csv
```

Fixed-prefix policy drift：

```text
<实验根目录>/analysis/policy_drift/policy_drift.csv
```

在 `evaluation/summary.json`、`teacher_evaluation/summary.json` 和上述分析文件实际生成前，不得将 RQ2.3 Evaluation 标记为完成。
