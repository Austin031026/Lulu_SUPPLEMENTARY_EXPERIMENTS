# LuLu 补充实验取舍

本文件按 `Lulu_SUPPLEMENTARY_EXPERIMENTS_README.md` 最后的 **Final paper output map** 编号，不按运行步骤（Step 1–7）编号。

## README 中实验数量的核对结果

README 顶层只有三个 Research Question：

1. RQ1 — Overall Effectiveness
2. RQ2 — Privileged Supervision Allocation
3. RQ3 — Scaling and Efficiency

不存在单独命名的 RQ4 或 RQ5。不过，README 的最终论文输出表把子问题拆成了六个正式实验项，因此本文用下面的六项作为“大实验”编号。

| 大实验编号 | README 对应项 | 内容 | 决定 | GPU 需求 |
|---:|---|---|---|---:|
| 1 | RQ1 | Base / No-Reasoning / Vanilla OPD / OPSD / ReN 的整体效果比较 | 保留 | 新训练 arm 各 8 卡 |
| 2 | RQ2.1 | Uniform-Matched / Shuffled-ReN / Causal-Matched / ReN allocation controls | 保留 | 新训练 arm 各 8 卡 |
| 3 | RQ2.2 | Causal mismatch 与 hindsight response 的 geometry/ranking 分析 | **不跑** | 原计划为 CPU 分析，无新训练 |
| 4 | RQ2.3 | 4B/8B/14B/32B Teacher × Vanilla/ReN Teacher-gap sweep | 保留 | 8 个训练 arm，各 8 卡 |
| 5 | RQ3.1 | 256/512/1024/2048 supervision-budget scaling | **不跑** | 原计划为 4 个训练 arm，各 8 卡 |
| 6 | RQ3.2 | 2k/4k/8k/16k/32k inference-budget prefix rescoring | 保留 | 0，CPU-only |

## 明确不运行

### 大实验 3：RQ2.2

不生成以下 geometry/ranking 分析：

- `dc_delta_quantiles.csv`
- `topk_overlap.csv`
- `D_C` 与 `D_C-D_H` 的 density/conditional-statistics 图
- ReN 排名与 causal-KL 排名的 top-k overlap 图

说明：RQ2.2 本来主要复用现有 ReN diagnostics，是 CPU-only 分析。RQ2.1 中的 `Causal-Matched` 是独立训练 control，仍归在大实验 2，不因为跳过 RQ2.2 而自动取消。

### 大实验 5：RQ3.1

不运行以下 supervision-scaling 训练：

- 256 unique prompts
- 512 unique prompts
- 1024 unique prompts
- 2048 unique prompts
- 可选的对应 Vanilla scaling matrix

也不需要生成 RQ3.1 的 `learning_curve.csv`、`benchmark_heatmap.csv`、`coverage.csv` 和 `compute_frontier.csv`。

## 保留的大实验

- 大实验 1：RQ1 overall-effectiveness baselines
- 大实验 2：RQ2.1 allocation controls
- 大实验 4：RQ2.3 Teacher-strength sweep
- 大实验 6：RQ3.2 inference-budget analysis

其中大实验 6 依赖已完成 evaluation 中保存的 `response_token_ids`，但不依赖大实验 5 的 scaling 训练；可以使用保留下来的 Base/ReN 或其他目标模型的 32k evaluation 结果做 prefix rescoring。

## 当前完成状态（2026-09-20）

| 实验 | 训练状态 | Evaluation 状态 | 后处理状态 | 记录 |
|---|---|---|---|---|
| RQ2.3 Teacher-strength sweep | **已完成：8/8 arm** | **待完成**：Base + 8 个 Student Round-4 checkpoint，以及 4 个 Teacher standalone evaluation | **待完成**：汇总表与 fixed-prefix policy drift | [`md/RQ2_3_EXPERIMENT_AND_EVALUATION_STATUS.md`](md/RQ2_3_EXPERIMENT_AND_EVALUATION_STATUS.md) |

RQ2.3 的 4B/8B/14B/32B Teacher × Vanilla/ReN 训练矩阵已经完成。八个 arm 的 Round 0–4 checkpoint 必须继续保留，直到 fixed-prefix policy drift 完成；当前只完成训练，不得将 RQ2.3 整体标记为已完成。

## 集群资源提醒

保留的训练 controller 仍假设 8 张 GPU 位于同一主机：5 张用于 Student DDP/rollout，1 张用于 Hindsight/Reference，2 张用于 Teacher TP。若目标集群每节点只有 4 张 GPU，必须先完成多节点适配或改用单节点 8 卡资源。
