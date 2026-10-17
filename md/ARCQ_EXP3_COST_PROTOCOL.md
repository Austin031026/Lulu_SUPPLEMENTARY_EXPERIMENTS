# ArcQ 实验三：成本—准确率联合评估

目的：测量 HinD 相对于 OPD、OPSD 的计算成本和准确率变化，不预设结果。
准确率和成本必须来自同一次训练的 checkpoint。

## 固定协议

- Student Qwen3-1.7B，Teacher Qwen3-32B。
- 三组均使用现有 `ren_balanced` control/reference 骨架：
  `vanilla_opd`、`hind`、`opsd` 对应 `reasoning_ablation=vanilla/ren/opsd`。
  这是匹配骨架比较，不能冒称三个独立原始实现。
- 历史SHA256固定的2048题训练池，seed42，4轮×256题，每题1条rollout，
  每轮1次全局更新；保存初始化及每轮checkpoint。
- 训练生成上限8192 tokens；评测上限32768 tokens。
- 五项Pass@1：MATH-500(500)、AIME25(30)、OlympiadBench(512)、
  MMLU-Pro(1400源题中的前512题)、GPQA-Diamond(198)，各组题序一致。
- 同一组8张空闲同型号GPU，顺序运行三组算法。
  Student占0–4，hindsight/reference占5，Teacher占6–7。
  OPSD不运行Teacher，但6–7仍为预留资源并计入GPU-hours。
  结果表示固定资源配置成本，不是各算法独立优化后的最低成本。

## 一键运行

代码须先同步到新集群 `$ARCQ_ROOT/code/Lulu_SUPPLEMENTARY_EXPERIMENTS`。
使用 `envs/lulu_probe_qwen3/bin/python`，读取 `env/arcq_env.sh`，不自动安装依赖。

```bash
bash "$ARCQ_ROOT/code/Lulu_SUPPLEMENTARY_EXPERIMENTS/runs/run_arcq_exp3.sh"
```

默认输出 `experiments/exp3_cost/suite_<timestamp>/`，或追加
`--output-dir "$ARCQ_ROOT/experiments/exp3_cost/my_run"`。
必须通过 `bash` 运行脚本，不要source；失败不会退出调用者终端。

流程：资产预检 → 冻结源码/命令/输入哈希 → 三组真实GPU小规模训练测试 →
三组正式训练和各自评测 → 原始记录重算 → 汇总。

只生成计划而不运行训练：

```bash
"$ARCQ_ROOT/envs/lulu_probe_qwen3/bin/python" \
  "$ARCQ_ROOT/code/Lulu_SUPPLEMENTARY_EXPERIMENTS/scripts/run_arcq_exp3.py" \
  --arcq-root "$ARCQ_ROOT"
```

任何smoke失败都阻止正式训练。失败目录保留，不自动覆盖或恢复半成品；
修复后指定新目录。重复指定原目录只跳过监控验证通过的已完成阶段。
同一工作区不允许两个本启动器同时调度训练。

## 统计口径

| 问题 | 指标 |
| --- | --- |
| 训练总成本 | 单调时钟端到端训练墙钟时间×8/3600；包括加载、同步、存盘和收尾，不包括评测/smoke |
| 时间分布 | 生成、生成请求整体、causal scoring、Teacher、hindsight、reference、权重计算、更新、checkpoint；worker保留sum和max |
| 生成量 | 由保存的response_ids重算token数；辅助指标为每百万生成tokens的GPU-hours |
| 显存 | 每卡采样device峰值；另列Student PyTorch allocated/reserved峰值 |
| 收益 | 各benchmark准确率、五项等权macro、相对OPD的百分点差 |
| 实际总开销 | 训练、评测、smoke及失败尝试分别留档；另列训练+评测成本 |

并行worker耗时之和是工作量，不是端到端时间；重叠阶段不能相加当总时间。
生成请求含权重刷新等开销，不等同纯生成；pipeline包含生成和评分。
device采样可能漏掉瞬时显存峰值，不能称为精确硬件最大值。

cost-mode保留损失/梯度，关闭可选梯度诊断和画图。
OPD跳过不参与其目标的hindsight计算；匹配OPSD跳过不参与其目标的Teacher计算。
CPU数值回归测试检查三组改动前后的权重、loss、梯度一致性。

## 输出与实验一复用

以下路径均相对于suite目录：

- `experiment_plan.json`：冻结命令、输入/源码哈希、原checkout Git信息。
- `code/`：执行源码快照；`status.json`：阶段或错误；`logs/`：每阶段日志。
- `runs/<algorithm>/train/`：配置、逐轮指标、rollouts和checkpoints。
- `runs/<algorithm>/telemetry/`：训练原始GPU采样、manifest、summary。
- `runs/<algorithm>/evaluation/`：评测计划、逐题结果和summary。
- `runs/<algorithm>/evaluation_telemetry/`：独立评测成本。
- `smoke/`：小规模训练记录；`attempt_costs.json`：含失败/未知成本的尝试清单。
- `analysis/REPORT.md`、`summary.json`、`summary.tsv`：主报告。
- `analysis/phase_seconds.tsv`、`benchmark_accuracy.tsv`：阶段耗时和各benchmark准确率。

实验一复用 `runs/vanilla_opd/train/checkpoints/round_000000` 和 `round_000004`
（以及需要的中间轮次）。留出题、固定前缀和预测关系分析需另行实现，
不能把这里的评测结果当作实验一诊断数据。

## 自动验收与边界

要求四轮完整且无跳过更新、final checkpoint存在、tokens与轨迹一致、五项题数/答案对齐，
逐题reward重算准确率与summary/macro一致。检查命令、冻结源码、输入数据、
科学配置、硬件、评测协议是否可比。缺失、NaN或监控异常不能默认为0或宣告成功。

重算验证保存reward的汇总一致性，不代表重新执行数学判题器。
模型预检检查safetensors可读和配置/索引，不独立证明Hub revision或全部权重来源。
单seed是一次受控测量，不提供跨seed显著性结论。
本地CPU测试不能替代新集群真实GPU测试；最终以 `ARCQ_EXP3=PASS` 为验收依据。
