# ReN：bounded weighting 与 reference-KL 稳定化

本轮按用户要求只运行这一个训练配置，不跑 Vanilla OPD baseline。从 Qwen3-1.7B Base 开始全参数训练；Teacher 为冻结 Qwen3-32B，thinking .6/.95/20，固定 DAPO 2048 题，每轮 64 条新 rollout，最多 32 次更新。

在同一轮冻结 Student S_r 的相同 rollout 前缀上，C 为普通输入，H 额外看到 gold final answer；Teacher T 始终 answer-blind。保持原始 hindsight-resolved 分数，改为有界分配：

\[
D_C=KL(q_T\Vert p_C),\quad R=D_C-KL(q_T\Vert p_H),\quad
\rho=\operatorname{clip}(R/(D_C+10^{-6}),0,1).
\]

实现将数值误差导致的负 D_C 钳至 0 后用于分母。rho 停止梯度，不除以 batch 内平均权重，不使用 top-percent hard mask。

结构区域完全依据真实 `<think>` / `</think>` token：思考内出现 `answer is`、boxed 等字符串不结束监督。reasoning 集合为 R；其余生成 token 为 control 集合 A，包括开始/结束 think、最终答案、实际采样 EOS/im_end。每个采样 token 都被覆盖，truncated rollout 不伪造 EOS。

令 N 为所有 DDP rank 的生成 token 总数，统一 loss 为：

\[
L=\frac1N\left[\sum_{t\in R}\rho_t KL(\mathrm{sg}(q_T)\Vert p_\theta)
+\sum_{t\in A}KL(\mathrm{sg}(q_T)\Vert p_\theta)
+0.1\sum_t KL(p_\theta\Vert\mathrm{sg}(p_{\rm ref}))\right].
\]

所有词表分布精确计算，按 128 个位置分块。Teacher、C/H、Reference 和 rho 均冻结；仅 Student 有梯度。Reference 是 **round0 初始 Student**，始终不刷新；Hindsight 则每轮刷新。即使 rho 全零，reference/control 损失仍保留。权重上界 1 不构成参数步长或策略变化的硬上界，仍需监控。

采用 [StableOPD §4.2](https://arxiv.org/html/2604.08527v1#S4.SS2) 的 KL(Student‖Reference) 思路。本实现保持全词表 forward-KL distillation，没有切换成论文的 policy-gradient estimator，也未新增 golden-data mixture；不是完整 StableOPD 复现。该论文观察到的失效现象可以作为动机，不能视作本实验失效原因已经得到证明。

本轮 lr=1e-6，AdamW、weight decay=0、grad clip=1；全部 1.7B 参数、梯度和 Adam 状态 FP32，BF16 autocast 计算。GPU0–4 常驻 5 个 Student DDP 副本及各自 vLLM rollout 引擎；GPU5 同时常驻更新轮次的 Hindsight 和不变的 Reference；GPU6–7 为 Teacher TP2。H/Reference 打分与 Teacher 打分并行；每轮提交完整 checkpoint 后刷新 rollout 权重。 Teacher 仍是 TP2：使用 Transformers 原生加载分片，再将前向切换为本地张量和 row-parallel 显式同步 all-reduce，避免本机 Torch 2.7 / Transformers 4.52 的 DTensor 前向同步故障。已用两 rank CPU 的 GQA/变长/padding logits 与 dense 模型对照验证。

每轮记录并画出 cap、mean/P90 response length、token 八元组重复率、rho 集中度/max/mean/max-to-mean、loss/generated token 比例，以及各损失分量和 reference KL。

早停规则预先固定：至少 5 轮，以前 2 轮为基线、最近 2 轮均值为当前值。若 cap ≥ max(75%, baseline+20pp)、重复率>50%的轨迹比例 ≥ max(25%, baseline+20pp)，且 mean length ≥ baseline×1.1，则停止后续更新；或 cap≥85% 且高重复轨迹≥50% 时停止。不同轮题目不同，使用联合持续条件减少单个难 batch 的误触发。停止后依然评估已提交 Final 和可用中间 checkpoint，明确报告早停轮数。

保留 round0、round4（早停备用）、round8、round16、每20轮节点，以及最新版本。正常结束评估 Base/8/16/Final，每模型 825 题，使用 8 卡 vLLM；每数据集最多 199 题，response cap 8192，沿用固定题号/seed/评分协议。

```bash
/pfss/mlde/workspaces/mlde_wsp_Eco_Inference/envs/trl/bin/python scripts/run_decisive.py \
  --output-dir ../LuLu_outputs/experiments/ren_stable_full_qwen1p7_teacher32b_pool2048_s42 \
  --method ren_stable --lora-rank 0 --rollout-backend vllm \
  --rollout-batch-size 16 --eval-rounds 8,16 --learning-rate 1e-6 \
  --reference-kl-coef .1 --idle-checks 1 --poll-seconds 15 --run --detach
```

从启动器启动会冻结代码和数据哈希，先等 8 卡空闲，再训练、评估并自动生成 `RESULTS.md`。`live_progress.json`、`train/analysis/stability.csv` 和 `stability.png` 提供进度和诊断；`train/early_stop.json` 只在触发早停后生成。
