# Probability-space certified-correction audit

当前只有 offline diagnostic，没有注册新的训练 objective，也没有启动三组训练。

固定同一轮 Student 快照、同一题和同一 Student continuation：

- `pC`: causal Student。
- `pH`: 同参数 Student，只有 hindsight prompt 额外含 gold final answer。
- `qT`: answer-blind Teacher。

```text
uT = qT - pC
uH = pH - pC
lambda = clip(sum(uH * uT) / (sum(uT ** 2) + epsilon), 0, 1)
target = (1 - lambda) * pC + lambda * qT
```

所有分布为 full vocabulary、temperature 1；不使用生成时 temperature/top-p/top-k 来扭曲评分分布。
`epsilon=1e-8` 先固定；同时报告 1e-10、1e-6、1e-4 的敏感性，但不据结果挑选主值。

## 必须区分的保证

在 `p_theta=pC` 且 target detached 时：

```text
g_new = pC - target = lambda * (pC - qT)
||g_new|| <= ||g_Vanilla||
```

这是每位置的 initial logit gradient。它不保证小于旧的 `w*(pC-qT)`，因为 `lambda` 可以大于 `w`；也不保证异质权重下整个 batch 参数梯度或 Adam 步长更小。

固定 snapshot 时，loss 与下面的表达式只差一个与当前参数无关的常数：

```text
lambda * KL(qT || p_theta) + (1-lambda) * KL(pC || p_theta)
```

所以每轮只有一次 optimizer update、在 snapshot 上计算整批梯度时，首次梯度与 `lambda * KL(qT || p_theta)` 相同。微批次梯度累积不等于多次参数更新：当前 `persistent.update_records` 在全部 microbatch backward 完成后才 `optimizer.step()`。

固定端点保证不跨轮成立。反例：固定 state/Teacher/λ，每轮都拟合到本轮 mixture，再刷新 snapshot，则：

```text
p_r = qT + (1-lambda)^r * (p_0-qT)
```

λ=0.1 时，8 次这种理想化刷新已经走过原 correction 的 56.95%，并非总共只走 10%。此反例只用于检验“永远不会靠近 Teacher”的说法，不是实际训练速度预测。真实网络有共享参数、control/reference、Adam 和有限更新，不能从独立 state 最优点直接推出任务性能。

同样，`w=0.01` 不代表真实 Adam 训练恰好慢 100 倍；该直觉只适用于特定 SGD/固定梯度条件。

## 实现与数值验证

- `lulu/certified_audit.py`: detached projection 与逐位置诊断，无训练状态修改。
- `scripts/audit_certified_correction.py`: 复用原 directional audit 的 selected_records.jsonl，6 个 Student scorers + 2 个单卡 Teacher replicas；score batch 1。两份 32B BF16 Teacher 需要每卡 80 GB。
- `scripts/summarize_certified_correction.py`: prompt-balanced / global-token 两种统计、分组、bootstrap、归一化与梯度恒等式验证。
- `tests/test_certified_audit.py`: 独立 autograd 验证、边界、saturation、跨轮反例、对旧加权梯度的反例。

主评分 Student 采用原 checkpoint 的 FP32 master weights + BF16 compute。另在相同 C/H/T hidden states 上运行 FP32 输出头（TF32 关闭），不再跑 backbone。Teacher FP32 head 是原 BF16 head 的升精度计算，不是重新恢复 FP32 checkpoint。

浮点运算在 `pC-target` 中存在消减误差。验证同时记录：

1. FP32 显式 target 的绝对 vector error；
2. 使用相同概率和 λ、独立 FP64 构造 target 的 vector identity；
3. Teacher correction norm >1e-7 的 ratio error；
4. 所有位置的梯度范数上界及 target simplex/凸 KL 上界。

1e-7 只是避免给 0/0 或接近零的 ratio 赋义的报告阈值，不参与 lambda/target 构造，不是 gate。

`hidden/` 保存每个选中位置的 C/H/T states，避免后续分析反复执行 32B backbone。full-vocabulary logits 不持久化，以减少磁盘占用。统计按 reasoning-token mean within rollout，再按 prompt 等权，保留 capped 和错误轨迹，不按活跃权重重新归一化。

2026-09-16 运行先尝试 TP2，处理 4 条后停止推进；该尝试保存在输出目录的 `attempt_tp2/`。实际完整审计采用两个独立 TP1 replicas，从全部相同 48 条轨迹重新评分。旧尝试不混入主结果。

```bash
python scripts/audit_certified_correction.py --source-audit PREVIOUS_AUDIT --output-dir NEW_AUDIT
python scripts/audit_certified_correction.py --output-dir NEW_AUDIT --run
python scripts/summarize_certified_correction.py --output-dir NEW_AUDIT
```

准备阶段冻结选样文件 hash、输入/代码 hash、precision 设置及数值容差。训练/后续评估 horizon 分别保持 8192/32768；dev 应与实际 thinking sampling 匹配。
