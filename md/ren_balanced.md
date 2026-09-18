# Prompt-balanced absolute ReN

本轮只训练新方法。Student Qwen3-1.7B 全参数，冻结 Teacher Qwen3-32B；thinking sampling .6/.95/20，response cap 8192。DAPO 固定 2048 题训练池，每轮 64 prompt、当前每题一条 rollout，最多 12 轮（768 次 prompt exposure）。保留所有成功、失败、截断轨迹。学习率 1e-6，control 系数 1、reference KL 系数 0.1 均不变。

同一轮快照 S_r 的 C/H 与 answer-blind Teacher 沿用同一 Student 前缀。定义完整词表上的 D_C=KL(T||C)、D_H=KL(T||H)，g=max(D_C-D_H,0)，w=g/(1+g)。只用局部 outcome-resolved mismatch，不要求 H 全局接近 Teacher，也不把分数解释为 trajectory success predictor。

令 R_ij 是结构 think mask 内的全部 reasoning 位置，包含 w=0 的位置；N 是全 batch 实际生成 token 总数。精确目标为：

\[
L=\frac1B\sum_i\frac1{M_i}\sum_j
  \frac{\sum_{t\in R_{ij}}w_{ijt}KL(q_{T,ijt}\Vert p_{\theta,ijt})}{|R_{ij}|}
+\frac1N\sum_{i,j,t\notin R_{ij}}KL(q_{T,ijt}\Vert p_{\theta,ijt})
+\frac{0.1}{N}\sum_{i,j,t}KL(p_{\theta,ijt}\Vert p_{\rm ref,ijt}).
\]

**reasoning 不除以总生成长度、不除以非零权重 token 数、不除以权重和/均值。** 空 reasoning 区域的该项记为 0，但这条 rollout 仍计入该 prompt 的 M_i，该 prompt 仍计入 B；control/reference 不受影响。按 source_id 跨 DDP ranks 汇总 M_i，支持每个 prompt 不同 rollout 数量。Reference 始终使用 round0，Hindsight 每轮同步。

固定 dev256 从原 DAPO held-out dev 冻结，已按既定完整题目文本标准检查与整个训练池和所有评估集的精确重复（均为0；不代表语义近重复已排除）。在 round0/4/8/12 做 greedy validation，temperature=0、固定题序、分片和8192预算。最高 dev accuracy 选 checkpoint，相同准确率选更早轮次，round0 也可被选中。Greedy 消除采样随机性，不宣称底层 GPU 运算 bitwise 确定。没有按 benchmark 选 checkpoint。

若 dev 选中 round4/8/12，外部 Math500/AIME25/OlympiadBench/MMLU-Pro/GPQA Diamond 只测所选模型一次，最多199题/集（AIME30、GPQA198），共825条。复用旧的修正评分后 Base 回答作参照，检查模型、数据、采样协议及文件哈希；不重跑 Base，不训练其他 baseline。通用评估使用修复后的选择题 parser。若最终 dev 仍选择 round0，则逐张量精确核对它与 Base 的权重、模型结构、tokenizer、chat template 及评估协议，验证通过才复用已有 825 条原始回答，不重复生成。复用有独立 provenance，并明确说明这不是训练收益或新的独立评估。

8卡分工：0–4 常驻 Student DDP 与 vLLM rollout 引擎；5 常驻 Hindsight 和固定 Reference；6–7 常驻 Teacher TP2（显式同步的本地分片）。C/H/T teacher forcing batch=2；Student 更新 micro-batch=1、完整词表按128位置分块。dev 复用5个常驻 vLLM 引擎，max_num_seqs=32；最终评估用8个独立 vLLM 引擎。没有增加 ESR 或截断失败轨迹的过滤。

每轮日志包含：原全局 token-average 的 reasoning loss（audit）、实际 prompt-balanced reasoning loss、平均 reasoning 长度、新平均方式下 capped reasoning loss share、g 均值、bounded weight 的 top1/top10 mass、三个 loss 分量。第1/4/8/12次更新前，在相同完整 global batch 上测三项全参数梯度范数（reference 含0.1系数），分三次反向串行复用缓存，不额外驻留三套梯度、不改变 optimizer state。梯度范数不等于 Adam 更新贡献百分比。

保留round0/4/8/12和latest，原长度/重复率联合早停仍保留；不新增依据测试集成绩的早停。

```bash
/pfss/mlde/workspaces/mlde_wsp_Eco_Inference/envs/trl/bin/python scripts/run_decisive.py \
 --method ren_balanced --lora-rank 0 --rounds 12 --rollout-backend vllm \
 --rollout-batch-size 16 --rollout-vllm-max-seqs 32 --score-batch-size 2 \
 --learning-rate 1e-6 --reference-kl-coef .1 --validation-every 4 --gradient-norm-every 4 \
 --output-dir ../LuLu_outputs/experiments/ren_balanced_abs_full_qwen1p7_teacher32b_pool2048_dev256_r12_s42 \
 --idle-checks 1 --poll-seconds 15 --run --detach
```

控制器冻结全部执行代码和数据/旧Base哈希。`live_progress.json` 提供阶段；`train/validation/history.json` 与 `selected.json` 记录 dev 选择；`train/analysis/stability.csv` 记录训练预算；`RESULTS.md` 和 `comparison.json` 在最终评估后自动生成。
