# Hindsight-resolved Teacher mismatch：全参数 on-policy distillation

本实验 Student 为 Qwen3-1.7B，**全部参数可训练，LoRA rank=0**。
Teacher 为冻结的 Qwen3-32B；Hindsight 使用本轮开始时 Student 的同一份参数，额外看到正确最终答案，沿用 Student rollout 前缀，不另生成推理。
之前 `ren_qwen1p7_teacher32b_pool2048_s42` 使用 LoRA rank=16；其结果不是全参训练结果。

## 目标

在每个 reasoning 位置，以完整词表计算：

\[
D_t^C=KL(q_T\Vert p_C),\quad D_t^H=KL(q_T\Vert p_H),\quad
R_t=D_t^C-D_t^H=\sum_vq_T(v)(\log p_H(v)-\log p_C(v)),\quad w_t=[R_t]_+.
\]

为减小数值误差，代码直接计算上面的对数概率差，同时记录两个 KL 作交叉校验。
设本轮 **所有 Student rank 合计** 有 \(N\) 个 reasoning 位置：

\[
L=\frac{\sum_t\operatorname{sg}(w_t)KL(\operatorname{sg}(q_T)\Vert p_\theta)}
{\operatorname{sg}(\sum_t w_t)+10^{-8}N}.
\]

这等价于先将权重除以全局均值加 epsilon，再对所有 token 求均值。
不是逐轨迹或逐卡归一化；qT、pC、pH、权重都没有梯度。Teacher 是唯一训练 target，没有混合 Student 分布，没有 Top-K 支持域截断。
Top-K=32 仅用于记录旧 alpha 的集中度；采样 Top-K=20 是另一项独立配置。
全局权重全部为零时不调用 optimizer.step，因此不会触发 Adam 动量或 weight decay 更新。
仍然提交本轮状态，并进入下一轮；checkpoint 同时记录 completed_rounds 和实际 completed_updates。

“权重均值约为 1”不保证梯度范数相同；是否自然集中、是否改善泛化，需要从记录和评估验证。
本轮从 LoRA 改为全参，同时训练采样由旧实验的 T=1/TopP=1 改为 .6/.95/20。
因此与旧 LoRA 实验的差异不能单独归因于新目标；严谨方法对照应保持 trainability 和采样一致。

## 8 卡分工与刷新

- GPU 0–4：5 个全参数 Student DDP 副本。FP32 参数、梯度和 Adam 状态，BF16 autocast 计算，gradient checkpointing，梯度累积到同一个全局 batch。
- 同一组 GPU 0–4：每卡一个独立进程的常驻 vLLM 引擎，连续 batching；每轮只更新模型权重、清空 prefix cache，不重建引擎。
- GPU 5：冻结的本轮 Hindsight Student，仅 teacher forcing，不生成。
- GPU 6–7：32B Teacher 真正 TP=2，answer-blind，仅 teacher forcing。

一个 round：加载上轮已提交 snapshot → Student rollout/因果打分 → 并行 Hindsight 与 Teacher 打分 → 汇总全局权重 → 一次 DDP optimizer update → 原子 checkpoint → 下一轮。
Teacher 请求只允许 causal prompt、rollout、位置；正确答案不会传入 Teacher。
同一轮的评分任务重叠执行；不同轮之间保留更新屏障，避免使用旧参数 rollout。
HF/BF16 与 vLLM 的算子和数值舍入可以存在微小差别；两者的参数版本必须相同。

全参同步从同一个 safetensors checkpoint 流式加载，使用共享文件页缓存；避免在父子进程间 pickle 数 GB 参数。
vLLM 使用原生权重 loader，核对全部参数已加载。Hindsight 也在原模型对象内加载完整参数。
vLLM memory utilization=0.42，max_num_seqs=16。当前 vLLM 内存规划会计入同卡已常驻的 Student，实际引擎新增内存小于 42%；为后续 Adam 状态和 8192-token backward 留空间。
Teacher 的原生 BF16 TP 路径不套 Student autocast；已用 32B/TP2 实测 8192-token 前缀打分。
全词表投影按 128 个位置分块，隐藏状态留在 CPU RAM，避免构造整个 batch 的 vocabulary logits。
当前环境已安装 vLLM 0.9.0，使用项目最新的 vLLM continuous-batching 实现；没有将该版本冒称为上游最新发行版，也没有升级共享 CUDA/PyTorch 环境。

## 配置与启动

固定 DAPO-Math-17k 的 2048 题池，seed=42，每轮 64 题，每题 1 次 rollout，共 32 轮，恰好覆盖一次。
所有 rollout 都保留；math verifier 只写 metadata，不筛选训练数据。
thinking 模式，temperature=.6、top_p=.95、top_k=20；最大 response=8192，prompt≤4096，总 context≤16384。
learning rate=1e-5，weight decay=0，max grad norm=1。

```bash
/pfss/mlde/workspaces/mlde_wsp_Eco_Inference/envs/trl/bin/python scripts/run_decisive.py \
  --output-dir ../LuLu_outputs/experiments/ren_resolved_full_qwen1p7_teacher32b_pool2048_s42 \
  --method ren_resolved --lora-rank 0 --rollout-backend vllm \
  --rollout-batch-size 16 --eval-rounds 10,20 \
  --idle-checks 1 --poll-seconds 15 --run --detach
```

启动器冻结代码及数据哈希，等待 8 卡空闲，完成训练后自动评估 base、round10、round20、final。
每个 checkpoint 使用完整 Student 权重，无需合并 adapter。
评估全部 8 卡跑 vLLM，每卡跨数据集连续补充 batch；thinking .6/.95/20，固定每题 seed，8192 response cap。
每个数据集最多 199 题：Math500 199、AIME25 30、OlympiadBench 199、MMLU-Pro 199、GPQA Diamond 198，共 825/模型。

每轮都保存可恢复的 latest（完整参数 + Adam），历史保留 round0、round10、round20、final round32。
全参 checkpoint 比 LoRA 大很多；保存耗时单独记录。旧的非保留 latest 在新 checkpoint 原子提交后删除。

## 运行记录

实验目录下：

- `live_progress.json`：训练/评估阶段、已完成轮数、实际更新次数、最近指标、日志路径。
- `train/runtime_plan.json`：实际 GPU 分配与 Student 总参数/可训练参数数量。
- `train/metrics/round_XXXX.json`：KL、梯度范数、各阶段时间、权重统计。
- `train/diagnostics/round_XXXX/summary.json`、`position_scores.npz`、`concentration.csv`：更新前 DC、DH、R、w、旧 alpha、top-fraction mass、ESS。
- `train/rollouts/round_XXXX/shard-NNN.jsonl`：全部轨迹 token/text、题号、正确性、截断、snapshot round 和 rollout checkpoint。
- `evaluation/summary.json`：训练后自动写出的完整评估结果。

第一轮开始即可检查“权重是否自然集中”；集中度本身不代表训练有效。中间节点与 final 的相同样本评估用于检查过拟合与泛化退化。
