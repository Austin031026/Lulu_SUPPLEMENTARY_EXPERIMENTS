# LuLu / ReN On-Policy Distillation

当前 action-level 实验：见 [shared positive correction：目标、8卡分工与自动评估](md/current_shared_correction.md)。入口为 `scripts/run_shared_correction.py`，方法 `ren_shared`，从 Base 全参数训练，reasoning 使用共享概率纠正目标；保留既有 control/reference。下方为早期实验与通用接口记录。

稳定化全参数 ReN：见 [结构化 mask、bounded weighting 与固定 Reference-KL](md/ren_stable.md)。

最新全参数 ReN 实验：见 [hindsight-resolved 目标、8 卡常驻 vLLM 与自动评估](md/ren_resolved.md)。该配置使用 **Student 全参数训练（LoRA rank=0）**；旧 `ren_opd` 实验配置保留不变。

实现最新方案：**Student 自己 rollout，单个 answer-blind external Teacher 评分，同一轮 frozen Student 的 hindsight view 只决定 recognition frontier**。训练沿用本仓库的 Transformers、PEFT、PyTorch/DDP 栈，默认使用模型常驻、轮内 microbatch 流水线和 Teacher tensor parallel；评测沿用现有 general benchmark parser、数据 manifest、批量普通模型生成和评分口径。

## 当前 decisive experiment：2048 道固定训练题

当前配置为 **Student `Qwen/Qwen3-1.7B`、Teacher `Qwen/Qwen3-32B`，均使用 thinking context**。Teacher 只评分 Student 已采样的 prefix，不生成训练轨迹。目标仍为下文的 `ren_opd`：`alpha KL(qT || p_theta) + (1-alpha) KL(pC || p_theta)`，其中 `alpha=qT(TopK(pH))`。

已从 DAPO-Math-17k 的 prepared train 中固定抽取 **2048 道题，seed42**，存于 [train.jsonl](../LuLu_outputs/data/dapo_pool2048_s42/train.jsonl)。抽取前排除了与完整评测集规范化精确重合的 5 道题（MATH500 4 道、OlympiadBench 1 道）；原来的 256 道 dev 不进入训练池。[Pool manifest](../LuLu_outputs/data/dapo_pool2048_s42/manifest.json) 记录题目 ID、来源与 SHA256。此检查只覆盖规范化精确重复。

训练固定使用 **32 次更新 × 每次 64 道题 = 2048 次 prompt exposure**，完整经过训练池一次。每轮从当前 Student 重新采样，完成该轮全部梯度累积后更新一次，随后同步 hindsight 副本再开始下一轮。固定的是题目池，rollout 随 policy 更新。

| 阶段 | 8 卡分工与 batch |
| --- | --- |
| 训练 | GPU0–4：5 个 Student，rollout batch8/卡；GPU5：hindsight batch1；GPU6–7：32B Teacher TP2，score batch1 |
| 更新 | 5 卡 Student DDP，micro-batch1，全局累积64道题；LoRA rank16 |
| 评测 | 释放训练模型后，8卡运行vLLM，默认每卡最多64条并发；所有checkpoint共享评测配置 |

本实验将 rollout batch 从通用默认4提高至8；保留 response 上限8192、prompt 上限4096、总序列上限16384。模型常驻，`ren_opd` 的 hindsight 和 answer-blind Teacher 对同一批 rollout **并行评分**，两路结果齐备后进入更新。checkpoint 每20次更新保留一个历史节点，每次更新发布 `latest`；本实验保留初始化、step20 和最终 step32。

从工作区 `Rona_Soraka` 根目录运行，使用已配置的 `PYTHON_BIN`：

```bash
# 固定代码快照、数据哈希及训练／评测命令；不启动 GPU 作业。
"$PYTHON_BIN" Lulu/scripts/run_decisive.py

# 独立会话运行：等8卡空闲后训练并自动评测；训练已完成则直接接续评测。
"$PYTHON_BIN" -u Lulu/scripts/run_decisive.py --run --detach
```

`--detach` 将调度器放到独立进程会话，stdin断开、日志写入 `logs/controller.detached.log`，PID记录于 `detached_controller.json`。已有完整32/32 checkpoint时，恢复会跳过训练和Teacher加载，只接续评测。

Teacher 使用本地 `LuLu_outputs/models/Qwen3-32B`，下载固定版本的权重。队列等待现有 GPU 作业自然结束；也支持用 `--wait-for-progress /path/live_progress.json` 等待指定作业报告完成。`--run` 的等待过程只查询资源状态，训练和评测尚未完成时不会报告成功。此配置没有额外启动大型 GPU 计时或验证任务。

初步评测默认从完整 split 中取每个数据集固定的前 **199** 题，不足199题时保留全部：**MATH500 199、AIME25 30、OlympiadBench 199、MMLU-Pro 199、GPQA Diamond 198，共825题/模型**，base＋final合计1650次生成。上限在8卡分片之前应用，所有checkpoint共享完全相同的题目和顺序。原始 HF 评测的 Base 与 final 使用相同 thinking、greedy decoding 和8192输出预算，并报告配对 accuracy delta、rescues、degradations、输出长度与截断比例。`run_decisive.py --eval-max-examples 0` 可明确恢复完整2640题/模型；通用评测入口对应 `--max-examples 0`（shell入口 `MAX_EXAMPLES=0`）。`--eval-split probe` 仍支持使用已有快速split，但每个数据集也受该上限约束。此前全部2640题已通过CPU thinking prompt长度检查。

当前评测使用 `evaluation_settings.json` 中的 `{"max_examples_per_benchmark": 199}`。原始 `experiment_plan.json` 和 `code/` 保留启动时记录；后续评测以 `effective_evaluation_plan.json` 为准。训练进程和冻结训练代码不需重启。

2026-09-15状态核对：训练在07:11 UTC完成32/32更新；旧调度器心跳停在03:13 UTC，评测未自动启动，日志未记录其具体退出原因。训练checkpoint完整保留；现通过独立会话恢复825题/模型评测。诊断记录见 [recovery_20260915/diagnosis.json](../LuLu_outputs/experiments/ren_qwen1p7_teacher32b_pool2048_s42/recovery_20260915/diagnosis.json)。

默认实验目录是 [LuLu_outputs/experiments/ren_qwen1p7_teacher32b_pool2048_s42](../LuLu_outputs/experiments/ren_qwen1p7_teacher32b_pool2048_s42)：

- `experiment_plan.json` 与 `code/`：冻结的配置及代码快照。
- `live_progress.json`：等待资源、训练、评测、完成或失败的状态。
- `logs/training.log`、`logs/evaluation.log`：阶段日志。
- `train/checkpoints/latest`：最新完整 Student／optimizer checkpoint。
- `evaluation_settings.json`：当前评测上限覆盖配置；调度器在训练结束后读取。
- `effective_evaluation_plan.json`：实际生效的每项题数和评测命令。
- `evaluation/summary.json`：训练后 base／final 的初步评测结果。

当前实验已完成1650条生成。汇总时修正了OlympiadBench将重复boxed答案拼接成列表的旧解析规则，仅在CPU上重新评分，未重新生成。当前结果应读取 [evaluation_rescored_v2/summary.json](../LuLu_outputs/experiments/ren_qwen1p7_teacher32b_pool2048_s42/evaluation_rescored_v2/summary.json)；原始输出和原评分保留在 `evaluation/`，变更逐题记录于 `rescore_audit.json`。修正规则与其他数学benchmark一致：采用最后一个boxed答案，保留盒子内的元组和分数，不查看gold来选择答案。

## 评测吞吐与中间节点复评

原实验保留冻结的HF batch16评测；中间节点复评位于 [checkpoint_analysis_20260915](../LuLu_outputs/experiments/ren_qwen1p7_teacher32b_pool2048_s42/checkpoint_analysis_20260915/)。后续新GPU评测默认使用vLLM continuous batching：每卡把全部benchmark样本合并到一个队列，默认最多64条并发序列，完成一题即补入下一题；启用chunked prefill和CUDA graphs。PEFT adapter先在CPU合并一次，八卡读取同一份普通模型，HF兼容入口也会合并adapter。新评测仍使用199题上限、thinking和8192输出预算，解码改为 Qwen 推荐的 T=0.6、top-p=.95、top-k=20；可显式传入 `--decoding greedy` 复现旧配置。vLLM为每题固定采样种子，三个checkpoint使用同一口径。

2026-09-15补评已完成：Base / Step20 / Step32宏平均为46.03% / 45.99% / 44.69%，后期下降主要来自MMLU-Pro；详见 [完整核查报告](../LuLu_outputs/experiments/ren_qwen1p7_teacher32b_pool2048_s42/checkpoint_analysis_20260915/REPORT.md)。8卡均已释放。

新入口为 `--backend auto|vllm|hf`（auto在GPU选择vLLM、CPU选择HF），可调整 `--vllm-max-num-seqs 64`、`--vllm-max-num-batched-tokens 4096`、`--vllm-gpu-memory-utilization 0.8`。`--batch-size` 仅控制HF后端。模型切换前退出上一批worker，释放其engine与KV cache。新增生成/评分耗时、tokens/s记录，以及调度进度中的已完成评测题数。详见 [评测说明](md/lulu_evaluation.md)。

## 目录与依赖

```text
Rona_Soraka/
├── Lulu/                  # 独立项目：lulu/、scripts/、runs/、tests/、md/
├── LuLu_outputs/          # 当前 decisive experiment 的数据、模型与实验结果
├── Soraka/                # 现有项目
└── Soraka_rlrl/           # 可复用的历史数据与 benchmark manifest
```

评测 runner、benchmark parser 和数学 parser 均已包含在 `Lulu/lulu/` 中，不需要导入 Soraka 代码或设置 `LULU_SORAKA_ROOT`。历史 benchmark Parquet 仍可直接复用；通用入口默认输出到 `Lulu/outputs`，上面的 decisive workflow 显式使用同级 `LuLu_outputs`。

入口：

- `scripts/run_decisive.py`：固定2048题实验的准备、排队、训练与评测。
- `scripts/prepare_lulu_data.py`：DAPO／自定义数据准备。
- `scripts/train_lulu.py` 或 `runs/train_lulu.sh`：多轮 on-policy 蒸馏。
- `scripts/evaluate_lulu.py` 或 `runs/eval_lulu.sh`：base／多个 checkpoint 的通用评测。
- `lulu/objective.py`：可独立复用的 target 和 forward KL。

## 算法与边界

设本轮冻结快照为 $S_r$，所有 response IDs 都由 $S_r$ 自己采样，因此训练 state 严格来自当前 Student policy。对每个 reasoning token 的同一个 sampled prefix：

```
pC = S_r(causal prompt + sampled response prefix)
pH = S_r(hindsight prompt + exact same sampled response prefix)
qT = answer-blind external Teacher(causal prompt + sampled response prefix)
H  = TopK(pH)
alpha = sum_{a in H} qT[a]
loss_t = alpha * KL(stopgrad(qT) || p_theta)
       + (1-alpha) * KL(stopgrad(pC) || p_theta)
```

这是当前 `ren_opd` 的核心：**preserve with Student, gate with hindsight, learn from Teacher**。`pH` 从不作为 target，也不再用 $H\setminus C$ 去拼一个人工 target distribution；它只通过 `alpha` 控制当前位置应当信任 external Teacher 多少。Teacher 与 frozen causal Student 的 mismatch 本身提供 novelty，Teacher 在 hindsight Top-K 上的 probability mass 提供 recognizability。当 `alpha=1` 时退化成标准 Teacher OPD；当 `alpha=0` 时只保留 round-start Student policy。

Gold outcome 只加在 hindsight user context 中，不使用 gold solution，也不进入 Student rollout、Teacher context 或 trainable Student context。缓存保留 sampled token IDs，不将轨迹重新 tokenize；预测 response token t 的 hidden index 是 `prompt_length + t - 1`。

ReN 只启用明确的 `<think>...</think>` reasoning span，支持 token 上限截断的未闭合 thinking；排除 thinking 标签、special/EOS、prompt，以及首次 boxed/final-answer 标记起的 remainder。无明确 thinking span 的 response 不产生 loss，整轮无 reasoning 会报错。不同模型／语言的 reasoning 协议需要相应扩展 mask。

每条有 reasoning 的轨迹先对 reasoning positions 取平均，再对全局有效轨迹取平均。所有 target 均来自本轮开始时的 frozen Student / Teacher scoring 并 stop-gradient；每轮只做一次 DDP update 后立即 refresh rollout。旧的 $H\setminus C$ positive-correction graft 仍保留为 `ren_graft` ablation，便于比较此前的局部正向修正目标，但不再是默认 ReN。

当前严格单步配置还有一个数学性质：每轮所有 loss 都在更新前的同一参数处计算，此时 `p_theta=pC`，因此 preservation KL 的梯度在精确算术下为零；整轮梯度等价于由 `alpha` 加权的 Teacher KL。代码仍计算完整双 KL，32B Teacher 提供非零学习信号。若对同一个冻结快照连续做多次 optimizer update，preservation 项才会在 Student 偏离快照后产生约束；当前实验不采用这种多步设置。

默认每轮一整个 rollout batch、一次全局 optimizer update，然后刷新快照和 rollout。常驻后端只接受 `--update-passes 1`：本轮全部轨迹和 target 准备完成后才进入 DDP update，下一轮必须等待更新与 hindsight 权重同步完成。优化器状态跨 round 常驻；断点续训从完整 checkpoint 恢复，未完成 round 重新采样、评分。旧的 `--backend staged` 仍支持多次 update-passes，但后续更新使用的是本轮旧快照采样的轨迹。

## 并行与显存

默认 `--backend persistent` 将模型和 optimizer 常驻于固定 GPU 组。当前机器是 8 × A100-SXM4-80GB；默认 ReN 配置分工如下：

| GPU | 常驻角色 | 并行方式 |
| --- | --- | --- |
| 0–4 | Qwen3-1.7B Student | 每卡一份完整 Student；rollout 数据并行，update 使用 5 卡 DDP |
| 5 | privileged / hindsight Student | 同一个 Student 的独立推理副本，每轮开始同步最新可训练参数 |
| 6–7 | 单个 Qwen3-32B Teacher | Transformers 原生 `tp_plan="auto"`，同一实例做 2 卡 tensor parallel |

这不是 5 卡分片加载一个 1.7B Student；5 份 Student 同时处理不同题目。Privileged 模型也不是额外的固定 Teacher，而是当前 Student 的 gold-conditioned view。LoRA 训练只同步 adapter 参数；全参数训练需要同步全部可训练权重，内存、通信和保存开销都会增加。Teacher 始终只看到 causal prompt 和 Student sampled prefix，不接收 gold 或 hindsight prompt。

同一轮内，Student 按 batch 流式提交已采样轨迹。默认 `ren_opd` 中，hindsight 与 answer-blind Teacher 独立接收同一批轨迹并行评分，controller 等待两路结果齐备后把监督数据交回 Student；其他 Student batch 同时继续采样。全部 batch 完成后进行一次 DDP update。需要稀疏 correction IDs 的旧 `ren_graft` 等对照仍按其依赖顺序评分。**流水线只在同一份冻结快照的 round 内重叠，不提前用旧权重采样下一轮。**

Frozen causal hidden states 留在各 Student 进程内存中。默认 `ren_opd` 中，hindsight 服务只把 Top-K token IDs 返回给 Student/controller；这些 gold-dependent IDs **不会发送给 Teacher**。Teacher 服务只接收 causal prompt + Student sampled prefix，并返回 answer-blind Teacher selected hidden states。Teacher output head 在启动时只传一次到 Student worker；update 端按 `--logit-chunk-size` 重建 full-vocabulary $q_T$，再用本地 hindsight Top-K 计算 $\alpha=q_T(\mathrm{TopK}(p_H))$。因此 Teacher 始终严格 answer-blind，也不需要传输或保存 `[所有 tokens, vocabulary]` logits。`ren_graft` / Top-K projection controls 仍走原来的稀疏 Teacher probability 路径。

`--student-gpus`、`--hindsight-gpus`、`--teacher-gpus` 可显式分配互不重叠的组；不指定时从 `--gpus` 中自动分配，默认保留 2 卡给 Teacher、1 卡给 hindsight，其余给 Student。`--teacher-gpus-per-worker` 控制自动分配的 Teacher TP 大小，显式 `--teacher-gpus` 则直接决定 TP 大小；常驻后端只有一个 Teacher 实例。`opsd` 不启动 Teacher，`vanilla_opd` / `causal_topk` 不启动 hindsight，未使用角色应保持 GPU 参数为 `auto`。

5+1+2 是起始配置，实际吞吐取决于 rollout 长度和两个评分服务的速度。针对默认 8192-token response 上限，每个 Student 默认 `--rollout-batch-size 4`，Teacher/hindsight 评分默认 `--score-batch-size 1`，反向默认 `--train-micro-batch-size 1`。评分 batch1 限制长上下文 padding 带来的激活显存开销；全局 batch 仍是 64 道题，由各 Student 分批处理并累积成一次更新。生成目前仍是静态 batch 的 HF `generate`，没有 continuous batching。Teacher TP 要求所选模型提供兼容的原生 TP plan；不能把任意模型的层分片当成 TP。

当前 Torch 2.7 / Transformers 4.52.4 下，Teacher TP 的 rowwise 输出归约显式同步完成；矩阵仍按原生 TP plan 分片。此前较大 GPU batch 出现过 collective 超时，已应用该规避方案，并通过 CPU 双进程 8192-token 投影对照检查；该修复尚未重做大型 GPU 验证。

默认 LoRA rank16、dropout0；`--lora-rank 0` 支持全参数更新。Student/Teacher 必须有相同 tokenizer token→ID 映射和 output vocabulary size，不相容时明确报错。

需要复现旧实验、查看磁盘缓存或复用同一轮轨迹多次更新时，使用 `--backend staged`。该后端仍是 collect → Teacher → update 子进程阶段，每轮重新加载模型，Teacher 使用 HF 层分片；`--teacher-workers`、`--teacher-memory-gib` 和 `--keep-round-cache` 用于此模式。常驻后端不接受 `--keep-round-cache`。

## 在当前机器运行

从 `Rona_Soraka/Lulu` 目录执行。当前可用环境：

```bash
export PYTHON_BIN=/pfss/mlde/workspaces/mlde_wsp_Eco_Inference/envs/trl/bin/python
export HF_HOME=/pfss/mlde/workspaces/mlde_wsp_Eco_Inference/huggingface_cache
export HF_HUB_CACHE="$HF_HOME/hub"
export HUGGINGFACE_HUB_CACHE="$HF_HUB_CACHE"
unset TRANSFORMERS_CACHE
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
```

该环境实际安装 PyTorch2.7.0、Transformers4.52.4、PEFT0.20.0；不要求另装名为 `trl` 的 Python 包。Qwen3 thinking 模式参见 [Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3-1.7B)。

先准备数据：

```bash
"$PYTHON_BIN" scripts/prepare_lulu_data.py \
  --dataset BytedTsinghua-SIA/DAPO-Math-17k \
  --output-dir ../LuLu_outputs/data/lulu_dapo \
  --dev-size 256 --seed 42
```

上面的命令在新输出目录准备数据。当前可直接复用的数据位于 `../Soraka_rlrl/data/lulu_dapo`：**16,920 train / 256 dev**，去重后排除了 12 个 gold 矛盾题目组。默认离线读取已有 HF dataset cache。缓存中的 DAPO train split 含重复样本；程序按归一化题目去重，默认排除存在相互矛盾 gold labels 的整个题目组（manifest 会报告数量），然后按 seed 固定划分，输出 train/dev JSONL、与通用 evaluator 兼容的 `dev_eval.parquet` 和带 SHA256 的 manifest。Gold 字段来自 `reward_model.ground_truth`；参见 [DAPO 数据集](https://huggingface.co/datasets/BytedTsinghua-SIA/DAPO-Math-17k)。`--train-limit` 在去重、划分之后执行，不改变 dev 集。使用新数据源时可加 `--allow-download`。

下面的训练示例直接复用已有 DAPO 数据；若使用上面新生成的数据，将 `--train-data` 改为 `../LuLu_outputs/data/lulu_dapo/train.jsonl`。正式配置先 dry-run：

```bash
"$PYTHON_BIN" scripts/train_lulu.py \
  --train-data ../Soraka_rlrl/data/lulu_dapo/train.jsonl \
  --output-dir ../LuLu_outputs/experiments/lulu_ren_qwen3_1p7b \
  --model Qwen/Qwen3-1.7B --teacher-model Qwen/Qwen3-32B \
  --backend persistent --gpus 0,1,2,3,4,5,6,7 \
  --student-gpus 0,1,2,3,4 --hindsight-gpus 5 --teacher-gpus 6,7 \
  --global-batch-prompts 64 --rollout-batch-size 4 --score-batch-size 1 \
  --train-micro-batch-size 1 --logit-chunk-size 32 \
  --rounds 100 --max-new-tokens 8192 \
  --max-prompt-tokens 4096 --max-sequence-tokens 16384 \
  --top-k 32 --save-every 20 --dry-run
```

去掉 `--dry-run` 开始训练；同命令加 `--resume` 恢复。恢复要求模型、训练配置和 train.jsonl 的 SHA256 一致；输入/输出路径的写法以及保存间隔可调整，原始 run_config.json 不会被重写。常驻后端从原子发布的 `checkpoints/latest` 恢复完整 Student 和 optimizer；历史 staged 实验应显式使用 `--backend staged` 续训，不能直接切换后端复用其运行目录。

`--save-every 20` 默认保留初始化、每 20 次 optimizer update 的节点、最终节点和最新版本。**latest 仍在每次更新后写入**；减少的是历史文件数量，不代表磁盘写入频率降低 20 倍。保存成功后才切换 latest 并清理旧的非保留节点，失败不会损坏此前 latest。默认 LoRA 的保存量较小，全参数更新的 checkpoint I/O 会明显增加。

直接使用 `train_lulu.py` 时，Teacher 权重须提前准备好；当前 decisive workflow 等待本地 `LuLu_outputs/models/Qwen3-32B` 的全部权重完整后启动。更换 Teacher 可显式传入其他模型路径，模型不会自动替换。8192-token上限／32B Teacher的生产运行已完成32次更新；Base、Step20、Step32的2475次补评也已完成。

默认采样 `temperature=1, top_p=1, top_k=0`，对应未经截断的 Student policy。可以改温度／top-p，但实验方法应记录为相应 sampling policy。`--max-new-tokens 8192` 是 response 上限，包含 thinking 和最终回答；完整序列还必须容纳 causal 或 hindsight prompt。默认分别检查两种 prompt 均不超过 4096 tokens，且 prompt + response 不超过 16384；4096 + 8192 = 12288，处于该预算内。超过预算会明确报错，不截断 prompt/hindsight context。当前8192-token上限生产实验已通过多轮采样、评分、更新和checkpoint保存。

更换训练数据：

```bash
"$PYTHON_BIN" scripts/prepare_lulu_data.py \
  --dataset /path/problems.parquet \
  --question-column problem --answer-column answer \
  --output-dir ../LuLu_outputs/data/lulu_custom --dev-size 128
```

更换 Student 使用 `--model`；更换 Teacher 使用 `--teacher-model`。模型必须支持普通 causal LM 的 `get_decoder()` 与 linear output embeddings、相容的 chat template 和 reasoning mask。

## 对照方法

相同代码、rollout refresh、随机种子、batch 与 reasoning mask：

| `--method` | Supervision |
| --- | --- |
| `ren_opd`（默认） | `alpha KL(qT||pθ) + (1-alpha) KL(pC||pθ)`，其中 `alpha=qT(TopK(pH))` |
| `ren_graft` | 旧版：在 `TopK(pH)\TopK(pC)` 上 graft positive Teacher corrections |
| `vanilla_opd` | 完整 qT |
| `opsd` | 完整 frozen pH，直接作为 target |
| `causal_topk` | qT 投影到 causal Top-K，再归一化 |
| `union_topk` | qT 投影到 causal/hindsight Top-K 的 union，再归一化 |

`ren_graft` 是此前局部正向修正目标的复现/消融；默认 `ren_opd` 不再修改 Teacher distribution 的 support，而是由 hindsight recognition 决定 Teacher-vs-Student-preservation 的连续权重。这些对照共享 rollout refresh、reasoning mask 与 optimizer 配置，但各自随后从自己的 Student rollout。Base 在 evaluation 中直接评测初始化模型；Teacher trajectory SFT/SeqKD 不属于此 on-policy 训练入口。

## Evaluation

详见 [LuLu evaluation 说明](md/lulu_evaluation.md)。默认套件同时含数学与 general reasoning：MATH500、AIME25、OlympiadBench、MMLU-Pro、GPQA Diamond。支持可选 LiveCodeBench 官方评分、任意已有 benchmark Parquet、多 checkpoint 与 base 配对比较。

```bash
export DATA_MANIFEST=../Soraka_rlrl/experiments/v6_5-success-q-scale-pool4096-phase11024-seed42/crossbench_v631/data/manifest.json
export CHECKPOINT=../LuLu_outputs/experiments/lulu_ren_qwen3_1p7b/checkpoints/latest
export OUTPUT_DIR=../LuLu_outputs/experiments/lulu_ren_qwen3_1p7b/evaluation
PYTHON="$PYTHON_BIN" GPUS=0,1,2,3,4,5,6,7 BATCH_SIZE=8 bash runs/eval_lulu.sh
```

评测只运行部署时的 Student，无 Teacher/hindsight/gold prompt。Checkpoint 是普通 HF／PEFT 文件，可以被现有工具继续加载。数学 parser 使用项目声明的 `latex2sympy2-extended`；超过 `--max-prompt-tokens` 的 benchmark prompt 会明确报错，完整保留输入。

DAPO held-out dev 也可直接评测：

```bash
"$PYTHON_BIN" scripts/evaluate_lulu.py \
  --checkpoint ren=../LuLu_outputs/experiments/lulu_ren_qwen3_1p7b/checkpoints/latest \
  --benchmark dapo_dev=../Soraka_rlrl/data/lulu_dapo/dev_eval.parquet \
  --output-dir ../LuLu_outputs/experiments/lulu_ren_qwen3_1p7b/dapo_dev_eval \
  --gpus 0,1,2,3,4,5,6,7 --batch-size 8
```


## 输出与验证

- `run_config.json`：模型、训练配置与训练数据 SHA256。
- `runtime_plan.json`：常驻 GPU 分工、并行方式与一次性 `startup_seconds`。
- `checkpoints/step_000000`：初始化；`step_000020` 等是保留的中间节点，包含 Student、tokenizer、optimizer 和恢复状态。
- `checkpoints/latest`：指向最近完成更新的原子 symlink；可直接传给 evaluator。最终更新也保留其 `step_XXXXXX` 目录。
- `latest.json`：最新 checkpoint 信息；恢复以 symlink 指向的 checkpoint 自身元数据为准。
- `metrics/round_XXXX.json`：KL、gradient norm、reasoning token counts、`round_seconds`、各阶段耗时与预计剩余时间。
- evaluation `summary.json`：各 benchmark 准确率、response length、truncation 与相对 base 的配对变化。

旧 staged 后端沿用 `checkpoints/round_XXXX` 和可选 `round_cache/round_XXXX`，每轮保留 checkpoint，不使用上述常驻后端的保存策略。

```bash
"$PYTHON_BIN" -m pytest tests -q
```

测试包含公式/梯度、空 mask、teacher positivity、稀疏缓存等价性、数据去重与信息隔离、真实 tiny HF/PEFT 加载和通用评测子进程。真实 Qwen smoke 的具体结果记录在 [验证记录](md/lulu_validation.md)；短 smoke 不能证明最终 benchmark 提升。

## On-policy、checkpoint、评测与耗时

当前默认训练是严格 batch on-policy：所有轨迹来自本轮 Student，所有 microbatch 梯度累积完成后才更新一次参数，下一轮同步 hindsight 后重新采样。DAPO JSONL 只提供题目和 gold outcome；没有正确性筛选或 Teacher trajectory generation。

100 次更新、`--save-every 20` 的最终保留目录是 `step_000000`、`step_000020`、`step_000040`、`step_000060`、`step_000080`、`step_000100`；训练中额外保留最新节点。每个更新节点包含 optimizer state，没有 microbatch 中间 checkpoint。训练入口没有自动周期评测 hook；可以将多个已保留 `--checkpoint NAME=PATH` 交给 evaluator。训练占用全部 8 卡时，应在训练结束后或另有 GPU 时运行评测。

评测默认5个数据集、最多199题/数据集，当前套件共825题/模型。后续新GPU运行使用vLLM，每卡最多64条并发序列；当前冻结实验仍是HF batch16/卡。HF静态batch会为已结束的短回答继续占用槽位，本次已完成输出测得约三分之一的token计算槽位用于padding。此比例不是对vLLM加速倍数的测量。

常驻训练的 `round_seconds` 记录完整一轮墙钟时间，包含快照同步、rollout/评分流水线、DDP update 和 checkpoint 保存；首次模型加载单独记录在 `runtime_plan.json` 的 `startup_seconds`。`teacher_seconds` 与 `hindsight_seconds` 是可能互相重叠的服务时间，不应直接相加推算整轮耗时。`remaining_seconds_at_last_round_rate` 使用最近一轮速度估算剩余训练时间，短回答和长回答混合时会波动。

已完成 Qwen3-1.7B Student、显式选择的 Qwen3-8B Teacher、5+1+2 八卡常驻配置的三轮短 smoke，验证多轮流水线、同步与 checkpoint 保存。它使用每轮 10 题、64-token response cap，不能证明 32B Teacher 或完整 8192-token 配置的吞吐与显存表现；具体记录保留在 [验证文档](md/lulu_validation.md)。

默认 100 轮 × 64 题 = 6400 次 prompt exposures，约为 16,920 训练题的 37.8%；至少 265 轮覆盖一遍。每题都生成到 8192 tokens 时，100 轮最多约 5243 万 generated tokens。8K 回答比 4K 增加生成长度、KV cache 和评分/反向的上下文，耗时及显存不能直接按两倍线性外推。当前32B Teacher已缓存，8192上限的生产实验已完成32次更新，评估正在接续；此处不提供精确 ETA，也不为估时额外启动训练或 benchmark。

## 独立项目迁移验证

LuLu 自带 `pyproject.toml`，包名 `lulu-ren-opd`，不再由 Soraka 的 packaging 配置收集。训练 worker 使用 `python -m lulu.training` 启动，显式传递本项目导入路径；数据与训练本身不依赖 Soraka。仓库内的 Python／shell 入口可以直接使用，若安装包，也提供 `lulu-train` 命令。

迁移当时共 **78 项测试及 6 个 subtests 通过**，覆盖原有算法、真实 tiny-model DDP、从任意 cwd 启动训练/评测子进程、独立数据输出和路径变化后的严格续训检查。独立 wheel 构建及解包后 module-entry dry-run 通过。旧 Qwen smoke checkpoint 从新 LuLu 目录恢复检查通过；当时的 `/tmp` 输入已消失，已从同一 DAPO 数据恢复出 SHA256 完全一致的两题输入，原 run manifest 未修改。详情见 [迁移验证记录](md/lulu_validation.md)。

[Prompt-balanced absolute ReN：目标、dev 选择和8卡运行](md/ren_balanced.md)
