# 16k Base restart: trajectory-balanced bounded absolute ReN

从旧实验保留的 round0 Base **仅加载权重与 tokenizer**，新建 optimizer、轮次和固定 reference。模型为 Qwen3-1.7B 全参数 thinking Student，Qwen3-32B Teacher；不使用 LoRA、不接续旧 Round8/Final12。

保持固定 DAPO 2048 pool、seed42、每轮64题、每题1条 rollout、12轮、1 optimizer update/round，temperature .6 / top_p .95 / top_k20。Loss、LR1e-6、control系数1、reference系数.1不变：

```
g = max(KL(T||C) - KL(T||H), 0)
w = g / (1 + g)
L_reason = mean_prompt mean_rollout sum_reasoning(w * KL(T||live)) / reasoning_token_count
```

无 weight-sum/mean renormalization。保留失败、截断、零权重位置。唯一训练任务参数变化是 response horizon 8192→16384；总 context 上限20480，prompt预算4096，不截断输入。数值探针维持旧训练实现，未替换为后续 probability-alignment λ。

## 指标

绝对 response positions `[0,8192)`、`[8192,16384)`，且只统计结构 thinking mask 内的位置。每轮报告 reasoning token 数、E[w]、E[w KL]、实际 prompt-balanced objective contribution。均值 denominator 包含该段全部 reasoning positions，包括 w=0；空段 mean=null。

第1/4/8/12次更新前，精确测量同一完整global batch、所有trainable parameters的总 reasoning/control/reference 梯度及两段 reasoning 梯度。分段 loss **保留完整rollout的reasoning denominator**，因此是实际 objective 的两部分，不是分别除以每段长度。记录两段gradient cosine；范数不应直接相加。Reference norm包含.1系数；所有norm在clipping/Adam前。

`train/metrics/round_*.json` 是完整原始记录，`train/analysis/stability.csv` 每轮展开两段字段；位置分数仍保存为 `train/diagnostics/round_*/position_scores.npz`。诊断串行复用一套梯度，不保留五份GPU梯度，不改变optimizer状态。

## 执行与评估

5张卡常驻 Student DDP+vLLM rollout，1张卡同步 Hindsight+固定 Reference，2张卡 Teacher TP；scoring batch2、train microbatch1。Student vLLM memory .40、max_seqs16，覆盖每rank约13条rollout并给16k训练留显存。服务timeout3600秒，编译缓存隔离并禁用有兼容问题的vLLM compile cache；保留CUDA graph推理。

保留 Round0/4/8/12/latest。为节省资源，本次不做dev生成和dev选点；Round8与Final12在训练前固定用于外部评估，严重collapse guard仍生效，早停时只评已提交checkpoint。

训练结束后自动用现有vLLM评估 Math500/AIME25/OlympiadBench/MMLU-Pro/GPQA Diamond，每集最多199题（共825题/模型），sampling .6/.95/20，response32768、model context40960、safety128，输入不截断。统一使用 `thinking_final_parser`，仅评分 `</think>` 后的最终回答；未闭合思考判未完成。旧完整文本parser保留审计分数。

Base与旧8k Round8/Final12使用历史38,912-token生成的32k前缀，先核对prompt hash、题序、gold和逐request seed，再以相同parser重新评分。它们是复用样本，不是新独立重复。只新生成16k训练得到的checkpoint评估。

```bash
/pfss/mlde/workspaces/mlde_wsp_Eco_Inference/envs/trl/bin/python scripts/run_balanced_horizon.py --run --detach
```

实验目录 `LuLu_outputs/experiments/ren_balanced_abs_full_qwen1p7_teacher32b_pool2048_h16384_r12_s42`。`live_progress.json` 跟踪阶段；结束后自动生成 `REPORT.md`、`comparison.json`、`horizon_training_diagnostics.csv/png`。控制器运行冻结代码；不会停止其他GPU任务。
