# LuLu evaluation

从 `Rona_Soraka/Lulu` 运行。评估复用本项目的 benchmark parser、现有 crossbench Parquet 和逐题结果格式；Student 默认使用 thinking 和 Qwen 推荐采样（T=0.6、top-p=.95、top-k=20），默认输出上限8192，每数据集最多199题。当前五项套件共825题/模型，base和final使用相同题目。

```bash
export PYTHON_BIN=/pfss/mlde/workspaces/mlde_wsp_Eco_Inference/envs/trl/bin/python
export DATA_MANIFEST=../Soraka_rlrl/experiments/v6_5-success-q-scale-pool4096-phase11024-seed42/crossbench_v631/data/manifest.json
CHECKPOINT=/path/to/checkpoints/latest OUTPUT_DIR=/path/to/new-evaluation GPUS=0,1,2,3,4,5,6,7 bash runs/eval_lulu.sh
```

GPU评测默认 `--backend auto` 选择已安装的vLLM；CPU测试使用HF。可显式选择 `--backend hf` 或 `EVAL_BACKEND=hf`。新的vLLM入口需要对应Python环境中的vLLM（当前验证版本0.9.0；依赖组 `fast-eval`）。所有解析器代码已包含在Lulu中，无需设置Soraka代码根目录。

`--decoding auto` 在 thinking 模式选择 `qwen-thinking`；`--decoding greedy` 可复现旧口径。`--seed 42` 控制采样。vLLM 为每道题使用基于 benchmark/index 的独立种子，各 checkpoint 一致，并写入逐题记录；HF 使用固定 batch 的种子，跨 batch/shard 配置不保证逐题重现。单次采样的准确率是单样本 pass@1，不能报告 pass@4。Qwen 官方明确不建议 thinking 使用 greedy：[模型说明](https://huggingface.co/Qwen/Qwen3-1.7B#enable_thinkingtrue)。

## 生成效率

vLLM在每张卡上创建一个常驻实例。该卡所有benchmark的样本一起进入请求队列，完成的请求立即释放槽位。默认最多64条并发序列、4096个调度token、80%显存预算，启用chunked prefill和CUDA graphs；各GPU实例使用独立的vLLM/Inductor/Triton编译缓存，避免多个TP=1 rank0并发写同一路径；对应参数为 `--vllm-max-num-seqs`、`--vllm-max-num-batched-tokens`、`--vllm-gpu-memory-utilization`。`--batch-size` 仅控制HF静态batch。

PEFT checkpoint在主进程中先合并为普通模型，CPU执行一次，保存到评测目录的 `merged_models/MODEL`。八个worker共享该模型文件，避免每个decode token执行额外LoRA投影。原训练checkpoint不变。每个模型的八个worker结束后再运行下一个模型，进程退出释放旧engine、KV cache和CUDA graphs。

HF兼容后端同样在加载时合并adapter，但仍使用静态batch。输入使用相同chat template与token IDs；默认题数上限在分卡前应用。生成后使用相同答案提取和评分逻辑。不同backend的数值内核和采样实现可能造成输出差异，因此一次base/final对照应完整使用同一backend，并在 `eval_plan.json` 中记录；不会中途替换正在运行的backend。

原 `ren_qwen1p7_teacher32b_pool2048_s42/evaluation` 保留冻结的HF/greedy结果。新的中间节点对照位于 `checkpoint_analysis_20260915/evaluation_vllm`，统一评估 base、Step20、Step32，使用推荐采样。两种口径分别记录，不拼接成同一训练曲线。

## 输出与检查

- `eval_plan.json`：模型、backend、样本上限、生成预算和并发设置。
- `worker-MODEL-NNN.log`：vLLM进度和错误；HF日志为 `worker-NNN.log`。
- `MODEL/BENCHMARK/shard-NNN.jsonl`：逐题答案、分数、token数和截断标记。
- `timing-MODEL-NNN.json`：vLLM生成、评分时间与实际输出tokens/s；HF在对应分片旁保存 `.timing.json`。
- `summary.json`：完整base/final准确率和配对差异。缺失、重复、未评分的样本会阻止成功汇总。

`--dry-run` 只解析配置。`--max-examples 0` 恢复完整split；目前样本选择为每个Parquet的固定前N行。LiveCodeBench需指定官方scorer目录，并显式使用 `--max-examples 0`；默认五项不含代码题。早期8-token兼容性检查仅验证入口；2026-09-15三节点补评已用8192输出上限完成2475次生成，单卡实测平均约2200输出tokens/s（不含加载/编译时间），各GPU已释放。

## 保存生成后的CPU重评分

`rescore_lulu_evaluation.py --source /path/evaluation --output-dir /path/new-scored-view` 默认仅重评OlympiadBench，并保留原结果。该修复使用最后一个boxed答案，避免把thinking与最终回答中重复的正确值拼接成错误的多值答案；真正的元组或分数仍保留。`rescore_audit.json` 记录每题旧/新prediction和reward及parser哈希。输入必须是完整评测，不启动模型或GPU任务。


## Long-horizon evaluation with complete prompts

The vLLM evaluator supports `--max-model-len 40960 --max-response-tokens 38912 --context-safety-margin 128`.
Each request receives `min(38912, 40960 - actual_prompt_tokens - 128)` output tokens. Prompts are never truncated;
`--max-prompt-tokens` is a validation limit, and an input that exceeds it fails explicitly. The model context
configuration must support the requested total length; this option does not change RoPE or extend the model.
Outputs record the effective response budget, finish reason, and whether context limited the requested budget.
`--store-token-ids` additionally saves generated IDs and a hash of the complete prompt IDs for paired audits.

The September 16 long-horizon run evaluates initial Base, Round8, and Final12 on the same 825 questions each,
using eight independent GPU queues. Each GPU advances to its next checkpoint without waiting for the other
shards. Full response token IDs allow prefix-budget diagnostics without new generations. A newly generated
long response is not guaranteed to share its prefix with an earlier 8192 run, even with the same sampling seed.

`python scripts/audit_reasoning_positions.py --train TRAIN_DIRECTORY --output AUDIT_DIRECTORY` reads existing
ReN position diagnostics on CPU. It reports both token-weighted statistics and prompt-balanced loss mass,
including a matched capped-trajectory cohort. Its `w * D_C` is a pre-update snapshot loss reconstruction;
Teacher logprob of the actual sampled token cannot be recovered unless it was separately logged.


The reusable evaluation entry point now dispatches `(checkpoint, logical shard)` tasks from a global queue.
Any available physical GPU takes the next task, while logical shard IDs, question IDs, per-request seeds and
output filenames remain fixed. Compiler caches follow the physical GPU slot to prevent concurrent cache races.
`dispatch.jsonl` records this mapping. This improvement applies to subsequent runs; the September 16 run's
frozen evaluator and already-started per-GPU queues remain recorded in its experiment directory.
