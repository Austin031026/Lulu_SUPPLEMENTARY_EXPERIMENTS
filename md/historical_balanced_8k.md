# 历史 8k bounded absolute ReN：完整节点数据

原实验：`ren_balanced_abs_full_qwen1p7_teacher32b_pool2048_dev256_r12_s42`。Student Qwen3-1.7B 全参数、Teacher Qwen3-32B；DAPO固定池2048，64题/轮×12轮，共768个训练题；学习率1e-6，control=1、reference=0.1。训练 horizon=8192。

以下均为当时保存的历史统计，未重新生成或用最新parser重评分。正向迹象不等同已经确认的收益。

## 外部五集：原始 8k thinking sampling

T=0.6 / top-p=.95 / top-k=20，每题一次生成。每集至多199题，共825题/模型。原主口径允许从整个生成文本提取答案，可能计入未结束thinking中的答案。

| Benchmark | N | Base | Round8 | Final12 |
|---|---:|---:|---:|---:|
| math500 | 199 | 80.40% | 81.41% | 83.92% |
| aime25 | 30 | 20.00% | 26.67% | 20.00% |
| olympiadbench | 199 | 46.23% | 46.23% | 46.73% |
| mmlu_pro | 199 | 57.79% | 56.28% | 60.30% |
| gpqa_diamond | 198 | 26.77% | 28.28% | 29.80% |
| Macro | — | 46.24% | 47.77% | 48.15% |

| Model | Pooled accuracy | Hit cap | Mean response tokens |
|---|---:|---:|---:|
| Base | 51.64% | 36.73% | 5215.1 |
| Round8 | 52.12% | 35.52% | 5146.0 |
| Final12 | 53.94% | 33.94% | 5099.4 |

round8 对 Base：+1.54 pp；配对题目 bootstrap 95% CI [-0.95, +4.31] pp。

final 对 Base：+1.91 pp；配对题目 bootstrap 95% CI [-0.79, +4.66] pp。

## 仅评分结束 thinking 后的回答（附加审计）

| Benchmark | Base | Round8 | Final12 |
|---|---:|---:|---:|
| math500 | 77.89% | 80.40% | 81.41% |
| aime25 | 16.67% | 20.00% | 16.67% |
| olympiadbench | 42.21% | 41.21% | 40.70% |
| mmlu_pro | 54.77% | 55.78% | 57.79% |
| gpqa_diamond | 23.23% | 23.23% | 25.25% |
| Macro | 42.95% | 44.12% | 44.36% |

## 训练中的独立 dev256：Round0 / 4 / 8 / 12

dev 使用 greedy、8k；与上面的外部 thinking sampling 不是同一评估口径。事先的 dev 选择规则最终选中 Round0。之后 Round8/Final12 外部补评是描述性验证，不能重新包装成独立 dev 选模成功。

| Round | Correct / 256 | Accuracy | Hit cap | Mean response tokens |
|---|---:|---:|---:|---:|
| 0 | 115/256 | 44.92% | 48.44% | 6555.2 |
| 4 | 104/256 | 40.62% | 49.22% | 6515.5 |
| 8 | 109/256 | 42.58% | 46.48% | 6399.4 |
| 12 | 110/256 | 42.97% | 48.05% | 6398.6 |

Round4 没有外部五集评估记录，不填推测值。

## 相同 checkpoint 后续的长预算验证

38,912 response cap，逐题限制 min(38912,40960−prompt_length−128)，不截断题目。该轮是重新生成，不能认为与旧8k采样具有相同前缀。

| Benchmark | Base | Round8 | Final12 |
|---|---:|---:|---:|
| math500 | 89.45% | 89.95% | 89.45% |
| aime25 | 40.00% | 40.00% | 36.67% |
| olympiadbench | 66.33% | 65.83% | 62.31% |
| mmlu_pro | 64.32% | 63.82% | 62.81% |
| gpqa_diamond | 41.41% | 33.84% | 34.85% |
| Macro | 60.30% | 58.69% | 57.22% |

8k 的小幅正向迹象未在这次长预算复评中复现。单seed、单次采样和查看多个节点的限制都仍然存在。

## 训练记录与原始来源

`training_round1_to12.csv` 导出每轮 loss、梯度范数和 cap 统计；未测梯度范数的节点留空，不填0。

- [8k报告](/pfss/mlde/workspaces/mlde_wsp_Eco_Inference/Rona_Soraka/LuLu_outputs/experiments/ren_balanced_abs_full_qwen1p7_teacher32b_pool2048_dev256_r12_s42/verification_round8_final_20260916/REPORT.md)
- [长预算报告](/pfss/mlde/workspaces/mlde_wsp_Eco_Inference/Rona_Soraka/LuLu_outputs/experiments/ren_balanced_abs_full_qwen1p7_teacher32b_pool2048_dev256_r12_s42/long_horizon_38912_20260916/REPORT.md)
- [dev history](/pfss/mlde/workspaces/mlde_wsp_Eco_Inference/Rona_Soraka/LuLu_outputs/experiments/ren_balanced_abs_full_qwen1p7_teacher32b_pool2048_dev256_r12_s42/train/validation/history.json)

所有CSV中的accuracy、hit_cap等比例均采用0–1，Markdown表格采用百分比。`sources.json`保留源文件路径及SHA256。
