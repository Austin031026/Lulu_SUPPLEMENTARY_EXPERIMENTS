#!/usr/bin/env python3
"""Write a final report from the completed stable-ReN run without GPU work."""
import argparse
import json
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--experiment-dir',required=True);a=p.parse_args()
root=Path(a.experiment_dir).resolve()
summary=json.loads((root/'evaluation/summary.json').read_text())
plan=json.loads((root/'experiment_plan.json').read_text())
history=json.loads((root/'train/analysis/stability_history.json').read_text())
state=json.loads((root/'train/checkpoints/latest/lulu_state.json').read_text())
models=['base']+sorted((m for m in summary['models'] if m.startswith('round')),key=lambda m:int(m[5:]))+['final']
benchmarks=list(summary['models']['base']['benchmarks'])
text=['# ReN + fixed-reference stabilization：最终结果','',
      f"完成 {state['completed_rounds']}/{plan['rounds']} 轮，{state['completed_updates']} 次全参数 optimizer update。",
      '本次只训练稳定化 ReN，没有运行额外 Vanilla OPD baseline。Base 是未训练模型的评估参照。','',
      '| Benchmark | N | '+' | '.join(models)+' | Final − Base |',
      '|---|---:|'+'---:|'*(len(models)+1)]
for b in benchmarks:
    values=[summary['models'][m]['benchmarks'][b]['accuracy'] for m in models]
    n=summary['models']['base']['benchmarks'][b]['examples']
    text.append(f'| {b} | {n} | '+' | '.join(f'{v:.2%}' for v in values)+f' | {(values[-1]-values[0])*100:+.2f}pp |')
base=summary['models']['base']['macro_accuracy']; final=summary['models']['final']['macro_accuracy']
text+=['',f'五项等权宏平均：Base {base:.2%} → Final {final:.2%}（{(final-base)*100:+.2f}pp）。',
       '评估为 thinking .6/.95/20，每题一次生成，每数据集前最多 199 题；AIME25 30、GPQA Diamond 198，max response=8192。不能将子集变化直接解释为全量 benchmark 或多 seed 结论。','',
       '| Checkpoint | Hit-cap rate (all examples) |','|---|---:|']
for m in models:
    bs=summary['models'][m]['benchmarks'].values()
    total=sum(b['examples'] for b in bs)
    cap=sum(b['examples']*b['hit_cap_fraction'] for b in bs)/total
    text.append(f'| {m} | {cap:.2%} |')
first,last=history[0],history[-1]
text+=['',f"生成 token 的 loss 覆盖率：首轮 {first['loss_token_fraction']:.2%}，末轮 {last['loss_token_fraction']:.2%}。",
       f"训练 hit-cap：首轮 {first['hit_cap_fraction']:.2%} → 末轮 {last['hit_cap_fraction']:.2%}；平均 response tokens {first['mean_response_tokens']:.0f} → {last['mean_response_tokens']:.0f}。",
       f"训练平均 token 八元组重复率：{first['mean_repeated_8gram_fraction']:.2%} → {last['mean_repeated_8gram_fraction']:.2%}；轮次题目不同，此处仅为诊断。",
       f"末轮 ReN rho：mean={last['rho_mean']:.6f}, max={last['rho_max']:.6f}；KL(Student‖固定初始 Reference)={last['reference_kl']:.6f}。",'']
if (root/'train/early_stop.json').exists():
    stop=json.loads((root/'train/early_stop.json').read_text())
    text+=['**触发稳定性早停**：'+stop['reason']+'；Final 为停止时最后提交的 checkpoint，并非预定第 32 轮。','']
else:
    text+=['未触发预先设置的长度/重复率早停规则；这本身不等于准确率提升或方法有效。','']
text+=['当前目标是 bounded ReN + answer/stop Teacher KL + 固定初始 Student reference KL；去掉了旧版按 batch mean 归一化。',
       f"学习率 {plan['stabilization']['learning_rate']}, reference KL 系数 {plan['stabilization']['reference_kl_coef']}。",
       '未新增 golden-data mixture，因此这是采用 StableOPD reference-KL 思路的 backbone，不是该论文的完整复现。',
       '同一轮 C/H 与 rollout 同源，H 每轮刷新；Reference 始终使用 round0，不刷新。所有 rollout 均保留，verifier 仅作为 metadata。','',
       '![训练稳定性曲线](train/analysis/stability.png)','',
       '[逐轮指标](train/analysis/stability.csv) · [原始评估](evaluation/summary.json) · [实际配置](experiment_plan.json)']
(root/'RESULTS.md').write_text('\n'.join(text)+'\n')
print(root/'RESULTS.md')
