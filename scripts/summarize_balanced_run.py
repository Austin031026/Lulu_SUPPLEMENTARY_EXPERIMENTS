#!/usr/bin/env python3
"""Report the dev-selected model and compare with frozen existing Base responses."""
from __future__ import annotations
import argparse,json,hashlib
from pathlib import Path
import evaluate_lulu as evaluation


def summarize(root):
    root=Path(root).resolve();plan=json.loads((root/'experiment_plan.json').read_text())
    selected=json.loads((root/'train/validation/selected.json').read_text())
    dev=json.loads((root/'train/validation/history.json').read_text())
    metrics=json.loads((root/'train/analysis/stability_history.json').read_text())
    current=json.loads((root/'evaluation/summary.json').read_text())
    source=Path(plan['reference_evaluation']['path'])
    previous=json.loads((source/'summary.json').read_text())
    current_plan=json.loads((root/'evaluation/eval_plan.json').read_text())
    previous_plan=json.loads((source/'eval_plan.json').read_text())
    for key in ('sampling','thinking','max_response_tokens','max_prompt_tokens','max_examples','dtype','decoding','split','backend','vllm'):
        if current_plan[key]!=previous_plan[key]:raise ValueError(f'Frozen Base protocol mismatch: {key}')
    if current_plan['benchmarks']!=previous_plan['benchmarks']:
        raise ValueError('Frozen Base benchmark definitions differ')
    if previous['models']['base']['model']!=plan['student']:
        raise ValueError('Frozen Base model differs from initial Student')
    for relative,expected in plan['reference_evaluation']['shard_sha256'].items():
        if hashlib.sha256((source/relative).read_bytes()).hexdigest()!=expected:raise ValueError('Frozen Base answers changed')
    if Path(current['models']['selected']['model']).resolve()!=Path(selected['checkpoint']).resolve():
        raise ValueError('Benchmark did not evaluate the dev-selected model')
    comparisons={}
    for benchmark in current_plan['benchmarks']:
        name=benchmark['name'];n=benchmark['expected_examples']
        base=evaluation.load_complete_rows(source/'base'/name,len(previous_plan['devices']),n)
        live=evaluation.load_complete_rows(root/'evaluation/selected'/name,len(current_plan['devices']),n)
        if {r['prompt_index']:r['generation_seed'] for r in base}!={r['prompt_index']:r['generation_seed'] for r in live}:
            raise ValueError('Per-question generation seeds differ')
        comparisons[name]={'base':evaluation.summarize_rows(base),'selected':evaluation.summarize_rows(live),
                           'paired':evaluation.paired_comparison(base,live)}
    reused=current.get('cache_reuse')
    output={'evaluation_cache_reuse':reused,'selected_checkpoint':selected,'base_reused_from':str(source),
            'selection_uses_only_dev':True,'benchmarks':comparisons,
            'base_macro_accuracy':previous['models']['base']['macro_accuracy'],
            'selected_macro_accuracy':current['models']['selected']['macro_accuracy']}
    (root/'comparison.json').write_text(json.dumps(output,indent=2)+'\n')
    text=['# Prompt-balanced absolute-ReN：最终结果','',
          f"完成 {len(metrics)}/{plan['rounds']} 轮；dev 选择 Round{selected['round']}，dev accuracy={selected['accuracy']:.2%}。",
          ('本轮只训练新方法；dev 选择初始模型，因此复用经一致性验证的 Base benchmark 回答。没有根据 benchmark 结果重新选模型。' if reused else '本轮只训练新方法。外部 benchmark 只生成 dev 所选模型一次；Base 使用上轮已保存且修正解析的同协议回答。没有根据 benchmark 结果重新选模型。'),'',
          '| Dev checkpoint | 正确 / 256 | Accuracy | Hit cap |','|---|---:|---:|---:|']
    for row in dev:text.append(f"| Round{row['round']} | {row['correct']}/{row['examples']} | {row['accuracy']:.2%} | {row['hit_cap_fraction']:.2%} |")
    text+=['','| Benchmark | N | Frozen Base | Selected | Δ |','|---|---:|---:|---:|---:|']
    for name,row in comparisons.items():text.append(f"| {name} | {row['selected']['examples']} | {row['base']['accuracy']:.2%} | {row['selected']['accuracy']:.2%} | {row['paired']['accuracy_delta']*100:+.2f}pp |")
    text+=['',f"宏平均：{output['base_macro_accuracy']:.2%} → {output['selected_macro_accuracy']:.2%}，Δ={(output['selected_macro_accuracy']-output['base_macro_accuracy'])*100:+.2f}pp。",'',
           'reasoning 先除以每条 rollout 的 reasoning token 数（含零权重位置），再对同题 rollout 平均，再对 prompt 平均。w=g/(1+g)，不做 weight sum/mean 归一化。control/reference 保持原全局生成 token 平均及系数。','',
           '| Round | Cap | Cap share of balanced reasoning loss | Token-global reasoning audit | Balanced reasoning loss |','|---|---:|---:|---:|---:|']
    for row in metrics:text.append(f"| {row['round']} | {row['hit_cap_fraction']:.2%} | {row['capped_reasoning_loss_share']:.2%} | {row['reasoning_loss_token_global']:.6f} | {row['reasoning_loss_rollout_balanced']:.6f} |")
    text+=['','| Round | ‖∇reason‖ | ‖∇control‖ | ‖∇weighted reference‖ |','|---|---:|---:|---:|']
    for row in metrics:
        if row['gradient_norm_reasoning'] is not None:text.append(f"| {row['round']} | {row['gradient_norm_reasoning']:.6f} | {row['gradient_norm_control']:.6f} | {row['gradient_norm_reference']:.6f} |")
    if reused:
        text+=['', '**本轮 dev 未选择任何训练后 checkpoint。所选 round0 已逐参数、tokenizer 和模板核对与 Base 完全一致，因此复用原 825 条回答，没有重复生成。表中零差值来自同一缓存，不代表另一次独立评估，也不表示新方法带来了性能提升。**']
    text+=['','梯度范数来自相同完整 global batch、所有可训练参数，位于 clipping/Adam 之前；不是更新贡献百分比。不同分量可能互相抵消，不能直接用范数比解释参数更新份额。',
           'Greedy dev 使用 temperature=0、固定题序/分片/预算；不声称底层 GPU 浮点计算保证 bitwise 重现。外部评估沿用 thinking .6/.95/20，最多199题，存在生成噪声和小样本不确定性。',
           f"固定训练池2048题，本次使用 {len(metrics)*plan['global_batch_prompts']} 条 prompt exposure；没有为了跑满训练池而延长预算。",'',
           '[实际计划](experiment_plan.json) · [Dev 选择](train/validation/selected.json) · [逐轮诊断](train/analysis/stability.csv) · [配对比较](comparison.json)',
           '![训练诊断](train/analysis/stability.png)']
    (root/'RESULTS.md').write_text('\n'.join(text)+'\n')
    print(json.dumps(output,indent=2))
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment-dir',required=True);a=p.parse_args();summarize(a.experiment_dir)
