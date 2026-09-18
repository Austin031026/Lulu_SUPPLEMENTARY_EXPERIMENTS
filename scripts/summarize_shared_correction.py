"""Validate shared-correction training, pair frozen evaluations and write a report."""
from pathlib import Path
import argparse,json
import numpy as np
from evaluate_lulu import load_complete_rows,summarize_rows,paired_comparison
from summarize_gate_diagnostics import load_evaluation_rows
from run_decisive import write_json


def summarize(root):
    plan=json.loads((root/'experiment_plan.json').read_text());old=Path(plan['historical_experiment'])
    previous=json.loads((old/'comparison.json').read_text());old_plan=json.loads((old/'experiment_plan.json').read_text())
    ep=json.loads((root/'evaluation/eval_plan.json').read_text());name='D_shared'
    benchmarks=ep['benchmarks'];external=[b['name'] for b in benchmarks if b['name']!='dapo_dev128']
    rows={};scores={};rng=np.random.default_rng(20260917)
    for model in ['base','A_control_ref','B_vanilla','C_ren',name]:
        rows[model]={};scores[model]={}
        for b in benchmarks:
            ds=b['name'];n=b['expected_examples']
            if model==name:rr=load_complete_rows(root/'evaluation'/name/ds,len(ep['devices']),n)
            elif model=='base' and ds!='dapo_dev128':
                path=Path(old_plan['source_16k'])/'reference_32k/base'/ds/'rows.jsonl'
                rr=sorted([json.loads(s) for s in path.read_text().splitlines()],key=lambda r:r['prompt_index'])
            else:rr=load_evaluation_rows(old,model,ds,n)
            if len(rr)!=n:raise ValueError('Evaluation coverage mismatch')
            if model!='base':
                for x,y in zip(rows['base'][ds],rr):
                    if not all(x[k]==y[k] for k in ('prompt_index','prompt_sha256','generation_seed','ground_truth')):
                        raise ValueError('Comparison prompt/seed/gold mismatch')
            rows[model][ds]=rr;scores[model][ds]=summarize_rows(rr)
            if model!=name and scores[model][ds]!=previous['scores'][model][ds]:
                raise ValueError('Historical evaluation changed')
    comparisons={}
    for reference in ['base','A_control_ref','B_vanilla','C_ren']:
        by={};draws=[]
        for ds in [b['name'] for b in benchmarks]:
            r=paired_comparison(rows[reference][ds],rows[name][ds]);n=r['paired_examples']
            up,down=r['rescues'],r['degradations'];z=rng.multinomial(n,[up/n,down/n,1-(up+down)/n],size=10000)
            boot=(z[:,0]-z[:,1])/n;r['ci95']=np.quantile(boot,[.025,.975]).tolist();by[ds]=r
            if ds in external:draws.append(boot)
        comparisons[reference]=dict(benchmarks=by,external_macro_delta=float(np.mean([by[b]['accuracy_delta'] for b in external])),
            external_macro_ci95=np.quantile(np.mean(draws,axis=0),[.025,.975]).tolist())
    dest=root/'arms'/name/'train';metrics=[]
    for i in range(4):
        m=json.loads((dest/'metrics'/f'round_{i:04d}.json').read_text())[0]
        if m['completed_updates']!=i+1 or m['skipped_update'] or m['reasoning_target']!='shared_positive_probability_correction':
            raise ValueError('Unexpected training update')
        rr=[json.loads(line) for p in (dest/'rollouts'/f'round_{i:04d}').glob('shard-*.jsonl') for line in p.read_text().splitlines()]
        prior=[json.loads(line)['source_id'] for p in (old/'arms/C_ren/train/rollouts'/f'round_{i:04d}').glob('shard-*.jsonl') for line in p.read_text().splitlines()]
        if len(rr)!=64 or any(r['snapshot_round']!=i or r['response_tokens']>8192 for r in rr):raise ValueError('On-policy rollout coverage mismatch')
        if sorted(r['source_id'] for r in rr)!=sorted(prior):raise ValueError('Training prompt schedule changed')
        metrics.append(m)
    middle_plan=json.loads((root/'intermediate_dev/eval_plan.json').read_text())
    middle=load_complete_rows(root/'intermediate_dev/D_shared_round2/dapo_dev128',len(middle_plan['devices']),128)
    for x,y in zip(rows['base']['dapo_dev128'],middle):
        if not all(x[k]==y[k] for k in ('prompt_index','prompt_sha256','generation_seed','ground_truth')):raise ValueError('Intermediate dev mismatch')
    macro={model:float(np.mean([scores[model][b]['accuracy'] for b in external])) for model in scores}
    micro={model:float(np.mean([r['reward'] for b in external for r in rows[model][b]])) for model in scores}
    result=dict(scores=scores,external_macro=macro,external_micro=micro,paired_vs= comparisons,training=metrics,
        round2_dev=summarize_rows(middle),validation=dict(all_four_updates_committed=True,prompt_seed_gold_pairing=True,
        same_training_prompt_schedule=True,baseline_training_launched=False),
        caveats=['One seed and four updates; paired question intervals condition on sampled models and do not cover training-seed variance.',
                 'Base external responses reuse long outputs clipped/rescored at32k; not a fresh generation.',
                 'Five Student workers versus four historically; request seeds and global objective are unchanged, batch/numerical schedules differ.',
                 'Hindsight supports positive recipients, not necessarily donor decreases. Target TV=m does not bound actual neural update TV.',
                 'Round2 dev is descriptive; fixed round4 external evaluation, no checkpoint selection.'])
    write_json(root/'comparison.json',result)
    lines=['# Shared positive correction：4轮训练与评估','',
        '已完成从 Base 开始的全参数 on-policy 训练。reasoning 使用 action-level 共享概率纠正；control/reference 保持原实现，未增加 scalar gate 或按纠正质量归一化。','',
        '| 数据集 | N | Base | A control/ref | B Vanilla | C bounded ReN | D shared correction |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for b in benchmarks:
        ds=b['name'];lines.append(f'| {ds} | {b["expected_examples"]} | '+' | '.join(f'{scores[model][ds]["accuracy"]:.2%}' for model in scores)+' |')
    lines+=['| 外部五集 macro | — | '+' | '.join(f'{macro[m]:.2%}' for m in scores)+' |',
        '| 外部825题 micro | 825 | '+' | '.join(f'{micro[m]:.2%}' for m in scores)+' |','',
        f'Round2 dev128：{result["round2_dev"]["accuracy"]:.2%}；Round4 dev128：{scores[name]["dapo_dev128"]["accuracy"]:.2%}。最终checkpoint事先固定为Round4。','',
        '| D 对比 | 外部macro变化(pp) | 配对题目bootstrap 95%CI(pp) |','|---|---:|---|']
    for base,value in comparisons.items():lines.append(f'| {base} | {100*value["external_macro_delta"]:+.3f} | {[round(100*x,3) for x in value["external_macro_ci95"]]} |')
    lines+=['','| Round | E[m] | E[M] | E[m/M] | Reasoning KL | Control loss | Ref penalty | R grad norm | Total preclip norm | Cap |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in metrics:
        d=m['diagnostics'];norm=m.get('component_gradient_norms',{}).get('reasoning');norm='—' if norm is None else f'{norm:.6g}'
        lines.append(f'| {m["completed_updates"]} | {d["shared_mass_mean"]:.6g} | {d["teacher_tv_mean"]:.6g} | {d["shared_fraction_mean"]:.6g} | {m["ren_loss"]:.6g} | {m["answer_stop_loss"]:.6g} | {m["reference_penalty"]:.6g} | {norm} | {m["grad_norm"]:.6g} | {m["stability"]["hit_cap_fraction"]:.2%} |')
    final=comparisons['base'];lo,hi=final['external_macro_ci95']
    interpretation=('区间跨过零，当前短实验不足以确认泛化提升。' if lo<=0<=hi else
        '配对题目区间高于零，存在初步正向信号，仍需注意单seed和4步的限制。' if lo>0 else
        '配对题目区间低于零，当前配置下出现退化。')
    lines+=['',interpretation,'',
        'm 是目标相对 frozen causal Student 的共享概率质量，M 是 Teacher 与 causal Student 的 TV；日志中的 target KL 不是 m×Teacher KL。各轮 position_scores.npz 包含每个 reasoning 位置的 m、M、目标KL、位置与capped标记，支持后续无GPU审计。','',
        '8张卡分配为5 Student、1 hindsight/reference、2 Teacher TP；eval使用独立vLLM worker。全局64题batch、prompt schedule、采样设置和loss归一化与旧实验一致，GPU分片改变可能导致数值差异。','',
        '需要保留的解释限制：H只认证增概率的action；donor由Teacher决定。目标TV=m不等于实际Adam更新的策略TV。零纠正位置仍有对当前轮pC的KL锚定，可能约束其他位置的参数共享更新。现有scalar gate审计没有验证新action correction的正确性特异性。','',
        'Base/A/B/C来自已完成实验，没有重跑基线；Base外部结果为现有长预算回答在32k截断并重评分。所有模型使用相同题目、request seed和最终答案解析；每个外部集合最多199题，AIME30、GPQA198。不得仅凭AIME的1–2题变化认定普遍提升。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment-dir',required=True);a=p.parse_args()
    result=summarize(Path(a.experiment_dir));print(json.dumps(dict(external_macro=result['external_macro'],round2_dev=result['round2_dev']),indent=2))
if __name__=='__main__':main()
