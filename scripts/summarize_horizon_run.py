"""Compare the 16k Base restart with cached 8k-trained models at a common 32k eval horizon."""
from __future__ import annotations
import argparse,csv,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lulu import benchmark_parser as legacy
from evaluate_lulu import load_complete_rows,summarize_rows,paired_comparison
from run_decisive import write_json


def read_rows(path):return [json.loads(s) for s in path.read_text().splitlines() if s]

def metrics(directory):
    result={}
    for file in sorted(Path(directory).glob('round_*.json')):
        entries=json.loads(file.read_text());assert len(entries)==1
        row=entries[0];result[row['round']+1]=row
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment-dir',required=True);x=p.parse_args();root=Path(x.experiment_dir)
    plan=json.loads((root/'experiment_plan.json').read_text());evalplan=json.loads((root/'evaluation/eval_plan.json').read_text())
    old_summary=json.loads((root/'reference_32k/summary.json').read_text());summary=dict(old_summary)
    reference={b['name']:read_rows(root/'reference_32k/base'/b['name']/'rows.jsonl') for b in evalplan['benchmarks']}
    paired={};score_audit={};new_rows={}
    rng=np.random.default_rng(42)
    for model in evalplan['models']:
        name='train16k_'+model['name'];summary[name]={};paired[name]={};score_audit[name]={};new_rows[name]={}
        for b in evalplan['benchmarks']:
            dataset=b['name'];rows=load_complete_rows(root/'evaluation'/model['name']/dataset,len(evalplan['devices']),b['expected_examples'])
            base=reference[dataset];assert len(base)==len(rows)
            for a,c in zip(base,rows):
                for field in ['prompt_index','prompt_sha256','generation_seed','ground_truth']:assert a[field]==c[field],(dataset,field)
                assert c['max_response_tokens']==32768 and c['prompt_tokens']+32768<=40960-128
            summary[name][dataset]=summarize_rows(rows);paired[name][dataset]=paired_comparison(base,rows)
            delta=np.array([c['reward']-a['reward'] for a,c in zip(base,rows)])
            draws=rng.integers(0,len(delta),(4000,len(delta)))
            paired[name][dataset]['paired_question_bootstrap_ci95']=np.quantile(delta[draws].mean(1),[.025,.975]).tolist()
            legacy_rewards=[]
            for r in rows:
                pred=legacy.extract_answer(r['response'],r['data_source']);legacy_rewards.append(float(legacy.math_equal(pred,r['ground_truth'])))
            score_audit[name][dataset]=dict(legacy_accuracy=float(np.mean(legacy_rewards)),final_only_accuracy=summary[name][dataset]['accuracy'],
                differing_examples=sum(v!=r['reward'] for v,r in zip(legacy_rewards,rows)))
            new_rows[name][dataset]=rows
    old=metrics(Path(plan['old_experiment'])/'train/metrics');new=metrics(root/'train/metrics');training=[]
    for round_,m in new.items():
        previous=old[round_];bins=m['reasoning_horizon'];g=m.get('component_gradient_norms',{});og=previous.get('component_gradient_norms',{})
        assert abs(sum(v['reasoning_objective_contribution'] for v in bins.values())-m['ren_loss'])<1e-5
        row=dict(round=round_,old8k_mean_w=previous['rho_sum']/previous['reasoning_tokens'],new16k_mean_w=m['rho_sum']/m['reasoning_tokens'],
            old8k_mean_wKL=previous['reasoning_loss_token_global']*previous['loss_tokens']/previous['reasoning_tokens'],new16k_mean_wKL=m['reasoning_mean_weighted_kl'],
            old8k_reasoning_grad=og.get('reasoning'),new16k_reasoning_grad=g.get('reasoning'),
            new16k_control_grad=g.get('control'),new16k_reference_grad=g.get('reference'),
            new16k_reasoning_loss=m['ren_loss'],new16k_control_loss=m['answer_stop_loss'],new16k_reference_penalty=m['reference_penalty'],
            early_late_gradient_cosine=m.get('reasoning_gradient_early_late_cosine'))
        for label,b in bins.items():
            for key in ['reasoning_tokens','mean_weight','mean_weighted_kl','reasoning_objective_contribution','reasoning_objective_share']:row[label+'_'+key]=b[key]
            row[label+'_gradient_norm']=g.get('reasoning_'+label)
        training.append(row)
    with (root/'horizon_training_diagnostics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(training[0]));w.writeheader();w.writerows(training)
    output=dict(summary=summary,paired_vs_base=paired,scoring_audit=score_audit,
        macro_accuracy={model:float(np.mean([v['accuracy'] for v in benchmarks.values()])) for model,benchmarks in summary.items()},
        training=training,notes=['Same fixed 2048-prompt pool, schedule, initial Base, optimizer and loss coefficients; horizon 8192 to 16384.',
            'Old references are cached long-budget sampled-policy prefixes, not independent new generations.',
            'Primary score parses final answer after </think>; missing closure fails. All references use this same scorer.',
            'Each bin contribution retains the full-rollout reasoning denominator; conditional token means use all reasoning tokens within the bin.',
            'Exact full-batch gradient norms at updates 1,4,8,12 are before clipping/Adam; gradient vector norms do not add.',
            'No dev-based checkpoint selection; round8 and final fixed before training. External test scores never control training.'])
    write_json(root/'comparison.json',output)
    models=list(summary);datasets=[b['name'] for b in evalplan['benchmarks']]
    lines=['# 16k trajectory-balanced bounded absolute ReN','',
        '从同一个 Base checkpoint 重新进行全参数 on-policy 训练；16,384-token response horizon，固定 2048 DAPO prompt pool，12轮×64题（若触发既有 collapse guard 则提前停止）。Teacher Qwen3-32B；Student Qwen3-1.7B thinking。','',
        '目标保持 `g=[KL(T||C)-KL(T||H)]+; w=g/(1+g)`。Reasoning 按每条 rollout 的全部 reasoning token 数平均，再按 rollout/prompt 等权；不按权重和重归一化。Control/reference 全局 token mean 与 coefficient 保持不变，LR=1e-6，reference coefficient=0.1。','',
        '## 统一 32k 外部评估','',
        '全部使用 thinking sampling 0.6/0.95/20，每集最多199题（AIME30、GPQA198），每模型825题；包含 MMLU-Pro 与 GPQA general reasoning。原题不截断，保留128 token上下文余量。Base及旧8k模型复用历史长回答的32k前缀并重新评分；新16k模型使用8卡vLLM生成。','',
        '**主要评分只解析 `</think>` 后的最终回答，所有模型使用相同规则。** 旧完整文本 parser 的对照分数保留在 comparison.json；不混用旧评分报告。','',
        '| Model | '+' | '.join(datasets)+' | Macro |','|---|'+'---:|'*(len(datasets)+1)]
    for model in models:lines.append('| '+model+' | '+' | '.join(f"{100*summary[model][d]['accuracy']:.2f}%" for d in datasets)+f" | {100*output['macro_accuracy'][model]:.2f}% |")
    lines+=['','各数据集与 Base 的配对 rescues/degradations 和 95% question-bootstrap CI 见 comparison.json。小样本/单次 sampling；macro 对不同大小的数据集等权，不能据微小变化宣称稳定提升。','',
        '## 训练长度与梯度审计','',
        '每轮记录两段 E[w]、E[w KL]、reasoning token 数、实际 prompt-balanced loss contribution。梯度在第1/4/8/12次更新前，对同一个完整global batch精确计算，不采用梯度子样本或仅LM-head近似。两段的梯度保留完整 rollout 的 reasoning denominator，因此可以观察16k平均带来的真实稀释；没有人为放大后8k。','',
        '| Update | 8k ∥∇LR∥ | 16k ∥∇LR∥ | 0–8k contribution norm | 8–16k contribution norm | cosine |','|---|---:|---:|---:|---:|---:|']
    for row in training:
        if row['new16k_reasoning_grad'] is None:continue
        values=[row['old8k_reasoning_grad'],row['new16k_reasoning_grad'],row['early_gradient_norm'],row['late_gradient_norm'],row['early_late_gradient_cosine']]
        lines.append(f"| {row['round']} | "+' | '.join('—' if v is None else f'{v:.6g}' for v in values)+' |')
    lines+=['','完整逐轮数据：horizon_training_diagnostics.csv；旧8k与新16k比较保留各自的on-policy轨迹，因此后续轮次梯度差还包含策略/状态分布变化，并非纯粹分母效应。梯度范数也不等于Adam后的参数更新幅度。','',
        '![Horizon diagnostics](horizon_training_diagnostics.png)','',
        '## 复现与执行','',
        'experiment_plan.json 固定数据/模型/代码哈希、完整命令和预先确定的评估节点。训练日志 train/metrics/round_*.json 包含全部分量；train/diagnostics/round_*/position_scores.npz 保留逐位置分数；train/rollouts 保留包括失败/截断在内的所有轨迹。Reference 与 hindsight 参数分别固定初始Base、每轮同步Student快照。','',
        'CPU验证涵盖梯度与显式dense公式一致、开启诊断不改变optimizer更新、不均匀DDP分片归约，以及final-only parser。无需重新运行 baseline generation。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained');rounds=[r['round'] for r in training]
    for key,label in [('old8k_mean_w','8k run'),('early_mean_weight','16k: positions 0-8k'),('late_mean_weight','16k: positions 8-16k')]:axes[0].plot(rounds,[r[key] for r in training],marker='.',label=label)
    for key,label in [('old8k_mean_wKL','8k run'),('early_mean_weighted_kl','16k: positions 0-8k'),('late_mean_weighted_kl','16k: positions 8-16k')]:axes[1].plot(rounds,[r[key] for r in training],marker='.',label=label)
    measured=[r for r in training if r['new16k_reasoning_grad'] is not None]
    for key,label in [('old8k_reasoning_grad','8k total'),('new16k_reasoning_grad','16k total'),('early_gradient_norm','16k early contribution'),('late_gradient_norm','16k late contribution')]:axes[2].plot([r['round'] for r in measured],[r[key] for r in measured],marker='o',label=label)
    for ax,title in zip(axes,['Mean bounded weight','Mean weighted Teacher KL','Exact reasoning gradient norms']):ax.set_title(title);ax.set_xlabel('Update');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(root/'horizon_training_diagnostics.png',dpi=170);plt.close(fig)
    print(json.dumps({'status':'complete','macro_accuracy':output['macro_accuracy'],'report':str(root/'REPORT.md')},indent=2))
if __name__=='__main__':main()
