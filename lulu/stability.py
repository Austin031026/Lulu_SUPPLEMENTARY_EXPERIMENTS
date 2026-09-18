"""CPU diagnostics and a predeclared joint length/repetition early-stop rule."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import numpy as np


def repeated_ngrams(ids, n=8):
    count=len(ids)-n+1
    return 1-len({tuple(ids[i:i+n]) for i in range(count)})/count if count>0 else 0.


def collapse_decision(history, min_rounds=5):
    """Use two baseline rounds and two recent rounds, not a single hard prompt batch."""
    if len(history)<max(min_rounds,4):
        return False,''
    mean=lambda entries,key:float(np.mean([r[key] for r in entries]))
    baseline,recent=history[:2],history[-2:]
    cap=mean(recent,'hit_cap_fraction');repetition=mean(recent,'repetition_above_half_fraction')
    cap_up=cap>=max(.75,mean(baseline,'hit_cap_fraction')+.20)
    repetition_up=repetition>=max(.25,mean(baseline,'repetition_above_half_fraction')+.20)
    length_up=mean(recent,'mean_response_tokens')>=1.10*mean(baseline,'mean_response_tokens')
    extreme=cap>=.85 and repetition>=.50
    if (cap_up and repetition_up and length_up) or extreme:
        return True,'Two-round truncation/repetition collapse threshold crossed; stop further updates and evaluate committed checkpoints.'
    return False,''


def record_round(root, index, metrics, args):
    from lulu.training import atomic_json
    root=Path(root)
    records=[]
    for file in sorted((root/'rollouts'/f'round_{index:04d}').glob('shard-*.jsonl')):
        records.extend(json.loads(line) for line in file.read_text().split('\n') if line.strip())
    if len(records)!=args.global_batch_prompts*args.rollouts_per_prompt:
        raise RuntimeError('Stability diagnostics require every on-policy rollout')
    lengths=[len(r['response_ids']) for r in records]
    repetition=[repeated_ngrams(r['response_ids']) for r in records]
    generated=sum(lengths)
    loss_tokens=sum(len(r['positions']) for r in records)
    if loss_tokens!=generated or loss_tokens!=metrics['loss_tokens']:
        raise RuntimeError('Stable backbone failed to cover all generated tokens')
    shared=args.method=='ren_shared'
    rho=metrics['diagnostics']['concentration']['shared_mass' if shared else 'bounded_weight']
    row=dict(round=index+1,trajectories=len(records),generated_tokens=generated,loss_tokens=loss_tokens,
             loss_token_fraction=loss_tokens/generated,reasoning_token_fraction=metrics['reasoning_tokens']/generated,
             hit_cap_fraction=float(np.mean([r['truncated'] for r in records])),
             mean_response_tokens=float(np.mean(lengths)),p90_response_tokens=float(np.quantile(lengths,.9)),
             mean_repeated_8gram_fraction=float(np.mean(repetition)),
             repetition_above_half_fraction=float(np.mean(np.asarray(repetition)>.5)),
             stop_token_fraction=float(np.mean([r['has_stop_token'] for r in records])),
             rollout_accuracy=float(np.mean([r['correct'] is True for r in records])),
             rho_mean=rho['mean'],rho_max=rho['max'],rho_max_to_mean=rho['max']/rho['mean'] if rho['mean'] else 0.,
             rho_top1pct_mass=rho['top_fraction_mass']['0.01'],rho_top10pct_mass=rho['top_fraction_mass']['0.1'],
             reference_kl=metrics['reference_kl'],reference_penalty=metrics['reference_penalty'],
             ren_loss=metrics['ren_loss'],answer_stop_loss=metrics['answer_stop_loss'],
             objective_loss=metrics['objective_loss'],grad_norm=metrics['grad_norm'])
    if args.method in ('ren_balanced','ren_shared'):
        row.update(reasoning_loss_token_global=metrics['reasoning_loss_token_global'],
                   reasoning_loss_rollout_balanced=metrics['reasoning_loss_rollout_balanced'],
                   mean_reasoning_tokens_per_rollout=metrics['mean_reasoning_tokens_per_rollout'],
                   capped_reasoning_loss_share=metrics['capped_reasoning_loss_share'],
                   **({'shared_mass_mean':metrics['diagnostics']['shared_mass_mean'],
                       'teacher_tv_mean':metrics['diagnostics']['teacher_tv_mean'],
                       'shared_fraction_mean':metrics['diagnostics']['shared_fraction_mean'],
                       'target_causal_kl_mean':metrics['diagnostics']['target_causal_kl_mean']} if shared else
                       {'raw_ren_weight_mean':metrics['diagnostics']['raw_ren_weight_mean']}))
        for name in ('reasoning','control','reference','reasoning_early','reasoning_late'):
            row['gradient_norm_'+name]=metrics.get('component_gradient_norms',{}).get(name)
    if getattr(args,'reasoning_diagnostic_split',0):
        row['reasoning_mean_weighted_kl']=metrics['reasoning_mean_weighted_kl']
        row['reasoning_gradient_early_late_cosine']=metrics.get('reasoning_gradient_early_late_cosine')
        for name,values in metrics['reasoning_horizon'].items():
            for key in ('reasoning_tokens','mean_weight','mean_weighted_kl','reasoning_objective_contribution','reasoning_objective_share'):
                row[f'{name}_{key}']=values[key]
    if row['rho_max']>1+1e-7:
        raise RuntimeError('ReN weights are not bounded')
    if shared:
        for key in list(row):
            if key.startswith('rho_'):row['shared_mass_'+key[4:]]=row.pop(key)
    directory=root/'analysis';directory.mkdir(parents=True,exist_ok=True)
    history_file=directory/'stability_history.json'
    history=json.loads(history_file.read_text()) if history_file.exists() else []
    history=[r for r in history if r['round']<index+1]+[row]
    stop,reason=collapse_decision(history,args.stability_min_rounds)
    row.update(early_stop=stop and args.stability_early_stop,stop_reason=reason)
    atomic_json(history_file,history)
    temporary=directory/'stability.csv.tmp'
    with temporary.open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(row));writer.writeheader();writer.writerows(history)
    temporary.replace(directory/'stability.csv')
    render(history,directory)
    return row


def render(history,directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    shared='shared_mass_mean' in history[-1]
    # Plot legacy and shared-target runs with accurate movement labels.
    plotted=[dict(r,**{'rho_'+k[len('shared_mass_'):]:v for k,v in r.items() if k.startswith('shared_mass_')}) if shared else r for r in history]
    x=[r['round'] for r in history]
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    plots=[('Hit-cap / stop token',[('hit_cap_fraction','Hit cap'),('stop_token_fraction','Emitted stop')]),
           ('Response length',[('mean_response_tokens','Mean'),('p90_response_tokens','P90')]),
           ('Repetition (sampled token 8-grams)',[('mean_repeated_8gram_fraction','Mean repetition'),('repetition_above_half_fraction','Fraction > 50%')]),
           ('Bounded ReN weight concentration',[('rho_top1pct_mass','Top 1% mass'),('rho_top10pct_mass','Top 10% mass'),('rho_max','Max rho')]),
           ('Loss coverage / bounded weight scale',[('loss_token_fraction','Loss/generated'),('reasoning_token_fraction','Reasoning/generated'),('rho_mean','Mean rho')]),
           ('Actual weighted loss components',[('ren_loss','ReN'),('answer_stop_loss','Answer/stop'),('reference_penalty','Reference penalty')])]
    for ax,(title,series) in zip(axes.flat,plots):
        if shared:title=title.replace('Bounded ReN weight','Shared target mass').replace('bounded weight','target movement')
        for key,label in series:ax.plot(x,[r[key] for r in plotted],marker='.',label=label.replace('rho','shared mass') if shared else label)
        ax.set_title(title,fontsize=10);ax.set_xlabel('Round');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('ReN + fixed-reference stabilization: live diagnostics')
    temporary=directory/'stability.tmp.png';fig.savefig(temporary,dpi=130);temporary.replace(directory/'stability.png')
    plt.close(fig)
