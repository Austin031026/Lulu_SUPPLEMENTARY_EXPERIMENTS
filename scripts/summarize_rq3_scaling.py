#!/usr/bin/env python3
"""CPU-only RQ3.1 learning-curve, heatmap, coverage and compute tables."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from rq_metrics import MATH,OOD,EXTERNAL,grouped_metrics,with_deltas


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)


def summarize(root):
    root=Path(root);plan=json.loads((root/'experiment_plan.json').read_text());out=root/'analysis';out.mkdir(exist_ok=True)
    dev=json.loads((root/'evaluation_dev/summary.json').read_text());final=json.loads((root/'evaluation_final/summary.json').read_text())
    dev_base=dev['models']['base']['benchmarks']['dapo_dev128']['accuracy'];base_metrics=grouped_metrics(final['models']['base'])
    curves=[];heat=[];coverage=[];compute=[]
    for row in plan['matrix']:
        name=row['name'];batch=row['batch'];budget=row['unique_prompts']
        for r in range(1,5):
            key=f'{name}_r{r}';acc=dev['models'][key]['benchmarks']['dapo_dev128']['accuracy']
            curves.append({'arm':name,'method':row['arm'],'final_budget':budget,'batch':batch,'round':r,
                           'cumulative_unique_prompts':batch*r,'dev_accuracy':acc,'dev_delta':acc-dev_base})
            diag=json.loads((root/'arms'/name/'train/diagnostics'/f'round_{r-1:04d}'/'summary.json').read_text());metric=json.loads((root/'arms'/name/'train/metrics'/f'round_{r-1:04d}.json').read_text())[0]
            c=diag['concentration']['applied_weight'];positions=max(c['positions'],1);prompts=max(diag.get('prompt_count') or batch,1)
            coverage.append({'arm':name,'method':row['arm'],'budget':budget,'batch':batch,'round':r,
                             'reasoning_positions':c['positions'],'token_weight_ess':c['effective_positions'],'token_weight_ess_fraction':c['effective_positions']/positions,
                             'prompt_count':prompts,'prompt_loss_ess':diag['prompt_reasoning_loss_ess'],'prompt_loss_ess_fraction':diag['prompt_reasoning_loss_ess']/prompts,
                             'top1_weight_mass':c['top_fraction_mass'].get('0.01'),'top10_weight_mass':c['top_fraction_mass']['0.1'],
                             'applied_weight_mean':diag['applied_weight_mean'],
                             'retained_teacher_kl_prompt_balanced':diag.get('teacher_supervision_retained_fraction_prompt_balanced',diag.get('teacher_supervision_retained_fraction')),
                             'retained_teacher_kl_token_global':diag.get('teacher_supervision_retained_fraction_token_global'),
                             'round_seconds':metric.get('round_seconds')})
        model_metrics=grouped_metrics(final['models'][name]);deltas=with_deltas(model_metrics,base_metrics)
        # Raw benchmark rows plus the grouped rows used in the planned 8xB heatmap.
        for b in EXTERNAL:
            heat.append({'arm':name,'method':row['arm'],'budget':budget,'benchmark':b,'accuracy':model_metrics[b],'delta':deltas.get(b+'_delta')})
        for key,label in [('math_avg','math_avg'),('ood_avg','ood_avg'),('external_micro','external_micro')]:
            heat.append({'arm':name,'method':row['arm'],'budget':budget,'benchmark':label,'accuracy':model_metrics[key],'delta':deltas.get(key+'_delta')})
        metrics=[json.loads((root/'arms'/name/'train/metrics'/f'round_{r:04d}.json').read_text())[0] for r in range(4)]
        compute.append({'arm':name,'method':row['arm'],'budget':budget,'batch':batch,'rounds':4,
                        'math_avg':model_metrics['math_avg'],'math_delta':deltas.get('math_avg_delta'),
                        'ood_avg':model_metrics['ood_avg'],'ood_delta':deltas.get('ood_avg_delta'),
                        'external_micro':model_metrics['external_micro'],'external_micro_delta':deltas.get('external_micro_delta'),
                        'round_seconds':sum(m.get('round_seconds',0) or 0 for m in metrics),
                        'response_tokens':sum(m.get('response_tokens',0) or 0 for m in metrics),
                        'causal_prompt_tokens':sum(m.get('causal_prompt_tokens',0) or 0 for m in metrics),
                        'hindsight_prompt_tokens':sum(m.get('hindsight_prompt_tokens',0) or 0 for m in metrics),
                        'teacher_sequence_tokens':sum(m.get('teacher_sequence_tokens',0) or 0 for m in metrics),
                        'hindsight_sequence_tokens':sum(m.get('hindsight_sequence_tokens',0) or 0 for m in metrics),
                        'teacher_positions':sum(m.get('teacher_positions',0) or 0 for m in metrics),
                        'hindsight_positions':sum(m.get('hindsight_positions',0) or 0 for m in metrics),
                        'teacher_seconds':sum(m.get('teacher_seconds',0) or 0 for m in metrics),
                        'hindsight_seconds':sum(m.get('hindsight_seconds',0) or 0 for m in metrics)})
    write_csv(out/'learning_curve.csv',curves);write_csv(out/'benchmark_heatmap.csv',heat);write_csv(out/'coverage.csv',coverage);write_csv(out/'compute_frontier.csv',compute)
    summary={'learning_curve_rows':len(curves),'heatmap_rows':len(heat),'coverage_rows':len(coverage),'compute_rows':len(compute),'heatmap_metrics':list(EXTERNAL)+['math_avg','ood_avg','external_micro']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment-dir',required=True);a=p.parse_args();print(json.dumps(summarize(a.experiment_dir),indent=2))
if __name__=='__main__':main()
