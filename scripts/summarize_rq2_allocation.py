#!/usr/bin/env python3
"""CPU-only performance and allocation summary for RQ2.1/RQ2.2 controls."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from rq_metrics import grouped_metrics,with_deltas

ORDER=("base","uniform_matched","shuffled_ren","causal_matched","ren")


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)


def _diag_rows(train_dir,arm):
    rows=[]
    for path in sorted(Path(train_dir).glob('diagnostics/round_*/summary.json')):
        d=json.loads(path.read_text());c=d['concentration']['applied_weight']
        rows.append({
            'arm':arm,'round':int(d['round'])+1,
            'applied_weight_mean':d.get('applied_weight_mean'),
            'applied_weight_positive_fraction':c.get('positive_fraction'),
            'applied_top1_mass':c.get('top_fraction_mass',{}).get('0.01'),
            'applied_top10_mass':c.get('top_fraction_mass',{}).get('0.1'),
            'token_weight_ess':c.get('effective_positions'),
            'prompt_loss_ess':d.get('prompt_reasoning_loss_ess'),
            'prompt_loss_top10_share':d.get('prompt_reasoning_loss_top10_share'),
            'retained_teacher_kl_prompt_balanced':d.get('teacher_supervision_retained_fraction_prompt_balanced',d.get('teacher_supervision_retained_fraction')),
            'retained_teacher_kl_token_global':d.get('teacher_supervision_retained_fraction_token_global'),
            'causal_kl_mean':d.get('causal_kl_mean'),'resolved_mean':d.get('resolved_mean'),
            'resolved_positive_fraction':d.get('resolved_positive_fraction'),
        })
    return rows


def summarize(root):
    root=Path(root);plan=json.loads((root/'experiment_plan.json').read_text());out=root/'analysis';out.mkdir(exist_ok=True)
    evaluation=json.loads((root/'evaluation/summary.json').read_text());models=evaluation['models'];base=grouped_metrics(models['base']);perf=[]
    for name in ORDER:
        if name not in models:continue
        perf.append({'model':name,**with_deltas(grouped_metrics(models[name]),base)})
    write_csv(out/'performance.csv',perf)
    dynamics=[]
    for name in plan['arms']:
        dynamics += _diag_rows(root/'arms'/name/'train',name)
    # The source ReN was not retrained inside this experiment; include its saved
    # round diagnostics so plots/tables can compare allocation statistics directly.
    source_train=Path(plan['source_ren_checkpoint']).parents[1]
    dynamics += _diag_rows(source_train,'ren')
    write_csv(out/'allocation_dynamics.csv',dynamics)
    summary={'performance_rows':len(perf),'allocation_rows':len(dynamics),'models':[x['model'] for x in perf]}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment-dir',required=True);a=p.parse_args();print(json.dumps(summarize(a.experiment_dir),indent=2))
if __name__=='__main__':main()
