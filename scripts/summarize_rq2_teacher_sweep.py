#!/usr/bin/env python3
"""CPU-only table data for the RQ2.3 Teacher-strength panels.

Outputs performance, Teacher capability, per-round allocation dynamics, and
signed-mismatch distribution quantiles.  Fixed-prefix policy drift is a
separate one-GPU post-hoc task because it requires loading Student checkpoints.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np
from rq_metrics import MATH,OOD,grouped_metrics,with_deltas


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)


def qdict(values):
    qs=(.01,.05,.1,.25,.5,.75,.9,.95,.99)
    x=np.asarray(values,dtype=np.float64)
    vals=np.quantile(x,qs) if x.size else np.full(len(qs),np.nan)
    return {f'p{int(q*100):02d}':float(v) for q,v in zip(qs,vals)}


def summarize(root):
    root=Path(root);plan=json.loads((root/'experiment_plan.json').read_text());out=root/'analysis';out.mkdir(exist_ok=True)
    evaluation=json.loads((root/'evaluation/summary.json').read_text());base=grouped_metrics(evaluation['models']['base']);perf=[]
    for name,spec in plan['arms'].items():
        values=with_deltas(grouped_metrics(evaluation['models'][name]),base)
        perf.append({'arm':name,'teacher':spec['teacher'],'method':spec['reasoning'],**values})

    capability=[];cap_map={}
    path=root/'teacher_evaluation/summary.json'
    if path.exists():
        data=json.loads(path.read_text())
        for label in plan['teachers']:
            key='teacher_'+label
            if key not in data['models']:continue
            values=grouped_metrics(data['models'][key]);row={'teacher':label,**values};capability.append(row);cap_map[label]=row
        write_csv(out/'teacher_capability.csv',capability)
    for row in perf:
        cap=cap_map.get(row['teacher'])
        row['teacher_math_avg']=None if cap is None else cap['math_avg']
        row['teacher_math_gap']=None if cap is None or base['math_avg'] is None else cap['math_avg']-base['math_avg']
        row['student_math_gain']=row.get('math_avg_delta')
    write_csv(out/'performance.csv',perf)

    round_rows=[];dist_rows=[]
    for name,spec in plan['arms'].items():
        for r in range(4):
            diag_path=root/'arms'/name/'train/diagnostics'/f'round_{r:04d}'
            diag=json.loads((diag_path/'summary.json').read_text());metric=json.loads((root/'arms'/name/'train/metrics'/f'round_{r:04d}.json').read_text())[0]
            conc=diag['concentration']['bounded_weight'];ap=diag['concentration']['applied_weight']
            round_rows.append({'arm':name,'teacher':spec['teacher'],'method':spec['reasoning'],'round':r+1,
                'causal_kl_mean':diag['causal_kl_mean'],'hindsight_kl_mean':diag.get('hindsight_kl_mean'),
                'resolved_mean':diag['resolved_mean'],'resolved_positive_fraction':diag['resolved_positive_fraction'],
                'ren_weight_mean':diag['rho_mean'],'applied_weight_mean':diag['applied_weight_mean'],
                'ren_top1_mass':conc['top_fraction_mass'].get('0.01'),'ren_top10_mass':conc['top_fraction_mass']['0.1'],
                'applied_top10_mass':ap['top_fraction_mass']['0.1'],
                'retained_teacher_kl_prompt_balanced':diag.get('teacher_supervision_retained_fraction_prompt_balanced',diag.get('teacher_supervision_retained_fraction')),
                'retained_teacher_kl_token_global':diag.get('teacher_supervision_retained_fraction_token_global'),
                'prompt_loss_ess':diag['prompt_reasoning_loss_ess'],'prompt_loss_top10_share':diag.get('prompt_reasoning_loss_top10_share'),
                'round_seconds':metric.get('round_seconds'),'response_tokens':metric.get('response_tokens'),
                'teacher_sequence_tokens':metric.get('teacher_sequence_tokens'),'hindsight_sequence_tokens':metric.get('hindsight_sequence_tokens'),
                'teacher_positions':metric.get('teacher_positions'),'hindsight_positions':metric.get('hindsight_positions'),
                'teacher_seconds':metric.get('teacher_seconds'),'hindsight_seconds':metric.get('hindsight_seconds')})
            z=np.load(diag_path/'position_scores.npz');delta=z['resolved_mismatch'].astype(np.float64)
            dist_rows.append({'arm':name,'teacher':spec['teacher'],'method':spec['reasoning'],'round':r+1,
                              'positions':int(delta.size),'mean':float(delta.mean()) if delta.size else None,
                              'positive_fraction':float(np.mean(delta>0)) if delta.size else None,**qdict(delta)})
    write_csv(out/'round_dynamics.csv',round_rows);write_csv(out/'resolved_distribution_quantiles.csv',dist_rows)
    summary={'performance_rows':len(perf),'round_rows':len(round_rows),'distribution_rows':len(dist_rows),'teacher_capability_rows':len(capability)}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment-dir',required=True);a=p.parse_args();print(json.dumps(summarize(a.experiment_dir),indent=2))
if __name__=='__main__':main()
