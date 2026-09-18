#!/usr/bin/env python3
"""CPU-only table-ready summary for RQ1 matched baselines."""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from rq_metrics import grouped_metrics,with_deltas

ORDER=("base","control_ref","vanilla_opd","opsd","ren")


def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)


def summarize(root):
    root=Path(root);out=root/'analysis';out.mkdir(exist_ok=True)
    data=json.loads((root/'evaluation/summary.json').read_text())
    models=data['models'];base=grouped_metrics(models['base']);rows=[]
    for name in ORDER:
        if name not in models:continue
        metrics=grouped_metrics(models[name]);values=with_deltas(metrics,base)
        rows.append({'model':name,**values})
    write_csv(out/'main_results.csv',rows)
    summary={'models':[r['model'] for r in rows],'rows':len(rows),'external_examples':base['external_examples']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment-dir',required=True);a=p.parse_args()
    print(json.dumps(summarize(a.experiment_dir),indent=2))
if __name__=='__main__':main()
