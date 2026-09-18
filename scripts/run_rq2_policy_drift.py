#!/usr/bin/env python3
"""Build/run the fixed-prefix policy-drift audit for an RQ2.3 Teacher sweep.

The audit reuses one frozen round-0 Student rollout set and retained Student
checkpoints.  It performs teacher forcing only; it does not generate new text.
"""
from __future__ import annotations
import argparse,json,subprocess,sys
from pathlib import Path


def build_command(experiment_dir,gpu='0',max_trajectories=16,positions_per_bin=8):
    root=Path(experiment_dir).resolve();plan=json.loads((root/'experiment_plan.json').read_text())
    arms=sorted(plan['arms'])
    if not arms:raise ValueError('Teacher-sweep plan has no arms')
    # Prefer a ReN arm as the frozen-prefix source.  All arms start from the same
    # Student; selecting one source avoids comparing drift on different prefixes.
    ren=[x for x in arms if plan['arms'][x].get('reasoning')=='ren'];anchor=(ren or arms)[0]
    train=root/'arms'/anchor/'train';base=train/'checkpoints/round_000000'
    rollouts=sorted((train/'rollouts/round_0000').glob('shard-*.jsonl'))
    if not rollouts:raise FileNotFoundError(train/'rollouts/round_0000')
    script=(root/'code/scripts/audit_policy_drift.py') if (root/'code/scripts/audit_policy_drift.py').is_file() else Path(__file__).resolve().with_name('audit_policy_drift.py')
    cmd=[sys.executable,'-u',str(script),'--base-checkpoint',str(base),'--output-dir',str(root/'analysis/policy_drift'),
         '--gpu',str(gpu),'--max-trajectories',str(max_trajectories),'--positions-per-bin',str(positions_per_bin)]
    for path in rollouts:cmd += ['--rollout',str(path)]
    for arm in arms:
        for r in range(1,5):
            path=root/'arms'/arm/'train/checkpoints'/f'round_{r:06d}'
            if not path.is_dir():raise FileNotFoundError(path)
            cmd += ['--checkpoint',f'{arm}_r{r}={path}']
    return cmd,anchor


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment-dir',required=True);p.add_argument('--gpu',default='0')
    p.add_argument('--max-trajectories',type=int,default=16);p.add_argument('--positions-per-bin',type=int,default=8);p.add_argument('--run',action='store_true');a=p.parse_args()
    cmd,anchor=build_command(a.experiment_dir,a.gpu,a.max_trajectories,a.positions_per_bin)
    payload={'anchor_arm':anchor,'command':cmd,'generates_new_text':False}
    print(json.dumps(payload,indent=2))
    if a.run:subprocess.run(cmd,check=True)
if __name__=='__main__':main()
