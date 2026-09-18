#!/usr/bin/env python3
"""Matched RQ1 baselines on the frozen successful ReN recipe.

Prepares (default) or runs Control/Reference-only, Vanilla OPD and matched OPSD.
The source ReN Round4 checkpoint is evaluated in the same fresh evaluation job.
"""
from __future__ import annotations
import argparse, fcntl, json, os, signal, sys
from pathlib import Path

from rq_common import load_source_plan, freeze_current_code, train_command, eval_command, atomic_json, checkpoint, lightweight_provenance, student_worker_count
from run_gate_diagnostics import run_gpu_stage, command
from run_decisive import status, launch_detached

ARMS={'control_ref':'none','vanilla_opd':'vanilla','opsd':'opsd'}


def _gpus(train):
    return train[train.index('--gpus')+1].split(',')


def build_plan(source_experiment, root):
    source,src=load_source_plan(source_experiment);root=Path(root).resolve();code=root/'code'
    root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    hashes=freeze_current_code(code)
    commands={}
    for name,arm in ARMS.items():
        commands[name]=train_command(src,code_root=code,output_dir=root/'arms'/name/'train',arm=arm,batch=256,rounds=4,retain='4',control=.5)
    source_ren=source/'arms'/'balanced_recipe'/'train'/'checkpoints'/'round_000004'
    if not source_ren.exists():
        # Generic fallback for a source plan whose train output is stored directly.
        source_train=Path(src['train_command'][src['train_command'].index('--output-dir')+1])
        source_ren=source_train/'checkpoints'/'round_000004'
    cps=[('ren',source_ren)]+[(name,checkpoint(root,name,4)) for name in ARMS]
    evaluate=eval_command(src,code_root=code,output_dir=root/'evaluation',checkpoints=cps,include_base=True,
                          benchmarks='math500,aime25,olympiadbench,mmlu_pro,gpqa_diamond,dapo_dev128',max_examples=199)
    gpus=_gpus(commands['control_ref'])
    plan={'schema_version':2,'task':'RQ1 matched baselines','source_experiment':str(source),'arms':ARMS,
          'train_commands':commands,'eval_command':evaluate,'source_ren_checkpoint':str(source_ren),
          'training':{'student_workers':student_worker_count(commands['control_ref']),'rounds':4,'batch':256,'control':.5,'reference':.1,
                      'matched_causal':True,'update_passes':1},
          'resource_policy':{'gpus':gpus,'idle_memory_mib':1024,'idle_utilization':5,'idle_checks':2,'poll_seconds':10,
                             'evaluation_min_gpus':len(gpus)},
          'recovery_policy':{'training_max_retries':2},'input_sha256':lightweight_provenance(source,src),'code_sha256':hashes}
    atomic_json(root/'experiment_plan.json',plan);return plan


def execute(root,plan):
    root=Path(root);env=dict(os.environ,HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VLLM_DISABLE_COMPILE_CACHE='1')
    for name in ARMS:
        done=checkpoint(root,name,4)/'lulu_state.json'
        run_gpu_stage(root,plan,plan['train_commands'][name],f'training_{name}',done,env)
    run_gpu_stage(root,plan,plan['eval_command'],'evaluation',root/'evaluation/summary.json',env)
    command(root,plan,[sys.executable,str(root/'code/scripts/summarize_rq1_baselines.py'),'--experiment-dir',str(root)],'reporting',env)
    status(root,'complete',task=plan['task'],evaluation=str(root/'evaluation/summary.json'),analysis=str(root/'analysis'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-experiment',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--run',action='store_true');p.add_argument('--detach',action='store_true');a=p.parse_args();root=Path(a.output_dir).resolve()
    if a.detach:
        if not a.run:raise ValueError('--detach requires --run')
        launch_detached([sys.executable,'-u',str(Path(__file__).resolve()),'--source-experiment',a.source_experiment,'--output-dir',str(root),'--run'],root);return
    root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    with (root/'.controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=json.loads((root/'experiment_plan.json').read_text()) if (root/'experiment_plan.json').exists() else build_plan(a.source_experiment,root)
        if a.run:execute(root,plan)
        else:status(root,'prepared',task=plan['task'],gpu_jobs_launched=False)
if __name__=='__main__':main()
