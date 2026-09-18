#!/usr/bin/env python3
"""RQ2.1/2.2 matched allocation controls.

Runs Uniform-Matched, Shuffled-ReN and Causal-Matched on the successful ReN
recipe.  The source ReN checkpoint is evaluated alongside the new controls.
"""
from __future__ import annotations
import argparse,fcntl,json,os,sys
from pathlib import Path
from rq_common import load_source_plan,freeze_current_code,train_command,eval_command,atomic_json,checkpoint,lightweight_provenance,student_worker_count
from run_gate_diagnostics import run_gpu_stage
from run_decisive import status,launch_detached

ARMS={'uniform_matched':'uniform','shuffled_ren':'shuffled','causal_matched':'causal_matched'}


def build_plan(source_experiment,root):
    source,src=load_source_plan(source_experiment);root=Path(root).resolve();code=root/'code';root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    hashes=freeze_current_code(code);commands={}
    for name,arm in ARMS.items():
        commands[name]=train_command(src,code_root=code,output_dir=root/'arms'/name/'train',arm=arm,batch=256,rounds=4,retain='4',control=.5)
    source_ren=source/'arms'/'balanced_recipe'/'train/checkpoints/round_000004'
    if not source_ren.exists():
        source_train=Path(src['train_command'][src['train_command'].index('--output-dir')+1]);source_ren=source_train/'checkpoints/round_000004'
    cps=[('ren',source_ren)]+[(name,checkpoint(root,name,4)) for name in ARMS]
    evaluate=eval_command(src,code_root=code,output_dir=root/'evaluation',checkpoints=cps,include_base=True,
                          benchmarks='math500,aime25,olympiadbench,mmlu_pro,gpqa_diamond,dapo_dev128',max_examples=199)
    gpus=commands['uniform_matched'][commands['uniform_matched'].index('--gpus')+1].split(',')
    plan={'schema_version':2,'task':'RQ2 matched supervision allocation','source_experiment':str(source),'arms':ARMS,
          'train_commands':commands,'eval_command':evaluate,'source_ren_checkpoint':str(source_ren),
          'scientific_controls':{
              'uniform_matched':'same per-rollout ReN weight mass, spread uniformly over reasoning states',
              'shuffled_ren':'same exact within-rollout ReN weight multiset, deterministic position permutation',
              'causal_matched':'same exact weight multiset, positions ranked only by causal Teacher-Student KL'},
          'training':{'student_workers':student_worker_count(commands['uniform_matched']),'rounds':4,'batch':256,'control':.5,'reference':.1,'matched_causal':True},
          'resource_policy':{'gpus':gpus,'idle_memory_mib':1024,'idle_utilization':5,'idle_checks':2,'poll_seconds':10,'evaluation_min_gpus':len(gpus)},
          'recovery_policy':{'training_max_retries':2},'input_sha256':lightweight_provenance(source,src),'code_sha256':hashes}
    atomic_json(root/'experiment_plan.json',plan);return plan


def execute(root,plan):
    root=Path(root);env=dict(os.environ,HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VLLM_DISABLE_COMPILE_CACHE='1')
    for name in ARMS:
        run_gpu_stage(root,plan,plan['train_commands'][name],f'training_{name}',checkpoint(root,name,4)/'lulu_state.json',env)
    run_gpu_stage(root,plan,plan['eval_command'],'evaluation',root/'evaluation/summary.json',env)
    from run_gate_diagnostics import command
    source_train=Path(plan['source_ren_checkpoint']).parents[1]
    command(root,plan,[sys.executable,str(root/'code/scripts/analyze_rq2_geometry.py'),'--train-dir',str(source_train),'--output-dir',str(root/'rq2_geometry')],'analysis_geometry',env)
    command(root,plan,[sys.executable,str(root/'code/scripts/summarize_rq2_allocation.py'),'--experiment-dir',str(root)],'reporting',env)
    status(root,'complete',task=plan['task'],evaluation=str(root/'evaluation/summary.json'),geometry=str(root/'rq2_geometry'),analysis=str(root/'analysis'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-experiment',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--run',action='store_true');p.add_argument('--detach',action='store_true');a=p.parse_args();root=Path(a.output_dir).resolve()
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
