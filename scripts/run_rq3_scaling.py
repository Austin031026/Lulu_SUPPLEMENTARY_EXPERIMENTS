#!/usr/bin/env python3
"""RQ3.1 on-policy supervision scaling with fixed four optimizer updates.

Default budgets are 256/512/1024/2048 unique prompts, implemented as
64/128/256/512 prompts per round for four rounds.  Every round checkpoint is
retained.  A single dev evaluation covers all 16 checkpoints; a final external
evaluation covers the four Round4 models.
"""
from __future__ import annotations
import argparse,fcntl,json,os,sys
from pathlib import Path
from rq_common import load_source_plan,freeze_current_code,train_command,eval_command,atomic_json,checkpoint,lightweight_provenance,student_worker_count
from run_gate_diagnostics import run_gpu_stage
from run_decisive import status,launch_detached


def build_plan(source_experiment,root,batches=(64,128,256,512),include_vanilla=False):
    source,src=load_source_plan(source_experiment);root=Path(root).resolve();code=root/'code';root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    hashes=freeze_current_code(code);commands={};matrix=[]
    for batch in batches:
        budget=4*batch
        for arm in (('ren','vanilla') if include_vanilla else ('ren',)):
            name=f'{arm}_n{budget}'
            commands[name]=train_command(src,code_root=code,output_dir=root/'arms'/name/'train',arm=arm,batch=batch,rounds=4,retain='1,2,3,4',control=.5)
            matrix.append({'name':name,'arm':arm,'batch':batch,'rounds':4,'unique_prompts':budget})
    dev_checkpoints=[]
    final_checkpoints=[]
    for row in matrix:
        for r in range(1,5):dev_checkpoints.append((f"{row['name']}_r{r}",checkpoint(root,row['name'],r)))
        final_checkpoints.append((row['name'],checkpoint(root,row['name'],4)))
    dev_eval=eval_command(src,code_root=code,output_dir=root/'evaluation_dev',checkpoints=dev_checkpoints,include_base=True,
                          benchmarks='dapo_dev128',max_examples=128)
    final_eval=eval_command(src,code_root=code,output_dir=root/'evaluation_final',checkpoints=final_checkpoints,include_base=True,
                            benchmarks='math500,aime25,olympiadbench,mmlu_pro,gpqa_diamond,dapo_dev128',max_examples=199)
    gpus=next(iter(commands.values()));gpus=gpus[gpus.index('--gpus')+1].split(',')
    plan={'schema_version':2,'task':'RQ3.1 on-policy supervision scaling','source_experiment':str(source),'matrix':matrix,
          'train_commands':commands,'dev_eval_command':dev_eval,'final_eval_command':final_eval,
          'scientific_control':'four optimizer updates for every budget; same shuffled pool prefix so prompt sets are nested when the source scheduler is unchanged',
          'training':{'student_workers':student_worker_count(next(iter(commands.values()))),'rounds':4,'control':.5,'reference':.1,'retained_rounds':[0,1,2,3,4]},
          'rq_outputs':{
              'panel_a':'held-out dev versus cumulative unique on-policy prompts for every round/checkpoint',
              'panel_b':'benchmark x final supervision-budget gain heatmap',
              'panel_c':'effective supervision coverage: token-weight ESS, prompt-loss ESS, top-10% mass',
              'panel_d':'performance-compute frontier using Teacher-scored sequence tokens and wall-clock'},
          'resource_policy':{'gpus':gpus,'idle_memory_mib':1024,'idle_utilization':5,'idle_checks':2,'poll_seconds':10,'evaluation_min_gpus':len(gpus)},
          'recovery_policy':{'training_max_retries':2},'input_sha256':lightweight_provenance(source,src),'code_sha256':hashes}
    atomic_json(root/'experiment_plan.json',plan);return plan


def execute(root,plan):
    root=Path(root);env=dict(os.environ,HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VLLM_DISABLE_COMPILE_CACHE='1')
    for row in plan['matrix']:
        name=row['name'];run_gpu_stage(root,plan,plan['train_commands'][name],f'training_{name}',checkpoint(root,name,4)/'lulu_state.json',env)
    run_gpu_stage(root,plan,plan['dev_eval_command'],'evaluation_dev',root/'evaluation_dev/summary.json',env)
    run_gpu_stage(root,plan,plan['final_eval_command'],'evaluation_final',root/'evaluation_final/summary.json',env)
    from run_gate_diagnostics import command
    command(root,plan,[sys.executable,str(root/'code/scripts/summarize_rq3_scaling.py'),'--experiment-dir',str(root)],'reporting',env)
    status(root,'complete',task=plan['task'],dev=str(root/'evaluation_dev/summary.json'),final=str(root/'evaluation_final/summary.json'),analysis=str(root/'analysis'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-experiment',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--batches',default='64,128,256,512');p.add_argument('--include-vanilla',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--detach',action='store_true');a=p.parse_args();root=Path(a.output_dir).resolve()
    batches=tuple(int(x) for x in a.batches.split(',') if x.strip())
    if any(x<1 for x in batches) or len(set(batches))!=len(batches):raise ValueError('batches must be distinct positive integers')
    if a.detach:
        if not a.run:raise ValueError('--detach requires --run')
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--source-experiment',a.source_experiment,'--output-dir',str(root),'--batches',a.batches,'--run']
        if a.include_vanilla:cmd.append('--include-vanilla')
        launch_detached(cmd,root);return
    root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    with (root/'.controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=json.loads((root/'experiment_plan.json').read_text()) if (root/'experiment_plan.json').exists() else build_plan(a.source_experiment,root,batches,a.include_vanilla)
        if a.run:execute(root,plan)
        else:status(root,'prepared',task=plan['task'],gpu_jobs_launched=False)
if __name__=='__main__':main()
