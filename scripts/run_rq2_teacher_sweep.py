#!/usr/bin/env python3
"""RQ2.3 Teacher-strength sweep for a fixed Student.

Example:
  python scripts/run_rq2_teacher_sweep.py \
    --source-experiment /.../ren_balanced_matched_b256... \
    --output-dir /.../rq2_teacher_sweep \
    --teacher qwen4b=/models/Qwen3-4B --teacher qwen8b=/models/Qwen3-8B \
    --teacher qwen14b=/models/Qwen3-14B --teacher qwen32b=/models/Qwen3-32B

Each Teacher gets matched Vanilla and ReN training.  Rounds 1--4 are retained so
policy drift can be audited on fixed prefixes after the run.  Optional Teacher
standalone evaluation measures the actual capability gap rather than using
parameter count as a proxy.
"""
from __future__ import annotations
import argparse,fcntl,json,os,re,sys
from pathlib import Path
from rq_common import load_source_plan,freeze_current_code,train_command,eval_command,atomic_json,checkpoint,lightweight_provenance,student_worker_count
from run_gate_diagnostics import run_gpu_stage,command
from run_decisive import status,launch_detached


def named(value):
    name,sep,path=value.partition('=')
    if not sep or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',name) or not path:raise ValueError(f'expected NAME=PATH, got {value!r}')
    return name,str(Path(path).expanduser().resolve())


def build_plan(source_experiment,root,teachers,teacher_eval=False):
    source,src=load_source_plan(source_experiment);root=Path(root).resolve();code=root/'code';root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    hashes=freeze_current_code(code);commands={};teacher_map=dict(teachers)
    if len(teacher_map)<2:raise ValueError('Teacher-strength sweep needs at least two distinct Teachers')
    for label,path in teacher_map.items():
        for arm in ('vanilla','ren'):
            name=f'{label}_{arm}'
            commands[name]=train_command(src,code_root=code,output_dir=root/'arms'/name/'train',arm=arm,batch=256,rounds=4,
                                         teacher_model=path,retain='1,2,3,4',control=.5)
    finals=[(name,checkpoint(root,name,4)) for name in commands]
    evaluate=eval_command(src,code_root=code,output_dir=root/'evaluation',checkpoints=finals,include_base=True,
                          benchmarks='math500,aime25,olympiadbench,mmlu_pro,gpqa_diamond,dapo_dev128',max_examples=199)
    teacher_evaluate=eval_command(src,code_root=code,output_dir=root/'teacher_evaluation',
                                  checkpoints=[(f'teacher_{name}',path) for name,path in teacher_map.items()],
                                  include_base=False,benchmarks='math500,aime25,olympiadbench,mmlu_pro,gpqa_diamond',max_examples=199)
    # Large Teacher copies need a conservative vLLM scheduler for 32k generation.
    from rq_common import setarg
    teacher_evaluate=setarg(teacher_evaluate,'--vllm-max-num-seqs',4)
    teacher_evaluate=setarg(teacher_evaluate,'--vllm-gpu-memory-utilization',.94)
    gpus=next(iter(commands.values()));gpus=gpus[gpus.index('--gpus')+1].split(',')
    plan={'schema_version':2,'task':'RQ2.3 Teacher-strength sweep','source_experiment':str(source),'teachers':teacher_map,
          'arms':{name:{'teacher':label,'reasoning':arm} for label in teacher_map for arm,name in [(arm,f'{label}_{arm}') for arm in ('vanilla','ren')]},
          'train_commands':commands,'eval_command':evaluate,'teacher_eval_command':teacher_evaluate,'run_teacher_eval':bool(teacher_eval),
          'training':{'student_workers':student_worker_count(next(iter(commands.values()))),'rounds':4,'batch':256,'control':.5,'reference':.1,'retained_rounds':[0,1,2,3,4],
                      'diagnostics':'per-round position_scores, applied/raw weight concentration, position bins, scoring work and checkpoints'},
          'rq_outputs':{
              'panel_a':'Student post-training performance versus measured Teacher capability gap',
              'panel_b':'per-Teacher signed resolved-mismatch distributions from position_scores.npz',
              'panel_c':'per-round fraction of raw Teacher KL mass retained by applied weights',
              'panel_d':'fixed-prefix Student policy drift from retained round0--4 checkpoints'},
          'resource_policy':{'gpus':gpus,'idle_memory_mib':1024,'idle_utilization':5,'idle_checks':2,'poll_seconds':10,'evaluation_min_gpus':len(gpus)},
          'recovery_policy':{'training_max_retries':2},'input_sha256':lightweight_provenance(source,src,teacher_map.values()),'code_sha256':hashes}
    atomic_json(root/'experiment_plan.json',plan);return plan


def execute(root,plan):
    root=Path(root);env=dict(os.environ,HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VLLM_DISABLE_COMPILE_CACHE='1')
    for name in plan['train_commands']:
        run_gpu_stage(root,plan,plan['train_commands'][name],f'training_{name}',checkpoint(root,name,4)/'lulu_state.json',env)
    run_gpu_stage(root,plan,plan['eval_command'],'evaluation_students',root/'evaluation/summary.json',env)
    if plan.get('run_teacher_eval'):
        run_gpu_stage(root,plan,plan['teacher_eval_command'],'evaluation_teachers',root/'teacher_evaluation/summary.json',env)
    command(root,plan,[sys.executable,str(root/'code/scripts/summarize_rq2_teacher_sweep.py'),'--experiment-dir',str(root)],'reporting',env)
    status(root,'complete',task=plan['task'],evaluation=str(root/'evaluation/summary.json'),teacher_evaluation=plan.get('run_teacher_eval'),analysis=str(root/'analysis'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-experiment',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--teacher',action='append',default=[],metavar='NAME=PATH')
    p.add_argument('--teacher-eval',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--detach',action='store_true');a=p.parse_args();root=Path(a.output_dir).resolve();teachers=[named(x) for x in a.teacher]
    if a.detach:
        if not a.run:raise ValueError('--detach requires --run')
        cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--source-experiment',a.source_experiment,'--output-dir',str(root),'--run']
        for item in a.teacher:cmd+=['--teacher',item]
        if a.teacher_eval:cmd.append('--teacher-eval')
        launch_detached(cmd,root);return
    root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    with (root/'.controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=json.loads((root/'experiment_plan.json').read_text()) if (root/'experiment_plan.json').exists() else build_plan(a.source_experiment,root,teachers,a.teacher_eval)
        if a.run:execute(root,plan)
        else:status(root,'prepared',task=plan['task'],gpu_jobs_launched=False)
if __name__=='__main__':main()
