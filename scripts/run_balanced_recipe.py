"""Matched causal path, larger on-policy batch, weaker control; four full updates."""
from pathlib import Path
import argparse,fcntl,json,os,signal,sys,shutil
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_decisive import WORKSPACE,freeze_code,digest,write_json,status,launch_detached
from run_gate_diagnostics import setarg,remove,verify,run_gpu_stage,command

SOURCE=WORKSPACE/'LuLu_outputs/experiments/ren_shared_positive_full_qwen1p7_teacher32b_pool2048_8k_r4_s42_20260917'
DEFAULT=WORKSPACE/'LuLu_outputs/experiments/ren_balanced_matched_b256_c0p5_8k_r4_s42_20260918'
ARM='balanced_recipe'


def prepare(root,batch,control):
    prior=json.loads((SOURCE/'experiment_plan.json').read_text());code=root/'code'
    train=list(prior['train_command']);train[0]=sys.executable;train[2]=str(code/'scripts/train_lulu.py')
    for key,value in {'--method':'ren_balanced','--output-dir':root/'arms'/ARM/'train',
        '--global-batch-prompts':batch,'--control-loss-coef':control,'--retain-checkpoints':'2,4',
        '--train-micro-batch-size':1,'--reasoning-diagnostic-split':0}.items():setarg(train,key,value)
    train.append('--match-causal-update')
    evaluate=list(prior['eval_command']);evaluate[0]=sys.executable;evaluate[2]=str(code/'scripts/evaluate_lulu.py')
    remove(evaluate,'--checkpoint')
    evaluate+=['--include-base','--checkpoint',f'round2={root}/arms/{ARM}/train/checkpoints/round_000002',
               '--checkpoint',f'round4={root}/arms/{ARM}/train/checkpoints/round_000004']
    setarg(evaluate,'--parser-path',code/'lulu/thinking_final_parser.py');setarg(evaluate,'--output-dir',root/'evaluation')
    # One matched 32k generation per checkpoint; CPU prefix evaluation supplies 8k.
    setarg(evaluate,'--max-response-tokens',32768)
    files=[SOURCE/'experiment_plan.json',Path(train[train.index('--train-data')+1]),Path(evaluate[evaluate.index('--data-manifest')+1])]
    manifest=json.loads(files[-1].read_text())
    files += [Path(manifest['benchmarks'][name]['full']) for name in evaluate[evaluate.index('--benchmarks')+1].split(',')]
    base=Path(train[train.index('--model')+1]);teacher=Path(train[train.index('--teacher-model')+1])
    files+=list(base.glob('*.safetensors'))+[base/'config.json',base/'tokenizer_config.json',teacher/'config.json',teacher/'model.safetensors.index.json']
    audit=root/'causal_path_audit';audit.mkdir(exist_ok=True)
    shutil.copyfile('/tmp/probe_lulu_causal_path.py',audit/'probe.py');shutil.copyfile('/tmp/lulu_causal_probe.json',audit/'results.json')
    observed=json.loads((audit/'results.json').read_text())
    fixed=[r for r in observed if r['path']=='eval_batch1']
    if len(fixed)<2 or any(r['tv']>1e-7 or abs(r['kl'])>1e-7 or r['noop_gradient_norm']>1e-4 for r in fixed):
        raise RuntimeError('Real Qwen causal-path gate did not pass')
    files += [audit/'probe.py',audit/'results.json']
    rows=[json.loads(s) for s in files[1].read_text().splitlines()]
    from lulu.training import schedule
    schedule_ids=[rows[i]['id'] for rnd in range(4) for i in schedule(len(rows),batch,rnd,42)]
    if len(set(schedule_ids))!=4*batch:raise ValueError('Expected new prompts in all four rounds')
    plan=dict(schema_version=1,arms={ARM:'trajectory_balanced_bounded_absolute_ReN'},train_command=train,eval_command=evaluate,
        historical_shared=str(SOURCE),training=dict(student_workers=5,rounds=4,prompts_per_round=batch,unique_prompt_exposures=4*batch,
            prompt_pool=2048,rollouts_per_prompt=1,horizon=8192,learning_rate=1e-6,full_parameter=True,update_passes=1,
            causal_batch=1,hindsight_teacher_batch=2,update_microbatch=1,logit_chunk=128,
            control_coefficient=control,reference_coefficient=.1,reasoning_coefficient=1,
            target='Full answer-blind Teacher, scalar detached w=[DC-DH]+/(1+[DC-DH]+)',
            weight_renormalization=False,reasoning_reduction='reasoning tokens -> rollout -> prompt',
            hindsight_prompt='Current, unchanged',checkpoint_retention=[0,2,4],gradient_diagnostic_steps=[1,4]),
        evaluation=dict(models=['base','round2','round4'],response_budget=32768,prefix_budget=8192,
            examples_per_dataset=199,dev_examples=128,final_only_parser=True,fresh_base=True,checkpoint_selection=False,
            note='32k fresh paired generation; 8k is a CPU truncation of these same trajectories, not a separate fresh 8k run'),
        causal_correctness=dict(real_model_audit='causal_path_audit/results.json',
            per_round_guard='abs(actual_reason_loss-scored_reason_loss) <= 1e-7 + 1e-4*abs(scored_reason_loss), before optimizer.step'),
        resource_policy=dict(gpus=[str(i) for i in range(8)],idle_memory_mib=1024,idle_utilization=5,idle_checks=2,poll_seconds=10,evaluation_min_gpus=8),
        recovery_policy=dict(training_max_retries=2,only_explicit_nccl_transport_errors=True,optimizer_resume_from_committed_round=True),
        input_sha256={str(p):digest(p) for p in dict.fromkeys(files)})
    write_json(root/'training_prompt_ids.json',schedule_ids)
    plan['input_sha256'][str(root/'training_prompt_ids.json')]=digest(root/'training_prompt_ids.json')
    plan['code_sha256']=freeze_code(root);write_json(root/'experiment_plan.json',plan)
    return plan


def execute(root,plan):
    status(root,'verifying_frozen_inputs');verify(root,plan)
    env=dict(os.environ,HF_HUB_OFFLINE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
        VLLM_DISABLE_COMPILE_CACHE='1',NCCL_DEBUG='WARN');env.pop('TRANSFORMERS_CACHE',None)
    done=root/'arms'/ARM/'train/checkpoints/round_000004/lulu_state.json'
    run_gpu_stage(root,plan,plan['train_command'],'training_'+ARM,done,env)
    state=json.loads(done.read_text())
    if state['completed_rounds']!=4 or state['completed_updates']!=4:raise RuntimeError('Incomplete training')
    verify(root,plan)
    run_gpu_stage(root,plan,plan['eval_command'],'evaluation_base_round2_round4',root/'evaluation/summary.json',env)
    command(root,plan,[sys.executable,str(root/'code/scripts/summarize_balanced_recipe.py'),'--experiment-dir',str(root)],'reporting',env)
    status(root,'complete',arms={ARM:4},report=str(root/'REPORT.md'),comparison=str(root/'comparison.json'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',default=str(DEFAULT));p.add_argument('--batch',type=int,default=256)
    p.add_argument('--control',type=float,default=.5);p.add_argument('--run',action='store_true');p.add_argument('--detach',action='store_true');a=p.parse_args()
    root=Path(a.output_dir).resolve();root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    if a.detach:
        if not a.run:raise ValueError('--detach requires --run')
        launch_detached([sys.executable,'-u',str(Path(__file__).resolve()),'--output-dir',str(root),'--batch',str(a.batch),'--control',str(a.control),'--run'],root);return
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Controller signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    with (root/'.controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            plan=json.loads((root/'experiment_plan.json').read_text()) if (root/'experiment_plan.json').exists() else prepare(root,a.batch,a.control)
            if a.run:execute(root,plan)
            else:status(root,'prepared',gpu_jobs_launched=False)
        except BaseException as exc:status(root,'failed',error=str(exc));raise
if __name__=='__main__':main()
