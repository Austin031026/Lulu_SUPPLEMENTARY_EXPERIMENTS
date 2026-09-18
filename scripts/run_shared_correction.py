"""One fresh full-parameter shared-positive-correction run and automatic vLLM eval.

Reuse frozen baseline predictions; never start extra baseline training. Run a
matched 8k x 4 update pilot, retain round2, assess fixed round4 on external sets.
"""
from pathlib import Path
import argparse,fcntl,json,os,signal,sys
from run_decisive import WORKSPACE,freeze_code,digest,write_json,status,launch_detached
from run_gate_diagnostics import setarg,remove,verify,run_gpu_stage,command

SOURCE=WORKSPACE/'LuLu_outputs/experiments/ren_gate_diagnostics_8k_r4_s42_20260917'
DEFAULT=WORKSPACE/'LuLu_outputs/experiments/ren_shared_positive_full_qwen1p7_teacher32b_pool2048_8k_r4_s42_20260917'
ARM='D_shared'


def prepare(root):
    source=json.loads((SOURCE/'experiment_plan.json').read_text());code=root/'code'
    if json.loads((SOURCE/'live_progress.json').read_text())['status']!='complete':
        raise ValueError('Historical matched comparison must be complete')
    train=list(source['train_commands']['C_ren']);train[0]=sys.executable;train[2]=str(code/'scripts/train_lulu.py')
    for key,value in {'--method':'ren_shared','--output-dir':root/'arms'/ARM/'train',
        '--gpus':'0,1,2,3,4,5,6,7','--student-gpus':'0,1,2,3,4','--hindsight-gpus':'5','--teacher-gpus':'6,7',
        '--retain-checkpoints':'2,4'}.items():setarg(train,key,value)
    evaluate=list(source['eval_command']);evaluate[0]=sys.executable;evaluate[2]=str(code/'scripts/evaluate_lulu.py')
    remove(evaluate,'--checkpoint');evaluate+=['--checkpoint',f'{ARM}={root}/arms/{ARM}/train/checkpoints/round_000004']
    setarg(evaluate,'--parser-path',code/'lulu/thinking_final_parser.py');setarg(evaluate,'--output-dir',root/'evaluation')
    middle=list(evaluate);remove(middle,'--checkpoint')
    middle+=['--checkpoint',f'D_shared_round2={root}/arms/{ARM}/train/checkpoints/round_000002']
    setarg(middle,'--benchmarks','dapo_dev128');setarg(middle,'--output-dir',root/'intermediate_dev')
    files=[SOURCE/'experiment_plan.json',SOURCE/'comparison.json',SOURCE/'base_dev/eval_plan.json',SOURCE/'evaluation/eval_plan.json',
        Path(train[train.index('--train-data')+1]),Path(evaluate[evaluate.index('--data-manifest')+1])]
    manifest=json.loads(files[-1].read_text())
    for name in evaluate[evaluate.index('--benchmarks')+1].split(','):
        files.append(Path(manifest['benchmarks'][name]['full']))
    base=Path(source['base']);files+=sorted(base.glob('*.safetensors'))+[base/'config.json',base/'tokenizer_config.json']
    teacher=Path(train[train.index('--teacher-model')+1]);files += [teacher/'config.json',teacher/'model.safetensors.index.json']
    # Freeze generation metadata and existing scores used in the comparison.
    files+=sorted((SOURCE/'base_dev/base').glob('*/shard-*.jsonl'))
    files+=sorted((SOURCE/'evaluation').glob('*/*/shard-*.jsonl'))
    files+=sorted((Path(source['source_16k'])/'reference_32k/base').glob('*/rows.jsonl'))
    plan=dict(schema_version=1,arms={ARM:'shared_positive_probability_correction'},historical_experiment=str(SOURCE),
        train_command=train,eval_command=evaluate,intermediate_eval_command=middle,
        training=dict(student_workers=5,rounds=4,pool=2048,prompts_per_round=64,rollouts_per_prompt=1,
            horizon=8192,learning_rate=1e-6,full_parameter=True,update_passes=1,
            reference_coefficient=.1,control_coefficient=1,reasoning_coefficient=1,
            target='pC + [min(qT,pH)-pC]+ - (m/M)[pC-qT]+; M=0 => pC',
            reasoning_reduction='reasoning-token mean -> rollout mean -> prompt mean',
            scalar_gate=False,mass_renormalization=False,topk_target_approximation=False,
            checkpoint_retention=[0,2,4],gradient_norm_and_cosine_updates=[1,4],
            roles='five Student DDP/vLLM, one synchronized hindsight + fixed reference, two TP Teacher'),
        evaluation=dict(final_checkpoint=4,intermediate_dev_checkpoint=2,primary='dapo_dev128',
            external=['math500','aime25','olympiadbench','mmlu_pro','gpqa_diamond'],max_examples=199,
            horizon=32768,context_limit=40960,safety_margin=128,truncate_prompt=False,
            baseline_reuse_only=True,checkpoint_selection=False,
            baseline_caveat='Base external predictions reuse long responses clipped/rescored to 32768; not a fresh bitwise-equivalent generation'),
        resource_policy=dict(gpus=[str(i) for i in range(8)],idle_memory_mib=1024,idle_utilization=5,idle_checks=4,poll_seconds=15,evaluation_min_gpus=1),
        recovery_policy=dict(training_max_retries=2,only_explicit_nccl_transport_errors=True,optimizer_resume_from_committed_round=True),
        input_sha256={str(p):digest(p) for p in dict.fromkeys(files)})
    plan['code_sha256']=freeze_code(root);write_json(root/'experiment_plan.json',plan)
    (root/'README.md').write_text('''# Shared positive correction: 8k × 4 pilot

Fresh Base Qwen3-1.7B full-parameter Student, same-round privileged Student,
answer-blind Qwen3-32B Teacher. Fixed DAPO2048 pool, 64 prompts per round.
Only the new method is trained. Existing Base / A / B / C evaluations are reused.

The reasoning target is C + [min(T,H)-C]+ - (m/M)[C-T]+ (M=0 => C).
No scalar gate, target Top-K, correction-mass normalization or active-token normalization.
Reasoning is averaged over reasoning tokens, then rollouts, then prompts.
Control token mean and reference reverse KL coefficient 0.1 are unchanged.

Eight GPUs: five Student rollout/update, one H/reference, two Teacher TP.
Each role stays resident. Full-vocabulary targets are reconstructed in chunks.
Latest optimizer checkpoint is committed every update; round2 and round4 are retained.
Round4 is preselected for all external evaluation; round2 has dev128 only.
Automatic vLLM evaluation uses the previous common 32768 response budget,
40960 context with 128-token margin, unchanged prompt text, <=199 questions/set.
Round1/4 record exact full-batch component gradient norms and cosines.

`live_progress.json` tracks the controller, `arms/D_shared/train/phase_progress.json`
tracks training. `comparison.json` and `REPORT.md` appear only after evaluation.
''')
    return plan


def execute(root,plan):
    status(root,'verifying_frozen_inputs');verify(root,plan)
    env=dict(os.environ,HF_HUB_OFFLINE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',
        VLLM_DISABLE_COMPILE_CACHE='1',HF_HOME=str(WORKSPACE.parent/'huggingface_cache'),
        HF_HUB_CACHE=str(WORKSPACE.parent/'huggingface_cache/hub'),HUGGINGFACE_HUB_CACHE=str(WORKSPACE.parent/'huggingface_cache/hub'))
    env.pop('TRANSFORMERS_CACHE',None);env.setdefault('NCCL_DEBUG','INFO')
    done=root/'arms'/ARM/'train/checkpoints/round_000004/lulu_state.json'
    run_gpu_stage(root,plan,plan['train_command'],'training_'+ARM,done,env)
    state=json.loads(done.read_text())
    if state['completed_rounds']!=4 or state['completed_updates']!=4:raise RuntimeError('Incomplete new-method training')
    verify(root,plan)
    # Finish primary/external final checkpoint before optional intermediate dev.
    run_gpu_stage(root,plan,plan['eval_command'],'evaluation_final',root/'evaluation/summary.json',env)
    run_gpu_stage(root,plan,plan['intermediate_eval_command'],'evaluation_round2_dev',root/'intermediate_dev/summary.json',env)
    command(root,plan,[sys.executable,str(root/'code/scripts/summarize_shared_correction.py'),'--experiment-dir',str(root)],'reporting',env)
    status(root,'complete',arms={ARM:4},report=str(root/'REPORT.md'),comparison=str(root/'comparison.json'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',default=str(DEFAULT));p.add_argument('--run',action='store_true');p.add_argument('--detach',action='store_true');a=p.parse_args()
    root=Path(a.output_dir).resolve();root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    if a.detach:
        if not a.run:raise ValueError('--detach requires --run')
        launch_detached([sys.executable,'-u',str(Path(__file__).resolve()),'--output-dir',str(root),'--run'],root);return
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Controller signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    with (root/'.controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            plan=json.loads((root/'experiment_plan.json').read_text()) if (root/'experiment_plan.json').exists() else prepare(root)
            if a.run:execute(root,plan)
            else:status(root,'prepared',gpu_jobs_launched=False)
        except BaseException as exc:status(root,'failed',error=str(exc));raise
if __name__=='__main__':main()
