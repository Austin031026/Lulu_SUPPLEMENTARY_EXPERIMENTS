"""Qwen3-4B replication of the matched b256/c0.5 recipe, with memory-only adaptations."""
from pathlib import Path
import argparse,fcntl,json,os,signal,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run_decisive import WORKSPACE,freeze_code,digest,write_json,status,launch_detached
from run_gate_diagnostics import setarg,remove
from run_balanced_recipe import execute,ARM

SOURCE=WORKSPACE/'LuLu_outputs/experiments/ren_balanced_matched_b256_c0p5_8k_r4_s42_20260918'
DEFAULT=WORKSPACE/'LuLu_outputs/experiments/ren_balanced_matched_qwen4b_b256_c0p5_8k_r4_s42_20260918'
MODEL=WORKSPACE.parent/'huggingface_cache/transformers/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c'


def prepare(root,model):
    from transformers import AutoTokenizer
    from lulu.training import check_vocab,load_prepared_jsonl,schedule,parser,validate_args
    prior=json.loads((SOURCE/'experiment_plan.json').read_text());code=root/'code'
    model=Path(model).resolve();teacher=Path(prior['train_command'][prior['train_command'].index('--teacher-model')+1])
    config=json.loads((model/'config.json').read_text());tc=json.loads((teacher/'config.json').read_text())
    tok=AutoTokenizer.from_pretrained(model,local_files_only=True);tt=AutoTokenizer.from_pretrained(teacher,local_files_only=True)
    check_vocab(tok,tt,config['vocab_size'],tc['vocab_size'])
    if config['model_type']!='qwen3' or config['hidden_size']!=2560 or config['max_position_embeddings']<40960:
        raise ValueError('Expected the original Qwen3-4B thinking model with the 40960 context')
    probes=json.loads((root/'causal_path_audit/results.json').read_text());sleep=json.loads((root/'sleep_probe/sleep_results.json').read_text())
    if len(probes)<2 or any(Path(v['model'])!=model or v['tv']>1e-7 or abs(v['kl'])>1e-7 or v['noop_gradient_norm']>1e-4 for v in probes):
        raise RuntimeError('The real 4B causal/live audit must pass before preparing the run')
    if sleep['status']!='passed' or Path(sleep['model'])!=model:raise RuntimeError('The real 4B sleep/refresh audit must pass')
    train=list(prior['train_command']);train[0]=sys.executable;train[2]=str(code/'scripts/train_lulu.py')
    for flag,value in {'--model':model,'--output-dir':root/'arms'/ARM/'train','--rollout-vllm-memory':.75,'--rollout-vllm-max-seqs':16}.items():setarg(train,flag,value)
    train+=['--optimizer-state-sharding','--rollout-vllm-sleep']
    validate_args(parser().parse_args(train[3:]))
    evaluate=list(prior['eval_command']);evaluate[0]=sys.executable;evaluate[2]=str(code/'scripts/evaluate_lulu.py');remove(evaluate,'--checkpoint')
    for flag,value in {'--model':model,'--output-dir':root/'evaluation','--parser-path':code/'lulu/thinking_final_parser.py'}.items():setarg(evaluate,flag,value)
    evaluate+=['--checkpoint',f'round2={root}/arms/{ARM}/train/checkpoints/round_000002','--checkpoint',f'round4={root}/arms/{ARM}/train/checkpoints/round_000004']
    data=Path(train[train.index('--train-data')+1]);rows=load_prepared_jsonl(data)
    ids=[rows[i]['id'] for rnd in range(4) for i in schedule(len(rows),256,rnd,42)]
    if len(set(ids))!=1024 or ids!=json.loads((SOURCE/'training_prompt_ids.json').read_text()):raise ValueError('4B and 1.7B must use the identical 1024 prompt schedule')
    write_json(root/'training_prompt_ids.json',ids)
    write_json(root/'model_checks.json',dict(student=str(model),teacher=str(teacher),matching_token_ids=True,
        vocabulary_size=config['vocab_size'],context=config['max_position_embeddings'],hindsight='same-round 4B snapshot',reference='initial 4B',
        actual_model_parameters='recorded by runtime_plan.json after model load',probe_scope='Two fixed existing 1.7B rollout prefixes used only for numerical preflight; training always collects fresh 4B on-policy rollouts'))
    manifest=Path(evaluate[evaluate.index('--data-manifest')+1]);sources=json.loads(manifest.read_text())['benchmarks']
    files=[SOURCE/'experiment_plan.json',SOURCE/'training_prompt_ids.json',data,manifest,root/'training_prompt_ids.json',root/'model_checks.json',
        root/'causal_path_audit/results.json',root/'sleep_probe/sleep_results.json',Path(__file__).with_name('probe_balanced_student.py'),
        SOURCE/'arms/balanced_recipe/train/rollouts/round_0000/shard-000.jsonl',teacher/'config.json',teacher/'model.safetensors.index.json']
    files += [Path(sources[b]['full']) for b in evaluate[evaluate.index('--benchmarks')+1].split(',')]
    files += [p for p in sorted(model.iterdir()) if p.is_file()]
    plan={k:v for k,v in prior.items() if k not in ('input_sha256','code_sha256','historical_shared','reporting_amendments')}
    plan.update(train_command=train,eval_command=evaluate,source_recipe=str(SOURCE),student_model=str(model),
        scope='One 4B full-parameter run from Base; same prompts, objective and four updates as the 1.7B matched recipe; fresh 4B Base/Round2/Round4 evaluation',
        input_sha256={str(p):digest(p) for p in dict.fromkeys(files)})
    plan['training']=dict(prior['training'],student_model='Qwen3-4B',hindsight_model='same-round Qwen3-4B',reference_model='initial Qwen3-4B',
        optimizer_state_sharding=True,optimizer='AdamW, ZeroRedundancyOptimizer state partitions, foreach=False; no gradient/parameter sharding',
        rollout_vllm_sleep=True,rollout_vllm_memory=.75,rollout_vllm_max_seqs=16)
    plan['memory_adaptations']=dict(unchanged_science=['lr=1e-6','batch256','rounds4','full FP32 master parameters','one update/round',
        'horizon8192','current hindsight prompt','control0.5','reference0.1','rollout sampling','same 1024 training prompts','same evaluation protocol'],
        changes=['Shard only AdamW moments across five Student DDP ranks; consolidate full optimizer on checkpoint',
        'vLLM sleep(level=1) after collection, wake and refresh at the next round','rollout vLLM memory 0.42 -> 0.75 (vLLM V1 whole-device accounting includes the co-resident Student), max sequences 32 -> 16'])
    plan['code_sha256']=freeze_code(root);write_json(root/'experiment_plan.json',plan)
    (root/'README.md').write_text('# Qwen3-4B matched bounded absolute ReN\n\nOne full-parameter run: same fixed 2048 pool and 1024 prompt schedule, 256 × 4 rounds, 8192 training response tokens, LR1e-6, control0.5, reference0.1, fixed Qwen3-32B Teacher. Hindsight and reference are 4B.\n\nMemory adaptations only: AdamW optimizer-state sharding across five Student ranks; resident vLLM sleep between collection/update; rollout memory0.75/maxseq16 (includes co-resident HF allocations in vLLM V1 profiling). The objective and full-parameter gradients are unchanged.\n\nAfter training: fresh 4B Base/Round2/Round4, eight-GPU vLLM, same external199-example caps plus dev128, actual32k generation and CPU8k prefix scoring. See experiment_plan.json, live_progress.json and logs/.\n')
    return plan


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',default=str(DEFAULT));p.add_argument('--model',default=str(MODEL));p.add_argument('--run',action='store_true');p.add_argument('--detach',action='store_true');a=p.parse_args()
    root=Path(a.output_dir).resolve();root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    if a.detach:
        if not a.run:raise ValueError('--detach requires --run')
        launch_detached([sys.executable,'-u',str(Path(__file__).resolve()),'--output-dir',str(root),'--model',a.model,'--run'],root);return
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Controller signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    with (root/'.controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            plan=json.loads((root/'experiment_plan.json').read_text()) if (root/'experiment_plan.json').exists() else prepare(root,a.model)
            if a.run:execute(root,plan)
            else:status(root,'prepared',gpu_jobs_launched=False)
        except BaseException as exc:status(root,'failed',error=str(exc));raise

if __name__=='__main__':main()
