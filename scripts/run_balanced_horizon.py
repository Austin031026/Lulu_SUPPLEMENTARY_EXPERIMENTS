"""Frozen Base restart of bounded absolute ReN at 16k, followed by 32k cross-benchmark evaluation."""
from __future__ import annotations
import argparse,fcntl,json,os,signal,subprocess,sys,time
from pathlib import Path
from datetime import datetime,timezone
from run_decisive import (PROJECT,WORKSPACE,BENCH_ROOT,BENCHMARKS,digest,write_json,freeze_code,
    gpu_snapshot,resources_idle,teacher_ready,child_run,status,launch_detached)

OLD=WORKSPACE/'LuLu_outputs/experiments/ren_balanced_abs_full_qwen1p7_teacher32b_pool2048_dev256_r12_s42'
DEFAULT=WORKSPACE/'LuLu_outputs/experiments/ren_balanced_abs_full_qwen1p7_teacher32b_pool2048_h16384_r12_s42'


def prepare(root):
    source=json.loads((OLD/'train/run_config.json').read_text())
    base=OLD/'train/checkpoints/round_000000'
    assert json.loads((base/'lulu_state.json').read_text())['completed_rounds']==0
    assert json.loads((base/'config.json').read_text())['max_position_embeddings']>=40960
    assert json.loads((Path(source['teacher_model'])/'config.json').read_text())['max_position_embeddings']>=20480
    source_data=Path(source['train_data']);pool_meta=json.loads((source_data.parent/'manifest.json').read_text())
    assert pool_meta['pool_size']==2048 and digest(source_data)==source['train_sha256']
    frozen=root/'code';train_dir=root/'train'
    cmd=[sys.executable,'-u',str(frozen/'scripts/train_lulu.py'),'--model',str(base),
        '--teacher-model',source['teacher_model'],'--train-data',str(source_data),'--output-dir',str(train_dir),
        '--method','ren_balanced','--backend','persistent','--gpus','0,1,2,3,4,5,6,7',
        '--student-gpus','0,1,2,3,4','--hindsight-gpus','5','--teacher-gpus','6,7','--teacher-tp-mode','eager-local',
        '--rounds','12','--global-batch-prompts','64','--rollouts-per-prompt','1','--update-passes','1',
        '--rollout-backend','vllm','--rollout-batch-size','16','--rollout-top-k','20',
        '--rollout-vllm-memory','.40','--rollout-vllm-max-seqs','16','--score-batch-size','2','--train-micro-batch-size','1',
        '--max-new-tokens','16384','--max-prompt-tokens','4096','--max-sequence-tokens','20480',
        '--temperature','.6','--top-p','.95','--top-k','32','--logit-chunk-size','128',
        '--learning-rate','1e-6','--reference-kl-coef','.1','--weight-decay','0','--max-grad-norm','1',
        '--lora-rank','0','--master-weights-fp32','--gradient-checkpointing','--seed','42',
        '--save-every','20','--retain-checkpoints','4,8,12','--gradient-norm-every','4','--reasoning-diagnostic-split','8192',
        '--validation-every','0','--worker-timeout','3600','--stability-early-stop']
    evaluate=[sys.executable,'-u',str(frozen/'scripts/evaluate_lulu.py'),'--model',str(base),
        '--checkpoint',f'round8={train_dir}/checkpoints/round_000008','--checkpoint',f'final={train_dir}/checkpoints/latest',
        '--data-manifest',str(BENCH_ROOT/'manifest.json'),'--benchmarks',','.join(BENCHMARKS),'--split','full',
        '--gpus','0,1,2,3,4,5,6,7','--backend','vllm','--vllm-gpu-memory-utilization','.90',
        '--vllm-max-num-seqs','16','--vllm-max-num-batched-tokens','4096','--batch-size','16',
        '--max-response-tokens','32768','--max-prompt-tokens','4096','--max-model-len','40960','--context-safety-margin','128',
        '--thinking','--store-text','--store-token-ids','--max-examples','199','--decoding','qwen-thinking','--seed','42',
        '--parser-path',str(frozen/'lulu/thinking_final_parser.py'),'--output-dir',str(root/'evaluation')]
    ref=OLD/'long_horizon_38912_20260916/evaluation';refplan=json.loads((ref/'eval_plan.json').read_text())
    assert refplan['sampling']==dict(temperature=.6,top_p=.95,top_k=20,seed=42) and refplan['store_token_ids']
    paths=[source_data,OLD/'train/run_config.json',base/'config.json',base/'lulu_state.json',ref/'eval_plan.json']
    paths+=sorted((OLD/'train/metrics').glob('round_*.json'))
    paths+=sorted(base.glob('*.safetensors'))+sorted(base.glob('*token*json'))
    paths+=sorted(ref.glob('*/*/shard-*.jsonl'))+[Path(b['path']) for b in refplan['benchmarks']]
    paths+=[Path(source['teacher_model'])/'config.json',Path(source['teacher_model'])/'model.safetensors.index.json']
    plan=dict(schema_version=1,method='ren_balanced',rounds=12,seed=42,teacher=source['teacher_model'],
        model=str(base),base_completed_rounds=0,fresh_optimizer=True,train_command=cmd,eval_command=evaluate,
        old_experiment=str(OLD),reference_generations=str(ref),input_sha256={str(p):digest(p) for p in paths},
        training=dict(pool_size=2048,global_batch_prompts=64,rollouts_per_prompt=1,response_horizon=16384,total_context=20480,
            full_parameter=True,learning_rate=1e-6,reference_kl_coef=.1,weight='relu(DC-DH)/(1+relu(DC-DH))',weight_renormalization=False,
            reasoning_reduction='reasoning-token mean per rollout, then rollout mean per prompt, then prompt mean',
            control_reference='unchanged global generated-token average',gradient_norm_updates=[1,4,8,12],
            diagnostic_bins=[[0,8192],[8192,16384]],gradient_bin_denominator='full rollout reasoning-token count, not bin count',
            dev_selection=False,checkpoint_policy='fixed round8 and final; retained round4/8/12/latest',
            efficiency='five resident Student/vLLM GPUs, one Hindsight/reference GPU, two Teacher TP GPUs; existing parallel scoring'),
        evaluation=dict(models=['round8','final'],benchmarks=[dict(b,examples=b['expected_examples']) for b in refplan['benchmarks']],
            examples_per_model=825,max_examples_per_benchmark=199,response_horizon=32768,
            selection='first rows before sharding; checkpoints fixed before training',scoring='thinking_final_parser; legacy score retained for audit',
            references='CPU-only 32k prefix rescoring of cached Base, 8k Round8 and 8k Final12; no fresh baseline generation'),
        hf_cache=str(WORKSPACE.parent/'huggingface_cache/hub'),
        resource_policy=dict(gpus=[str(i) for i in range(8)],idle_checks=1,poll_seconds=15,idle_memory_mib=1024,idle_utilization=5))
    plan['code_sha256']=freeze_code(root)
    write_json(root/'experiment_plan.json',plan)
    return plan


def verify(root,plan):
    for path,sha in plan['input_sha256'].items():
        if digest(path)!=sha:raise ValueError(f'Frozen input changed: {path}')
    for rel,sha in plan['code_sha256'].items():
        if digest(root/'code'/rel)!=sha:raise ValueError(f'Frozen source changed: {rel}')


def execute(root,plan):
    env=dict(os.environ,HF_HOME=str(Path(plan['hf_cache']).parent),HF_HUB_CACHE=plan['hf_cache'],HUGGINGFACE_HUB_CACHE=plan['hf_cache'],
        HF_HUB_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VLLM_DISABLE_COMPILE_CACHE='1')
    env.pop('TRANSFORMERS_CACHE',None)
    verify(root,plan)
    # This is CPU work, and checks reuse/prompt/seed/parser consistency before GPU allocation.
    if not (root/'reference_32k/summary.json').exists():
        child_run([sys.executable,str(root/'code/scripts/reference_horizon_evaluation.py'),'--experiment-dir',str(root)],root,'reference_rescore',env,plan)
    while not resources_idle(gpu_snapshot(),plan['resource_policy']):
        status(root,'waiting_for_resources',gpus=gpu_snapshot());time.sleep(15)
    if not teacher_ready(plan['teacher']):raise ValueError('Teacher shards are incomplete')
    state_file=root/'train/checkpoints/latest/lulu_state.json'
    completed=json.loads(state_file.read_text())['completed_rounds'] if state_file.exists() else 0
    stopped=(root/'train/early_stop.json').exists()
    if completed<plan['rounds'] and not stopped:
        command=list(plan['train_command'])
        if (root/'train/run_config.json').exists():command+=['--resume']
        child_run(command,root,'training',env,plan)
    state=json.loads(state_file.read_text());completed=state['completed_rounds']
    if completed<12 and not (root/'train/early_stop.json').exists():raise RuntimeError('Training ended without every intended update')
    verify(root,plan)
    eval_command=list(plan['eval_command']);models=['round8','final']
    if completed<=8:
        keep=[int(p.name.split('_')[1]) for p in (root/'train/checkpoints').glob('round_*') if 0<int(p.name.split('_')[1])<completed]
        pos=eval_command.index('--checkpoint')
        if keep:
            k=max(keep);eval_command[pos+1]=f'round{k}={root}/train/checkpoints/round_{k:06d}';models=[f'round{k}','final']
        else:del eval_command[pos:pos+2];models=['final']
    effective=dict(plan,evaluation=dict(plan['evaluation'],models=models),eval_command=eval_command)
    write_json(root/'effective_evaluation_plan.json',dict(effective['evaluation'],command=eval_command,completed_rounds=completed))
    if not (root/'evaluation/summary.json').exists():
        if (root/'evaluation').exists():raise RuntimeError('Partial evaluation preserved; inspect before resuming')
        child_run(eval_command,root,'evaluation',env,effective)
    child_run([sys.executable,str(root/'code/scripts/summarize_horizon_run.py'),'--experiment-dir',str(root)],root,'analysis',env,effective)
    status(root,'complete',completed_rounds=completed,report=str(root/'REPORT.md'),summary=str(root/'comparison.json'),
        gradient_diagnostics=str(root/'train/analysis/stability.csv'),early_stopped=(root/'train/early_stop.json').exists())


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',default=str(DEFAULT));p.add_argument('--run',action='store_true');p.add_argument('--detach',action='store_true');a=p.parse_args();root=Path(a.output_dir).resolve()
    root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    if a.detach:
        if not a.run:raise ValueError('--detach requires --run')
        launch_detached([sys.executable,'-u',str(Path(__file__).resolve()),'--output-dir',str(root),'--run'],root);return
    with (root/'.controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=json.loads((root/'experiment_plan.json').read_text()) if (root/'experiment_plan.json').exists() else prepare(root)
        if not a.run:
            status(root,'prepared',gpu_jobs_launched=False,plan=str(root/'experiment_plan.json'));return
        try:execute(root,plan)
        except BaseException as exc:
            status(root,'failed',error=str(exc));raise
if __name__=='__main__':main()
