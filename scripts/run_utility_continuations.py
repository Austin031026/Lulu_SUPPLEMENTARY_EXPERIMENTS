"""Batched vLLM matched first-token continuation audit; never modifies models."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lulu import training as tr


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def grade(text,gold):
    from lulu import benchmark_parser as bp
    result={}
    for name,body in [('legacy',text),('strict_final',text.rsplit('</think>',1)[1] if '</think>' in text else '')]:
        try:
            def timed_out(*_):raise TimeoutError('Math verifier exceeded 3 seconds')
            previous=signal.signal(signal.SIGALRM,timed_out);signal.setitimer(signal.ITIMER_REAL,3)
            try:
                pred=bp.extract_answer(body,'math500') if body.strip() else ''
                success=bool(pred and bp.math_equal(pred,gold))
            finally:
                signal.setitimer(signal.ITIMER_REAL,0);signal.signal(signal.SIGALRM,previous)
            result[name+'_prediction']=str(pred);result[name+'_success']=success
        except Exception as exc:
            result[name+'_prediction']='';result[name+'_success']=False
            result[name+'_error']=type(exc).__name__+': '+str(exc)
    return result


def worker(out,task_id):
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    task=json.loads((out/'continuation_tasks.json').read_text())['tasks'][task_id]
    wanted=set(task['job_ids']);jobs=[r for r in read_rows(out/'continuation_jobs.jsonl') if r['job_id'] in wanted]
    assert len(jobs)==len(wanted)
    path=out/'continuations'/f'task_{task_id:02d}.jsonl'
    if path.exists():raise ValueError('Refuse to overwrite completed shard')
    started=time.time();model=task['checkpoint'];tokenizer=AutoTokenizer.from_pretrained(model,local_files_only=True)
    active=[j for j in jobs if not j['first_token_is_eos']]
    outputs={};engine=None
    try:
        if active:
            engine=LLM(model=model,tokenizer=model,dtype='bfloat16',trust_remote_code=False,
                tensor_parallel_size=1,max_model_len=40960,gpu_memory_utilization=.90,
                max_num_seqs=32,max_num_batched_tokens=4096,enable_chunked_prefill=True,
                enable_prefix_caching=True,enforce_eager=False,swap_space=0,seed=42,generation_config='vllm')
            params=[SamplingParams(n=1,temperature=.6,top_p=.95,top_k=20,min_p=0.,repetition_penalty=1.,
                max_tokens=j['max_new_tokens'],seed=j['seed'],stop_token_ids=[151645,151643],
                ignore_eos=False,detokenize=False) for j in active]
            begin=time.time()
            generated=engine.generate([{'prompt_token_ids':j['prompt_token_ids']} for j in active],sampling_params=params,use_tqdm=True)
            generation_seconds=time.time()-begin
            assert len(generated)==len(active)
            for j,r in zip(active,generated):
                assert r.prompt_token_ids==j['prompt_token_ids'],'vLLM altered the input prefix'
                assert r.finished and len(r.outputs)==1
                completion=r.outputs[0];ids=list(completion.token_ids)
                assert len(ids)<=j['max_new_tokens']
                outputs[j['job_id']]=(ids,completion.finish_reason)
        else:generation_seconds=0.
    finally:
        if engine is not None:
            core=getattr(getattr(engine,'llm_engine',None),'engine_core',None)
            if core is not None:core.shutdown()
    rows=[]
    for j in jobs:
        ids,finish=([], 'forced_eos') if j['first_token_is_eos'] else outputs[j['job_id']]
        full=j['response_prefix_ids']+ids
        assert len(full)<=32768
        text=tokenizer.decode(full,skip_special_tokens=True)
        result={k:v for k,v in j.items() if k not in ['prompt_token_ids','response_prefix_ids','checkpoint']}
        result.update(continuation_token_ids=ids,response_text=text,finish_reason=finish,
            continuation_tokens=len(ids),full_response_tokens=len(full),hit_cap=finish=='length',
            verification=grade(text,j['gold_answer']))
        rows.append(result)
    tmp=path.with_suffix('.jsonl.tmp');tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));tmp.replace(path)
    tr.atomic_json(out/'continuations'/f'task_{task_id:02d}.summary.json',dict(task_id=task_id,
        jobs=len(jobs),generation_seconds=generation_seconds,elapsed_seconds=time.time()-started,
        continuation_tokens=sum(r['continuation_tokens'] for r in rows),hit_cap=sum(r['hit_cap'] for r in rows),
        verification_errors=sum(any(k.endswith('_error') for k in r['verification']) for r in rows)))
    print(json.dumps({'status':'complete','task':task_id,'jobs':len(jobs)}),flush=True)


def prepare_tasks(out):
    jobs=read_rows(out/'continuation_jobs.jsonl');events=read_rows(out/'pair_events.jsonl')
    paired=defaultdict(list)
    for j in jobs:paired[(j['state_id'],j['replicate'])].append(j)
    for pair in paired.values():
        assert len(pair)==2 and {j['arm'] for j in pair}=={'control','intervention'}
        a,b=pair;assert a['seed']==b['seed'] and a['eta']==b['eta']==.2
        assert a['prompt_token_ids'][:-1]==b['prompt_token_ids'][:-1] and a['first_token']!=b['first_token']
        assert len(a['prompt_token_ids'])+a['max_new_tokens']<=40960-128
    assert len(paired)==sum(e['generated_pair'] for e in events)
    tasks=[]
    for round_ in sorted({j['snapshot_round'] for j in jobs}):
        pairs=[p for p in paired.values() if p[0]['snapshot_round']==round_]
        bins=[[] for _ in range(4)];costs=[0]*4
        for pair in sorted(pairs,key=lambda p:-p[0]['max_new_tokens']):
            k=min(range(4),key=lambda i:costs[i]);bins[k].extend(j['job_id'] for j in pair);costs[k]+=sum(j['max_new_tokens'] for j in pair)
        for ids in bins:
            if ids:tasks.append(dict(task_id=len(tasks),snapshot_round=round_,checkpoint=pairs[0][0]['checkpoint'],job_ids=ids))
    return {'tasks':tasks,'total_jobs':len(jobs),'gpus':list(range(8)),
        'runtime':{'vllm':'installed 0.9.0','max_num_seqs':32,'max_num_batched_tokens':4096,'gpu_memory_utilization':.90,
                   'enable_prefix_caching':True,'chunked_prefill':True,'tensor_parallel_size':1},
        'verification':{'primary':'Existing math500 parser on full response',
            'secondary':'Same parser only after final </think>; no close-think => failure',
            'errors':'Retain and report; provisional success=False, never silently drop'},
        'jobs_sha256':hashlib.sha256((out/'continuation_jobs.jsonl').read_bytes()).hexdigest()}


def manager(out):
    out.mkdir(parents=True,exist_ok=True);(out/'continuations').mkdir(exist_ok=True);(out/'logs').mkdir(exist_ok=True)
    task_file=out/'continuation_tasks.json'
    plan=prepare_tasks(out)
    if task_file.exists():assert json.loads(task_file.read_text())==plan,'Frozen task plan differs'
    else:tr.atomic_json(task_file,plan)
    for src in [Path(__file__),Path(__file__).with_name('prepare_utility_states.py'),Path(__file__).parents[1]/'lulu/semantic_audit.py']:
        shutil.copy2(src,out/'code'/src.name)
    finished={t['task_id'] for t in plan['tasks'] if (out/'continuations'/f"task_{t['task_id']:02d}.summary.json").exists()}
    active={}
    previous=json.loads((out/'live_progress.json').read_text()) if (out/'live_progress.json').exists() else {}
    import psutil
    class Adopted:
        def __init__(self,pid,task_id):self.pid=pid;self.task_id=task_id
        def poll(self):
            try:alive=psutil.Process(self.pid).status()!=psutil.STATUS_ZOMBIE
            except psutil.NoSuchProcess:alive=False
            if alive:return None
            return 0 if (out/'continuations'/f'task_{self.task_id:02d}.summary.json').exists() else 1
    for running in previous.get('running',[]):
        pid=running['pid'];tid=running['task'];gpu=running['gpu']
        if tid in finished:continue
        try:
            cmd=psutil.Process(pid).cmdline()
            if not any('run_utility_continuations.py' in c for c in cmd):continue
            if str(out) not in cmd:continue
            if psutil.Process(pid).status()==psutil.STATUS_ZOMBIE:continue
        except psutil.NoSuchProcess:continue
        task=plan['tasks'][tid];log=out/'logs'/f'continuation_{tid:02d}.log'
        active[gpu]=(task,Adopted(pid,tid),log.open('a'),log)
    adopted={v[0]['task_id'] for v in active.values()}
    pending=sorted([t for t in plan['tasks'] if t['task_id'] not in finished|adopted],key=lambda t:-len(t['job_ids']))
    failed=[];began=time.time();run_started=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime())
    def progress():
        running=[];completed_jobs=sum(len(t['job_ids']) for t in plan['tasks'] if t['task_id'] in finished)
        for gpu,(task,proc,handle,log) in active.items():
            # Count only vLLM generation progress, never model shard loading.
            matches=re.findall(r'Processed prompts:[^\r\n]*?\|\s*(\d+)/(\d+)',log.read_text(errors='replace'))
            done=int(matches[-1][0]) if matches else 0
            running.append({'task':task['task_id'],'gpu':gpu,'pid':proc.pid,'completed_generations':done,'jobs':len(task['job_ids'])})
        tr.atomic_json(out/'live_progress.json',dict(status='failed' if failed else ('complete' if not pending and not active else 'running'),
            stage='downstream_utility',started_at=run_started,updated_at=time.strftime('%Y-%m-%d %H:%M:%S UTC',time.gmtime()),
            elapsed_seconds=time.time()-began,total_jobs=plan['total_jobs'],completed_verified_jobs=completed_jobs,
            completed_tasks=sorted(finished),running=running,pending_tasks=len(pending),failures=failed))
    while pending or active:
        for gpu in range(8):
            if gpu in active or not pending or failed:continue
            task=pending.pop(0);log=out/'logs'/f"continuation_{task['task_id']:02d}.log";handle=log.open('a')
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
            env.pop('TRANSFORMERS_CACHE',None)
            env.update(VLLM_DISABLE_COMPILE_CACHE='1',VLLM_CACHE_ROOT=str(out/'runtime_cache'/f'vllm_{task["task_id"]}'),TORCHINDUCTOR_CACHE_DIR=str(out/'runtime_cache'/f'inductor_{task["task_id"]}'))
            handle.write('\nRecovery/new worker with isolated compile caches\n');handle.flush()
            proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--output-dir',str(out),'--worker',str(task['task_id'])],env=env,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
            active[gpu]=(task,proc,handle,log)
        progress()
        time.sleep(10)
        for gpu,(task,proc,handle,log) in list(active.items()):
            rc=proc.poll()
            if rc is None:continue
            handle.close();del active[gpu]
            if rc==0 and (out/'continuations'/f"task_{task['task_id']:02d}.summary.json").exists():finished.add(task['task_id'])
            else:failed.append({'task':task['task_id'],'returncode':rc,'log':str(log)})
        if failed and not active:break
    progress()
    if failed:raise RuntimeError(f'Continuation tasks failed: {failed}')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--worker',type=int)
    x=p.parse_args();out=Path(x.output_dir).resolve()
    worker(out,x.worker) if x.worker is not None else manager(out)
if __name__=='__main__':main()
