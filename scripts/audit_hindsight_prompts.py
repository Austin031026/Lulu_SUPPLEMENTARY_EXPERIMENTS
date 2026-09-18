"""Three paired hindsight views on saved on-policy prefixes; no generation/update."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lulu import training as tr
from lulu.hindsight_prompt_audit import PROMPTS, prompt_ids, sample_positions, structural_vocabulary, diagnostics
from lulu.persistent import Workers, _port, head_from_state, teacher_payload
from lulu.teacher_service import teacher_worker


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def prepare(x):
    root, out = Path(x.experiment).resolve(), Path(x.output_dir).resolve()
    if (out/'manifest.json').exists():
        raise ValueError('Manifest exists; use --run to resume without resampling')
    train = root/'arms/D_shared/train'
    config = json.loads((train/'run_config.json').read_text())
    tok = tr.load_tokenizer(str(train/'checkpoints/round_000000'))
    sources = {r['id']: r for r in tr.load_prepared_jsonl(config['train_data'])}
    records, hashes = [], {str(train/'run_config.json'): digest(train/'run_config.json'), config['train_data']: digest(config['train_data'])}
    for ri, rnd in enumerate((0,2)):
        checkpoint = train/f'checkpoints/round_{rnd:06d}'
        for path in list(checkpoint.glob('*.safetensors')) + [checkpoint/'config.json',checkpoint/'tokenizer_config.json']:
            hashes[str(path)] = digest(path)
        rows = []
        for path in sorted((train/f'rollouts/round_{rnd:04d}').glob('shard-*.jsonl')):
            hashes[str(path)] = digest(path)
            rows += [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
        if len(rows) != 64 or len({r['index'] for r in rows}) != 64:
            raise ValueError('Expected all 64 saved rollouts per snapshot')
        old_path = train/f'diagnostics/round_{rnd:04d}/position_scores.npz'
        hashes[str(old_path)] = digest(old_path)
        with np.load(old_path) as old:
            for j,r in enumerate(sorted(rows,key=lambda r:r['index'])):
                if r['snapshot_round'] != rnd or Path(r['rollout_checkpoint']).resolve() != checkpoint:
                    raise ValueError('Rollout snapshot mismatch')
                src = sources[r['source_id']]
                views = tr.build_prompt_views(tok,src['messages'],src['gold_answer'],enable_thinking=True)
                if views['causal_prompt_ids'] != r['causal_prompt_ids'] or src['gold_answer'] != r['gold_answer']:
                    raise ValueError('Original prompt/answer mismatch')
                h = {name:prompt_ids(tok,src['messages'],src['gold_answer'],name) for name in PROMPTS}
                if h['current'] != views['hindsight_prompt_ids']:
                    raise ValueError('Current prompt changed')
                sampled = sample_positions(r['reasoning_mask'],x.seed+rnd*100003+r['index'],x.reasoning_per_bin,x.control_per_bin)
                pos,exp,region,bins = map(list,zip(*sampled))
                mask = old['trajectory_index'] == r['index']
                lookup = dict(zip(old['position'][mask].tolist(),old['shared_mass'][mask].tolist()))
                saved = [lookup[p] if reasoning else 0. for p,reasoning in zip(pos,region)]
                rec = {k:r[k] for k in ('index','source_id','snapshot_round','causal_prompt_ids','response_ids','truncated','correct','response_tokens')}
                rec.update(uid=len(records),worker=3*ri+j%3,checkpoint=str(checkpoint),hindsight_views=h,
                    positions=pos,position_expansion=exp,region=region,position_bin=bins,saved_shared_mass=saved,
                    reasoning_tokens=sum(r['reasoning_mask']),control_tokens=len(r['reasoning_mask'])-sum(r['reasoning_mask']),
                    response_sha256=hashlib.sha256(json.dumps(r['response_ids']).encode()).hexdigest())
                if max(map(len,h.values()))+len(r['response_ids']) > config['max_sequence_tokens']:
                    raise ValueError('Context too long; no truncation is permitted')
                records.append(rec)
    out.mkdir(parents=True,exist_ok=True)
    (out/'selected_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    vocab_size=json.loads((train/'checkpoints/round_000000/config.json').read_text())['vocab_size']
    np.save(out/'structural_vocabulary.npy',structural_vocabulary(tok,vocab_size))
    code=out/'code';code.mkdir(exist_ok=True)
    source=Path(__file__).resolve().parents[1]
    shutil.copytree(source/'lulu',code/'lulu',ignore=shutil.ignore_patterns('__pycache__'))
    (code/'scripts').mkdir();shutil.copyfile(__file__,code/'scripts'/Path(__file__).name)
    for path in code.rglob('*.py'):hashes[str(path)]=digest(path)
    manifest=dict(experiment=str(root),created_utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),
        train_dir=str(train),prompts=PROMPTS,seed=x.seed,snapshots=[0,2],rollouts=len(records),
        sampled_positions=sum(len(r['positions']) for r in records),reasoning_per_bin=x.reasoning_per_bin,
        control_per_bin=x.control_per_bin,edges=[0,2048,4096,6144,8192],
        sampling='All 128 available snapshot-matched rollouts; uniform positions within region x 2k bin, same positions for all prompts; inverse inclusion weights',
        scope='Snapshots 1/3 are not retained and are excluded, not replaced with a different policy',
        normalization='Reasoning: token mean within rollout then equal prompt mean. Control: token mean; also report rollout mean.',
        structural_proxy='Special tokens or decoded whitespace/punctuation/symbol-only tokens; excludes replacement chars. Includes mathematical punctuation, not a semantic classifier.',
        gpu_roles={'snapshot0_students':[0,1,2],'snapshot2_students':[3,4,5],'teacher_tp2':[6,7]},
        precision='FP32 Student master weights, BF16 autocast; Teacher BF16 eager TP2; FP32 probability reductions',
        no_training=True,no_generation=True,source_sha256=hashes,selected_records_sha256=digest(out/'selected_records.jsonl'),
        structural_vocabulary_sha256=digest(out/'structural_vocabulary.npy'),
        decision_rule=dict(note='Operational pilot screen, not a validated scientific threshold; fixed before scoring',
            minimum_reasoning_shared_gain=.20,paired_bootstrap_delta_lower_bound=0,
            both_snapshots_positive=True,minimum_overlap_efficiency_ratio=.90,
            maximum_control_kl_ratio=1.5,maximum_control_kl_absolute_increase=.05,
            maximum_control_tv_ratio=1.5,maximum_control_tv_absolute_increase=.02,
            maximum_reasoning_structural_l1_ratio=1.5,minimum_nonstructural_shared_gain=.20))
    tr.atomic_json(out/'manifest.json',manifest)
    tr.atomic_json(out/'live_progress.json',dict(status='prepared',rollouts=len(records),sampled_positions=manifest['sampled_positions']))
    print(json.dumps({k:manifest[k] for k in ('rollouts','sampled_positions','gpu_roles')},indent=2),flush=True)


def student_worker(a,worker,checkpoint,out,conn):
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES=str(worker),LOCAL_RANK='0',RANK='0',WORLD_SIZE='1')
        torch.set_num_threads(1)
        tok=tr.load_tokenizer(checkpoint)
        model=tr.load_student(a,checkpoint_dir=Path(checkpoint),trainable=True)
        model.gradient_checkpointing_disable();model.requires_grad_(False).eval()
        head=model.get_output_embeddings();structural=np.load(Path(out)/'structural_vocabulary.npy')
        conn.send(dict(op='ready',role='prompt_audit_student',worker=worker))
        teacher_head=None
        while True:
            message=conn.recv()
            if message['op']=='stop':
                conn.send(dict(op='stopped'));return
            if message['op']=='head':
                teacher_head=head_from_state(message['state'],torch.device('cuda',0));continue
            if message['op']!='project' or teacher_head is None:
                raise ValueError('Invalid worker command')
            r=message['record'];started=time.monotonic();pieces={}
            with torch.inference_mode():
                c=tr.selected_hidden(model,tok,[r],'causal_prompt_ids',a)[0]
                hs=[]
                for name in PROMPTS:
                    rr=dict(r,hindsight_prompt_ids=r['hindsight_views'][name])
                    hs.append(tr.selected_hidden(model,tok,[rr],'hindsight_prompt_ids',a)[0])
                th=message['teacher_hidden'].cuda()
                for start in range(0,len(c),64):
                    end=start+64
                    with tr.autocast_context(a):
                        cl=head(c[start:end]);tl=teacher_head(th[start:end])
                    for name,h in zip(PROMPTS,hs):
                        with tr.autocast_context(a):hl=head(h[start:end])
                        for key,value in diagnostics(cl,hl,tl,structural).items():
                            pieces.setdefault(name+'__'+key,[]).append(value.cpu().numpy())
            arrays={k:np.concatenate(v) for k,v in pieces.items()}
            arrays.update({k:np.asarray(r[k]) for k in ('positions','position_expansion','region','position_bin','saved_shared_mass')})
            dest=Path(out)/'scores'/f'rollout_{r["uid"]:03d}.npz'
            with dest.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**arrays)
            dest.with_suffix('.tmp').replace(dest)
            del c,hs,th
            conn.send(dict(op='projected',worker=worker,uid=r['uid'],seconds=time.monotonic()-started))
    except BaseException as error:
        conn.send(dict(op='error',role='prompt_audit_student',error=str(error),traceback=traceback.format_exc()));raise
    finally:conn.close()


def run(x):
    out=Path(x.output_dir).resolve();manifest=json.loads((out/'manifest.json').read_text())
    if digest(out/'selected_records.jsonl') != manifest['selected_records_sha256'] or digest(out/'structural_vocabulary.npy') != manifest['structural_vocabulary_sha256']:
        raise ValueError('Audit inputs changed')
    for path,sha in manifest['source_sha256'].items():
        if digest(path)!=sha:raise ValueError(f'Frozen input/source changed: {path}')
    records=[json.loads(s) for s in (out/'selected_records.jsonl').read_text().splitlines()]
    (out/'scores').mkdir(exist_ok=True)
    completed=[r['uid'] for r in records if (out/'scores'/f'rollout_{r["uid"]:03d}.npz').exists()]
    pending_records=[r for r in records if r['uid'] not in completed]
    if not pending_records:
        print('All scores already complete');return
    config=json.loads((Path(manifest['train_dir'])/'run_config.json').read_text())
    a=tr.parser().parse_args(['--train-data',config['train_data'],'--output-dir',str(out),'--model',config['model'],
        '--teacher-model',config['teacher_model'],'--method','ren_shared','--lora-rank','0','--master-weights-fp32',
        '--teacher-tp-mode','eager-local','--score-batch-size','2','--worker-timeout','600','--no-gradient-checkpointing'])
    workers=Workers(600);began=time.monotonic();events=[]
    def progress(phase,**kw):
        value=dict(status=phase,controller_pid=os.getpid(),completed_rollouts=len(completed),total_rollouts=len(records),
            elapsed_seconds=time.monotonic()-began,updated_utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),**kw)
        tr.atomic_json(out/'live_progress.json',value);print(json.dumps(value),flush=True)
    try:
        progress('loading');port=_port()
        teacher=workers.launch(teacher_worker,(a,0,2,['6','7'],'127.0.0.1',port))
        workers.launch(teacher_worker,(a,1,2,['6','7'],'127.0.0.1',port),connected=False)
        students={i:workers.launch(student_worker,(a,i,next(r['checkpoint'] for r in pending_records if r['worker']==i),str(out)))
                  for i in sorted({r['worker'] for r in pending_records})}
        pending=[teacher,*students.values()];head=None
        while pending:
            conn,message=workers.receive(pending)
            if message['op']!='ready':raise ValueError('Unexpected load response')
            pending.remove(conn)
            if conn is teacher:head=message.pop('teacher_head')
            events.append(message);progress('loading',last_ready=message)
        for conn in students.values():conn.send(dict(op='head',state=head))
        outstanding=set()
        from multiprocessing.connection import wait
        for start in range(0,len(pending_records),2):
            batch=pending_records[start:start+2]
            teacher.send(dict(op='score',round=batch[-1]['snapshot_round'],request_id=start,records=teacher_payload(batch)))
            message=workers.expect(teacher,'scored')
            for r,result in zip(batch,message['records']):
                students[r['worker']].send(dict(op='project',record=r,teacher_hidden=result['teacher_hidden']))
                outstanding.add(r['uid'])
            for conn in wait(list(students.values()),timeout=0):
                _,result=workers.receive([conn]);completed.append(result['uid']);outstanding.remove(result['uid']);events.append(result)
            progress('scoring',teacher_scored=start+len(batch))
        while outstanding:
            _,result=workers.receive(list(students.values()))
            completed.append(result['uid']);outstanding.remove(result['uid']);events.append(result);progress('scoring')
        tr.atomic_json(out/'runtime.json',dict(elapsed_seconds=time.monotonic()-began,events=events,optimizer_steps=0,new_rollouts=0))
        progress('complete')
    except BaseException as error:
        progress('failed',error=str(error));raise
    finally:workers.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment');p.add_argument('--output-dir',required=True)
    p.add_argument('--seed',type=int,default=20260918)
    p.add_argument('--reasoning-per-bin',type=int,default=128)
    p.add_argument('--control-per-bin',type=int,default=32)
    p.add_argument('--run',action='store_true')
    x=p.parse_args();run(x) if x.run else prepare(x)

if __name__=='__main__':main()
