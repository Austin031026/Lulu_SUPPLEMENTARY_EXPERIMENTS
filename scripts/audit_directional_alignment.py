"""Bounded rescoring of existing on-policy states. No rollout, optimizer or training.

Six independent Student scorers (two per snapshot) overlap a resident TP2 Teacher.
Only sampled positions reach the full-vocabulary output head; prefixes are intact.
"""
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
import random
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lulu import training as tr
from lulu.directional_audit import sample_reasoning_positions,projection_diagnostics
from lulu.persistent import Workers,_port,head_from_state,teacher_payload
from lulu.teacher_service import teacher_worker


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def prepare(x):
    root=Path(x.experiment).resolve();out=Path(x.output_dir).resolve()
    if (out/'manifest.json').exists():raise ValueError('Use the existing manifest with --run; refusing to resample')
    config=json.loads((root/'train/run_config.json').read_text())
    tok=tr.load_tokenizer(str(root/'train/checkpoints/round_000000'))
    sources={r['id']:r for r in tr.load_prepared_jsonl(config['train_data'])}
    records=[];population=[];hashes={str(root/'train/run_config.json'):digest(root/'train/run_config.json'),config['train_data']:digest(config['train_data'])}
    for ri,round_ in enumerate([0,4,8]):
        paths=sorted((root/f'train/rollouts/round_{round_:04d}').glob('shard-*.jsonl'))
        rows=[]
        for path in paths:
            hashes[str(path)]=digest(path)
            rows.extend(json.loads(s) for s in path.read_text().splitlines() if s.strip())
        rows.sort(key=lambda r:r['index'])
        if len({r['index'] for r in rows})!=len(rows):raise ValueError('Duplicate rollout indices')
        selected=sorted(random.Random(x.seed+round_*100003).sample(rows,x.rollouts_per_round),key=lambda r:r['index'])
        for r in rows:population.append({k:r[k] for k in ['snapshot_round','index','source_id','truncated','correct','response_tokens']})
        checkpoint=root/f'train/checkpoints/round_{round_:06d}'
        for path in checkpoint.glob('*.json'):hashes[str(path)]=digest(path)
        for j,r in enumerate(selected):
            assert r['snapshot_round']==round_ and Path(r['rollout_checkpoint']).resolve()==checkpoint
            src=sources[r['source_id']]
            views=tr.build_prompt_views(tok,src['messages'],src['gold_answer'],enable_thinking=True)
            assert views['causal_prompt_ids']==r['causal_prompt_ids'] and src['gold_answer']==r['gold_answer']
            positions,expansion,bins=sample_reasoning_positions(r['reasoning_mask'],x.seed+round_*100003+r['index'],x.positions_per_bin)
            if not positions:raise ValueError('Empty reasoning rollout; explicitly handle before launching any scoring')
            record={k:r[k] for k in ('index','source_id','snapshot_round','causal_prompt_ids','response_ids','truncated','correct','response_tokens')}
            record.update(uid=len(records),worker=2*ri+j%2,checkpoint=str(checkpoint),
                hindsight_prompt_ids=views['hindsight_prompt_ids'],positions=positions,position_expansion=expansion,
                position_bin=bins,reasoning_tokens=sum(r['reasoning_mask']),
                response_sha256=hashlib.sha256(json.dumps(r['response_ids']).encode()).hexdigest())
            # Add old scores at precisely the same positions for numerical reconstruction checks.
            old_path=root/f'train/diagnostics/round_{round_:04d}/position_scores.npz'
            if str(old_path) not in hashes:hashes[str(old_path)]=digest(old_path)
            with np.load(old_path) as old:
                select=old['trajectory_index']==r['index']
                lookup={int(p):k for k,p in enumerate(old['position'][select])}
                ids=[lookup[p] for p in positions]
                record['saved_causal_kl']=old['causal_kl'][select][ids].tolist()
                record['saved_resolved_mismatch']=old['resolved_mismatch'][select][ids].tolist()
            records.append(record)
    out.mkdir(parents=True,exist_ok=True)
    (out/'selected_records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    tr.atomic_json(out/'population.json',population)
    code=out/'code';code.mkdir(exist_ok=True)
    for relative in ['lulu/directional_audit.py','lulu/training.py','lulu/data.py','lulu/persistent.py','lulu/teacher_service.py','lulu/eager_tp.py','scripts/audit_directional_alignment.py']:
        src=Path(__file__).resolve().parents[1]/relative;dest=code/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest);hashes[str(src)]=digest(src)
    manifest=dict(status='prepared',created_utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),experiment=str(root),
        seed=x.seed,rounds=[0,4,8],rollouts_per_round=x.rollouts_per_round,rollouts=len(records),
        positions=sum(len(r['positions']) for r in records),positions_per_bin=x.positions_per_bin,
        edges=[0,2048,4096,6144,8192],cosine_near_zero=.05,cosine_threshold_sensitivity=[.01,.05,.1],
        high_alignment_threshold=.5,epsilon=1e-8,variance_floor=1e-10,
        sampling='Uniform without replacement within each selected round; positions uniform within each nonempty absolute bin; no outcome/signal filtering',
        weighting='position_expansion = bin_population / bin_sample; primary prompt-balanced weight = position_expansion / reasoning_tokens (one rollout per prompt)',
        scope='0/4/8 round populations, not all 12 rounds; not a training outcome or proof of benefit',
        no_generation=True,no_training=True,training_horizon=8192,future_evaluation_horizon=32768,
        model_precision='Student FP32 master weights + BF16 autocast; Teacher BF16 eager TP2; log-softmax/moments FP32',
        gpu_roles={'student_snapshot_0':[0,1],'student_snapshot_4':[2,3],'student_snapshot_8':[4,5],'teacher_tp2':[6,7]},
        source_sha256=hashes,selected_records_sha256=digest(out/'selected_records.jsonl'))
    tr.atomic_json(out/'manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ['status','rollouts','positions','gpu_roles']},indent=2),flush=True)


def student_worker(a,worker,records,out,conn):
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES=str(worker),LOCAL_RANK='0',RANK='0',WORLD_SIZE='1')
        torch.set_num_threads(1)
        started=time.monotonic();tok=tr.load_tokenizer(records[0]['checkpoint'])
        model=tr.load_student(a,checkpoint_dir=Path(records[0]['checkpoint']),trainable=True)
        model.gradient_checkpointing_disable();model.requires_grad_(False).eval()
        head=model.get_output_embeddings();cache={}
        with torch.inference_mode():
            for r in records:
                c=tr.selected_hidden(model,tok,[r],'causal_prompt_ids',a)[0].cpu()
                h=tr.selected_hidden(model,tok,[r],'hindsight_prompt_ids',a)[0].cpu()
                cache[r['uid']]=(r,c,h)
        conn.send(dict(op='ready',role='audit_student',worker=worker,seconds=time.monotonic()-started))
        teacher_head=None
        while True:
            message=conn.recv()
            if message['op']=='stop':
                conn.send({'op':'stopped'});return
            if message['op']=='head':
                teacher_head=head_from_state(message['state'],torch.device('cuda',0));continue
            if message['op']!='project' or teacher_head is None:raise ValueError('Bad audit request')
            uid=message['uid'];r,c,h=cache.pop(uid);th=message['teacher_hidden']
            if len(th)!=len(r['positions']):raise ValueError('Teacher position mismatch')
            pieces={};start_time=time.monotonic()
            with torch.inference_mode():
                for start in range(0,len(c),32):
                    end=start+32
                    with tr.autocast_context(a):
                        cl=head(c[start:end].cuda());hl=head(h[start:end].cuda());tl=teacher_head(th[start:end].cuda())
                    actual=[r['response_ids'][p] for p in r['positions'][start:end]]
                    diag=projection_diagnostics(cl,hl,tl,actual)
                    for key,value in diag.items():pieces.setdefault(key,[]).append(value.cpu().numpy())
            arrays={k:np.concatenate(v) for k,v in pieces.items()}
            for key in ['positions','position_expansion','position_bin','saved_causal_kl','saved_resolved_mismatch']:
                arrays[key]=np.asarray(r[key])
            arrays['uid']=np.full(len(c),uid,dtype=np.int32)
            arrays['round']=np.full(len(c),r['snapshot_round'],dtype=np.int32)
            arrays['trajectory_index']=np.full(len(c),r['index'],dtype=np.int32)
            arrays['reasoning_tokens']=np.full(len(c),r['reasoning_tokens'],dtype=np.int32)
            arrays['capped']=np.full(len(c),r['truncated'],dtype=bool)
            arrays['correct']=np.full(len(c),r['correct'],dtype=bool)
            dest=Path(out)/'scores';dest.mkdir(exist_ok=True)
            np.savez_compressed(dest/f'rollout_{uid:03d}.npz',**arrays)
            conn.send(dict(op='projected',worker=worker,uid=uid,seconds=time.monotonic()-start_time,positions=len(c)))
    except BaseException as error:
        conn.send(dict(op='error',role='audit_student',error=str(error),traceback=traceback.format_exc()))
        raise
    finally:conn.close()


def run(x):
    out=Path(x.output_dir).resolve();manifest=json.loads((out/'manifest.json').read_text())
    if (out/'scores').exists() and list((out/'scores').glob('*.npz')):raise ValueError('Existing scores; refusing to overwrite')
    assert digest(out/'selected_records.jsonl')==manifest['selected_records_sha256']
    for path,sha in manifest['source_sha256'].items():
        if digest(path)!=sha:raise ValueError(f'Input/source changed after sampling: {path}')
    root=Path(manifest['experiment']);records=[json.loads(s) for s in (out/'selected_records.jsonl').read_text().splitlines()]
    config=json.loads((root/'train/run_config.json').read_text())
    a=tr.parser().parse_args(['--train-data',config['train_data'],'--output-dir',str(out),
        '--model',str(root/'train/checkpoints/round_000000'),'--teacher-model',config['teacher_model'],
        '--method','ren_balanced','--lora-rank','0','--master-weights-fp32','--teacher-tp-mode','eager-local',
        '--score-batch-size','2','--worker-timeout','600','--no-gradient-checkpointing'])
    workers=Workers(600);began=time.monotonic();completed=[];events=[]
    def progress(phase,**kw):
        value=dict(status=phase,completed_rollouts=len(completed),total_rollouts=len(records),
            elapsed_seconds=time.monotonic()-began,updated_utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),**kw)
        tr.atomic_json(out/'live_progress.json',value);print(json.dumps(value),flush=True)
    try:
        progress('loading')
        port=_port();teacher=workers.launch(teacher_worker,(a,0,2,['6','7'],'127.0.0.1',port))
        workers.launch(teacher_worker,(a,1,2,['6','7'],'127.0.0.1',port),connected=False)
        students=[workers.launch(student_worker,(a,i,[r for r in records if r['worker']==i],str(out))) for i in range(6)]
        pending=[teacher,*students];head=None
        while pending:
            conn,message=workers.receive(pending);assert message['op']=='ready'
            pending.remove(conn)
            if conn is teacher:head=message.pop('teacher_head')
            events.append(message);progress('loading',last_ready=message)
        for conn in students:conn.send(dict(op='head',state=head))
        remaining=set()
        # Teacher sees only causal input, exact saved continuation, and selected offsets.
        for start in range(0,len(records),2):
            batch=records[start:start+2]
            teacher.send(dict(op='score',round=batch[0]['snapshot_round'],request_id=start,records=teacher_payload(batch)))
            message=workers.expect(teacher,'scored')
            events.append({k:v for k,v in message.items() if k!='records'})
            for r,result in zip(batch,message['records']):
                students[r['worker']].send(dict(op='project',uid=r['uid'],teacher_hidden=result['teacher_hidden']))
                remaining.add(r['uid'])
            # Drain available small acknowledgements without serializing GPU heads.
            from multiprocessing.connection import wait
            for conn in wait(students,timeout=0):
                _,result=workers.receive([conn]);assert result['op']=='projected'
                completed.append(result['uid']);remaining.remove(result['uid']);events.append(result)
            progress('scoring',teacher_scored=start+len(batch))
        while remaining:
            _,result=workers.receive(students);assert result['op']=='projected'
            completed.append(result['uid']);remaining.remove(result['uid']);events.append(result);progress('projecting')
        tr.atomic_json(out/'runtime.json',dict(elapsed_seconds=time.monotonic()-began,events=events,
            torch=torch.__version__,cuda=torch.version.cuda,teacher_tp='eager-local',optimizer_steps=0,new_rollouts=0))
        progress('complete')
    except BaseException as error:
        progress('failed',error=str(error));raise
    finally:workers.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment');p.add_argument('--output-dir',required=True)
    p.add_argument('--seed',type=int,default=20260916)
    p.add_argument('--rollouts-per-round',type=int,default=16)
    p.add_argument('--positions-per-bin',type=int,default=64)
    p.add_argument('--run',action='store_true')
    x=p.parse_args()
    run(x) if x.run else prepare(x)

if __name__=='__main__':main()
