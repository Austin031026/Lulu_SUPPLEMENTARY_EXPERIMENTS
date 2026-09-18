"""Bounded rescoring of existing on-policy states. No rollout, optimizer or training.

Six independent Student scorers overlap two independent single-GPU Teachers.
Same fixed positions as the earlier directional audit, probability-space targets.
Also compare BF16 and FP32 output heads on identical retained hidden states.
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
from lulu.certified_audit import certified_diagnostics
from lulu.persistent import Workers,_port,head_from_state,teacher_payload
from lulu.teacher_service import teacher_worker


def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


def prepare(x):
    source=Path(x.source_audit).resolve();out=Path(x.output_dir).resolve()
    if (out/'manifest.json').exists():raise ValueError('Refusing to overwrite a prepared audit')
    previous=json.loads((source/'manifest.json').read_text())
    assert digest(source/'selected_records.jsonl')==previous['selected_records_sha256']
    for path,sha in previous['source_sha256'].items():
        if digest(path)!=sha:raise ValueError(f'Original input changed: {path}')
    out.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(source/'selected_records.jsonl',out/'selected_records.jsonl')
    shutil.copyfile(source/'population.json',out/'population.json')
    hashes={str(source/'manifest.json'):digest(source/'manifest.json'),
            str(source/'selected_records.jsonl'):digest(source/'selected_records.jsonl')}
    # Inherit immutable experiment inputs, but record the new scorer separately.
    for path,sha in previous['source_sha256'].items():
        if '/Lulu/' not in path:hashes[path]=sha
    for relative in ['lulu/certified_audit.py','lulu/training.py','lulu/data.py','lulu/persistent.py',
                     'lulu/teacher_service.py','lulu/eager_tp.py','scripts/audit_certified_correction.py']:
        src=Path(__file__).resolve().parents[1]/relative;dest=out/'code'/relative
        dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dest);hashes[str(src)]=digest(src)
    manifest=dict(previous)
    manifest.update(status='prepared',created_utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),
        source_audit=str(source),source_sha256=hashes,selected_records_sha256=digest(out/'selected_records.jsonl'),
        diagnostic='probability_simplex_projection',epsilon=1e-8,epsilon_sensitivity=[1e-10,1e-6,1e-4],
        coefficient='clip(dot(pH-pC,qT-pC)/(squared_norm(qT-pC)+epsilon),0,1); no extra sparsity/norm gate',
        target='(1-lambda)*pC+lambda*qT; detached; unwarped temperature 1',
        ratio_norm_floor=1e-7,ratio_floor_purpose='Reporting only, does not alter lambda or target',
        checks={'absolute_fp64_gradient_identity_tolerance':1e-12,
                'relative_ratio_fp64_tolerance_above_norm_floor':1e-8,
                'fp32_absolute_gradient_identity_tolerance':5e-7},
        retained_hidden_states=True,head_precision_sensitivity='BF16 autocast versus FP32 GEMM, TF32 disabled; identical C/H/T hidden states',
        numerical_scope='FP32 head sensitivity does not test all-FP32 backbones or other scorer batches',
        teacher_backend='two independent TP1 replicas, score batch 1',
        model_precision='Student FP32 master weights + BF16 autocast; two single-GPU BF16 Teachers; probability reductions FP32; second head pass FP32 without TF32',
        gpu_roles={'student_snapshot_0':[0,1],'student_snapshot_4':[2,3],'student_snapshot_8':[4,5],'teacher_replicas':[6,7]})
    manifest.pop('variance_floor',None)
    tr.atomic_json(out/'manifest.json',manifest)
    print(json.dumps({k:manifest[k] for k in ['status','rollouts','positions','source_audit']},indent=2),flush=True)


def student_worker(a,worker,records,out,conn):
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES=str(worker),LOCAL_RANK='0',RANK='0',WORLD_SIZE='1')
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32=False
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
                teacher_head=head_from_state(message['state'],torch.device('cuda',0))
                teacher_weight_fp32=teacher_head.weight.float()
                teacher_bias_fp32=None if teacher_head.bias is None else teacher_head.bias.float()
                continue
            if message['op']!='project' or teacher_head is None:raise ValueError('Bad audit request')
            uid=message['uid'];r,c,h=cache.pop(uid);th=message['teacher_hidden']
            if len(th)!=len(r['positions']):raise ValueError('Teacher position mismatch')
            hidden_dir=Path(out)/'hidden';hidden_dir.mkdir(exist_ok=True)
            tr.atomic_torch(hidden_dir/f'rollout_{uid:03d}.pt',dict(causal=c,hindsight=h,teacher=th,positions=torch.tensor(r['positions'])))
            pieces={};start_time=time.monotonic()
            with torch.inference_mode():
                for start in range(0,len(c),32):
                    end=start+32
                    with tr.autocast_context(a):
                        cl=head(c[start:end].cuda());hl=head(h[start:end].cuda());tl=teacher_head(th[start:end].cuda())
                    actual=[r['response_ids'][p] for p in r['positions'][start:end]]
                    diag=certified_diagnostics(cl,hl,tl,actual)
                    for key,value in diag.items():pieces.setdefault(key,[]).append(value.cpu().numpy())
                    # Reuse backbone states; only the output-head precision changes.
                    from torch.nn import functional as F
                    cf=F.linear(c[start:end].cuda().float(),head.weight,head.bias)
                    hf=F.linear(h[start:end].cuda().float(),head.weight,head.bias)
                    tf=F.linear(th[start:end].cuda().float(),teacher_weight_fp32,teacher_bias_fp32)
                    full=certified_diagnostics(cf,hf,tf,actual)
                    for key,value in full.items():pieces.setdefault('fp32head_'+key,[]).append(value.cpu().numpy())
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
        '--score-batch-size','1','--worker-timeout','600','--no-gradient-checkpointing'])
    workers=Workers(600);began=time.monotonic();completed=[];events=[]
    def progress(phase,**kw):
        value=dict(status=phase,completed_rollouts=len(completed),total_rollouts=len(records),
            elapsed_seconds=time.monotonic()-began,updated_utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),**kw)
        tr.atomic_json(out/'live_progress.json',value);print(json.dumps(value),flush=True)
    try:
        progress('loading')
        teachers=[workers.launch(teacher_worker,(a,0,1,[str(gpu)],'127.0.0.1',_port())) for gpu in [6,7]]
        students=[workers.launch(student_worker,(a,i,[r for r in records if r['worker']==i],str(out))) for i in range(6)]
        pending=[*teachers,*students];head=None
        while pending:
            conn,message=workers.receive(pending);assert message['op']=='ready'
            pending.remove(conn)
            if conn in teachers:
                state=message.pop('teacher_head')
                if head is None:head=state
            events.append(message);progress('loading',last_ready=message)
        for conn in students:conn.send(dict(op='head',state=head))
        remaining=set()
        # Teacher sees only causal input, exact saved continuation, and selected offsets.
        for start in range(0,len(records),2):
            batch=records[start:start+2]
            for conn,r in zip(teachers,batch):
                conn.send(dict(op='score',round=r['snapshot_round'],request_id=r['uid'],records=teacher_payload([r])))
            pending_teacher={conn:r for conn,r in zip(teachers,batch)}
            while pending_teacher:
                conn,message=workers.receive(list(pending_teacher));r=pending_teacher.pop(conn)
                assert message['op']=='scored'
                events.append({k:v for k,v in message.items() if k!='records'})
                result=message['records'][0]
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
            torch=torch.__version__,cuda=torch.version.cuda,teacher_tp='two_replicas_tp1',optimizer_steps=0,new_rollouts=0))
        progress('complete')
    except BaseException as error:
        progress('failed',error=str(error));raise
    finally:workers.close()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-audit');p.add_argument('--output-dir',required=True)
    p.add_argument('--run',action='store_true');x=p.parse_args()
    run(x) if x.run else prepare(x)

if __name__=='__main__':main()
