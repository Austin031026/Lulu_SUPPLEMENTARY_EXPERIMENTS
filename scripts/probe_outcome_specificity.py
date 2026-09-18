"""Correct vs wrong answer probe, reusing saved C/H/T backbone states."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lulu import training as tr
from lulu.semantic_audit import load_output_weight,probability_probe,stable_seed
from lulu.persistent import Workers


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(x):
    source=Path(x.source_audit).resolve();out=Path(x.output_dir).resolve()
    if (out/'manifest.json').exists():raise ValueError('Already prepared')
    original=json.loads((source/'manifest.json').read_text());root=Path(original['experiment'])
    cfg=json.loads((root/'train/run_config.json').read_text())
    rows=[json.loads(s) for s in (source/'selected_records.jsonl').read_text().splitlines()]
    pool=tr.load_prepared_jsonl(cfg['train_data']);byid={r['id']:r for r in pool}
    tok=tr.load_tokenizer(rows[0]['checkpoint']);records=[]
    for r in rows:
        gold=byid[r['source_id']]['gold_answer']
        if not __import__('re').fullmatch(r'-?\d+',gold):raise ValueError('This prepared audit expects integer DAPO labels')
        candidates=[]
        for donor in pool:
            other=donor['gold_answer']
            if donor['id']==r['source_id'] or not __import__('re').fullmatch(r'-?\d+',other):continue
            if int(other)==int(gold) or len(other)!=len(gold) or other.startswith('-')!=gold.startswith('-'):continue
            if len(tok.encode(other,add_special_tokens=False))!=len(tok.encode(gold,add_special_tokens=False)):continue
            candidates.append(donor)
        random.Random(stable_seed(20260916,'wrong',r['uid'])).shuffle(candidates)
        for donor in candidates:
            views=tr.build_prompt_views(tok,byid[r['source_id']]['messages'],donor['gold_answer'],enable_thinking=True)
            if len(views['hindsight_prompt_ids'])==len(r['hindsight_prompt_ids']):break
        else:raise ValueError(f'No format/length matched wrong answer for {r["uid"]}')
        assert views['causal_prompt_ids']==r['causal_prompt_ids']
        gold_views=tr.build_prompt_views(tok,byid[r['source_id']]['messages'],gold,enable_thinking=True)
        assert gold_views['hindsight_prompt_ids']==r['hindsight_prompt_ids']
        record=dict(r,gold_answer=gold,wrong_answer=donor['gold_answer'],wrong_source_id=donor['id'],
            wrong_prompt_ids=views['hindsight_prompt_ids'],hidden_path=str(source/'hidden'/f'rollout_{r["uid"]:03d}.pt'))
        records.append(record)
    out.mkdir(parents=True,exist_ok=True)
    (out/'records.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    hashes={str(source/'manifest.json'):sha(source/'manifest.json'),str(source/'selected_records.jsonl'):sha(source/'selected_records.jsonl'),cfg['train_data']:sha(cfg['train_data'])}
    for r in records:hashes[r['hidden_path']]=sha(r['hidden_path'])
    for relative in ['lulu/semantic_audit.py','lulu/training.py','lulu/data.py','scripts/probe_outcome_specificity.py']:
        p=Path(__file__).resolve().parents[1]/relative;dest=out/'code'/relative;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dest);hashes[str(p)]=sha(p)
    manifest=dict(status='prepared',experiment=str(root),source_audit=str(source),teacher_model=cfg['teacher_model'],
        created_utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime()),records=48,positions=sum(len(r['positions']) for r in records),
        wrong_answer_rule='Other training-pool question; numerically unequal integer; equal sign pattern, character length, answer token count and complete hindsight prompt token count; deterministic seed, no outcome/signal selection',
        probe_precision='FP32 output-head matmul and softmax; TF32 disabled; existing BF16 backbone states C/H/T; new W from same Student snapshot FP32 master + BF16 autocast',
        epsilon=1e-8,normalization='Reasoning-token within rollout, then prompt mean; inverse sampling probability',
        eta=.2,continuation_replicates=2,target_matched_triplets=72,total_response_horizon=32768,
        lambda_strata={'high':'>0.25','medium':'(0,0.25]','zero':'=0'},no_training=True,
        input_sha256=hashes,records_sha256=sha(out/'records.jsonl'))
    tr.atomic_json(out/'manifest.json',manifest);print(json.dumps({k:manifest[k] for k in ['status','records','positions','wrong_answer_rule']},indent=2),flush=True)


def worker(a,worker_id,records,out,conn):
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES=str(worker_id),LOCAL_RANK='0',RANK='0',WORLD_SIZE='1')
        torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
        tok=tr.load_tokenizer(records[0]['checkpoint']);model=tr.load_student(a,Path(records[0]['checkpoint']),trainable=True)
        model.gradient_checkpointing_disable();model.requires_grad_(False).eval()
        cw=model.get_output_embeddings().weight;tw=load_output_weight(a.teacher_model,'cuda')
        for r in records:
            cache=tr.load_tensor_file(r['hidden_path']);assert cache['positions'].tolist()==r['positions']
            with torch.inference_mode():
                wh=tr.selected_hidden(model,tok,[r],'wrong_prompt_ids',a)[0].cpu()
                dest=Path(out)/'wrong_hidden';dest.mkdir(exist_ok=True)
                tr.atomic_torch(dest/f'rollout_{r["uid"]:03d}.pt',dict(hidden=wh,positions=cache['positions']))
                pieces={}
                for start in range(0,len(wh),32):
                    end=start+32
                    c=torch.nn.functional.linear(cache['causal'][start:end].cuda().float(),cw)
                    h=torch.nn.functional.linear(cache['hindsight'][start:end].cuda().float(),cw)
                    t=torch.nn.functional.linear(cache['teacher'][start:end].cuda().float(),tw)
                    w=torch.nn.functional.linear(wh[start:end].cuda().float(),cw)
                    for label,logits in [('gold',h),('wrong',w)]:
                        diag=probability_probe(c,logits,t)
                        for k,v in diag.items():pieces.setdefault(label+'_'+k,[]).append(v.cpu().numpy())
                    actual=torch.tensor([r['response_ids'][pos] for pos in r['positions'][start:end]],device='cuda')
                    values=c.log_softmax(-1).gather(1,actual[:,None]).squeeze(1).exp()
                    pieces.setdefault('student_actual_probability',[]).append(values.cpu().numpy())
            arrays={k:np.concatenate(v) for k,v in pieces.items()}
            for k in ['positions','position_expansion','position_bin']:arrays[k]=np.asarray(r[k])
            for k,v in [('uid',r['uid']),('round',r['snapshot_round']),('reasoning_tokens',r['reasoning_tokens']),('capped',r['truncated']),('correct',r['correct'])]:arrays[k]=np.full(len(wh),v)
            dest=Path(out)/'scores';dest.mkdir(exist_ok=True);np.savez_compressed(dest/f'rollout_{r["uid"]:03d}.npz',**arrays)
            conn.send(dict(op='scored',uid=r['uid'],worker=worker_id))
        conn.send(dict(op='done',worker=worker_id))
        if conn.recv()['op']!='stop':raise ValueError('Expected shutdown')
    except BaseException as error:
        conn.send(dict(op='error',role='outcome_probe',error=str(error),traceback=traceback.format_exc()));raise
    finally:conn.close()


def run(x):
    out=Path(x.output_dir).resolve();manifest=json.loads((out/'manifest.json').read_text())
    assert sha(out/'records.jsonl')==manifest['records_sha256']
    for path,value in manifest['input_sha256'].items():assert sha(path)==value,path
    records=[json.loads(s) for s in (out/'records.jsonl').read_text().splitlines()]
    if (out/'scores').exists():raise ValueError('Refusing to overwrite scores')
    a=tr.parser().parse_args(['--train-data','unused','--output-dir',str(out),'--teacher-model',manifest['teacher_model'],
        '--model',records[0]['checkpoint'],'--method','ren_balanced','--lora-rank','0','--master-weights-fp32','--no-gradient-checkpointing'])
    workers=Workers(600);started=time.monotonic();completed=[]
    def progress(status):tr.atomic_json(out/'live_progress.json',dict(status=status,completed_rollouts=len(completed),total_rollouts=len(records),elapsed_seconds=time.monotonic()-started,updated_utc=time.strftime('%Y-%m-%d %H:%M:%S',time.gmtime())))
    try:
        progress('running');pending=[workers.launch(worker,(a,i,[r for r in records if r['worker']==i],str(out))) for i in range(6)]
        while pending:
            conn,message=workers.receive(pending)
            if message['op']=='done':pending.remove(conn)
            else:assert message['op']=='scored';completed.append(message['uid'])
            progress('running');print(json.dumps(message),flush=True)
        progress('complete')
    except BaseException:progress('failed');raise
    finally:workers.close()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-audit');p.add_argument('--output-dir',required=True);p.add_argument('--run',action='store_true');x=p.parse_args()
    run(x) if x.run else prepare(x)
if __name__=='__main__':main()
