#!/usr/bin/env python3
"""Measure Student policy drift on a frozen set of causal rollout prefixes.

Designed for RQ2.3 panel (d).  It never generates new text: positions are
sampled deterministically from saved round-0 rollout token IDs, then every
retained Student checkpoint is teacher-forced on exactly those prefixes.
"""
from __future__ import annotations
import argparse,csv,json,random
from pathlib import Path
from types import SimpleNamespace
import torch


def sample_records(paths,max_trajectories=16,positions_per_bin=8,seed=42,bin_size=2048):
    rows=[]
    for path in paths:
        rows.extend(json.loads(x) for x in Path(path).read_text().splitlines() if x.strip())
    rows=sorted(rows,key=lambda x:str(x.get('source_id',x.get('index'))))
    random.Random(seed).shuffle(rows);rows=rows[:max_trajectories];out=[]
    for row in rows:
        reasoning=row.get('reasoning_mask',[True]*len(row['response_ids']));chosen=[]
        for low in range(0,len(reasoning),bin_size):
            candidates=[i for i in range(low,min(low+bin_size,len(reasoning))) if reasoning[i]]
            if len(candidates)>positions_per_bin:
                rng=random.Random(f'{seed}:{row.get("source_id",row.get("index"))}:{low}')
                candidates=sorted(rng.sample(candidates,positions_per_bin))
            chosen.extend(candidates)
        if chosen:
            out.append({'source_id':row.get('source_id',row.get('index')),'causal_prompt_ids':row['causal_prompt_ids'],
                        'response_ids':row['response_ids'],'positions':chosen})
    return out


def score_checkpoint(path,records,device,dtype,max_sequence_tokens,chunk=64):
    from transformers import AutoModelForCausalLM,AutoTokenizer
    from lulu.training import selected_hidden,base_model,disable_dropout
    model=AutoModelForCausalLM.from_pretrained(path,torch_dtype=dtype,low_cpu_mem_usage=True).to(device).eval();disable_dropout(model)
    tok=AutoTokenizer.from_pretrained(path);head=base_model(model).get_output_embeddings();args=SimpleNamespace(max_sequence_tokens=max_sequence_tokens,dtype='bfloat16' if dtype==torch.bfloat16 else 'float32',cpu=False)
    values=[]
    with torch.inference_mode():
        for record in records:
            hidden=selected_hidden(model,tok,[record],'causal_prompt_ids',args)[0]
            chunks=[]
            for start in range(0,len(hidden),chunk):
                logits=head(hidden[start:start+chunk]).float();chunks.append(logits.log_softmax(-1).cpu())
            values.append(torch.cat(chunks) if chunks else torch.empty((0,head.weight.shape[0])))
    del model;torch.cuda.empty_cache();return values


def main():
    p=argparse.ArgumentParser();p.add_argument('--base-checkpoint',required=True);p.add_argument('--checkpoint',action='append',default=[],metavar='NAME=PATH');p.add_argument('--rollout',action='append',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--gpu',default='0');p.add_argument('--max-trajectories',type=int,default=16);p.add_argument('--positions-per-bin',type=int,default=8);p.add_argument('--seed',type=int,default=42);p.add_argument('--max-sequence-tokens',type=int,default=16384);a=p.parse_args()
    records=sample_records(a.rollout,a.max_trajectories,a.positions_per_bin,a.seed);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);(out/'audit_records.json').write_text(json.dumps(records,indent=2)+'\n')
    device=torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu');dtype=torch.bfloat16 if device.type=='cuda' else torch.float32
    base=score_checkpoint(a.base_checkpoint,records,device,dtype,a.max_sequence_tokens)
    rows=[]
    for item in a.checkpoint:
        name,sep,path=item.partition('=')
        if not sep:raise ValueError('checkpoint must be NAME=PATH')
        current=score_checkpoint(path,records,device,dtype,a.max_sequence_tokens)
        fwd=[];rev=[];tv=[]
        for b,c in zip(base,current):
            pb=b.exp();pc=c.exp();fwd.extend((pc*(c-b)).sum(-1).tolist());rev.extend((pb*(b-c)).sum(-1).tolist());tv.extend((.5*(pc-pb).abs().sum(-1)).tolist())
        rows.append({'checkpoint':name,'positions':len(fwd),'kl_checkpoint_to_base':sum(fwd)/len(fwd),'kl_base_to_checkpoint':sum(rev)/len(rev),'tv':sum(tv)/len(tv)})
    with (out/'policy_drift.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    (out/'summary.json').write_text(json.dumps({'records':len(records),'checkpoints':rows},indent=2)+'\n')
if __name__=='__main__':main()
