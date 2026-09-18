"""Two-GPU validation on saved real long prefixes; no generation or updates."""
import argparse
import json
from pathlib import Path
import sys
import os
import time
os.environ.pop("TRANSFORMERS_CACHE", None)
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lulu import training as tr
from lulu.persistent import Workers,_port
from lulu.teacher_service import teacher_worker


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--teacher-model',required=True);p.add_argument('--rollouts',required=True);p.add_argument('--model',default='Qwen/Qwen3-1.7B')
    p.add_argument('--idle-seconds',type=float,default=0)
    x=p.parse_args()
    rows=[json.loads(s) for s in Path(x.rollouts).read_text().split('\n') if s]
    a=tr.parser().parse_args(['--train-data','unused','--output-dir',x.output_dir,'--teacher-model',x.teacher_model,
        '--model',x.model,'--method','ren_stable','--lora-rank','0','--master-weights-fp32','--teacher-tp-mode','eager-local','--worker-timeout','180'])
    tr.load_tokenizer(a.model)  # Fail any cache/config issue before allocating Teacher GPUs.
    workers=Workers(180);port=_port()
    try:
        conn=workers.launch(teacher_worker,(a,0,2,['6','7'],'127.0.0.1',port))
        workers.launch(teacher_worker,(a,1,2,['6','7'],'127.0.0.1',port),connected=False)
        ready=workers.expect(conn,'ready');print({k:v for k,v in ready.items() if k!='teacher_head'},flush=True)
        results=[]
        repeated_hidden=None
        longest=max(rows,key=lambda r:len(r['response_ids']))
        shortest=min(rows,key=lambda r:len(r['response_ids']))
        for request_id,selected in enumerate([[longest],[shortest,rows[len(rows)//2]],[longest]]):
            if request_id and x.idle_seconds:time.sleep(x.idle_seconds)
            payload=[{key:r[key] for key in ('causal_prompt_ids','response_ids','positions')} for r in selected]
            conn.send({'op':'score','round':request_id,'request_id':request_id,'records':payload})
            result=workers.expect(conn,'scored')
            for original,record in zip(selected,result['records']):
                h=record['teacher_hidden'];assert h.shape[0]==len(original['response_ids']) and h.isfinite().all()
            if request_id==0:repeated_hidden=result['records'][0]['teacher_hidden'].clone()
            if request_id==2:
                import torch
                torch.testing.assert_close(result['records'][0]['teacher_hidden'],repeated_hidden,rtol=0,atol=0)
            record={'round':request_id,'response_lengths':[len(r['response_ids']) for r in selected],
                    'seconds':result['seconds'],'finite':True,'positions':'all_sampled_tokens'}
            results.append(record);print(record,flush=True)
        tr.atomic_json(Path(x.output_dir)/'result.json',dict(status='passed',runtime=ready['tp_reductions'],control_backend=ready.get('control_backend'),nccl_initialization=ready.get('nccl_initialization'),repeated_long_prefix_identical=True,idle_seconds=x.idle_seconds,results=results))
    finally:workers.close()


if __name__=='__main__':main()
