"""Exact full-parameter component angles on one fixed saved batch, no updates/generation."""
from pathlib import Path
import argparse,json,os,sys,hashlib,time,copy,traceback
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lulu import training as tr
from lulu.data import build_prompt_views
from lulu.persistent import Workers,_port,head_from_state,teacher_payload
from lulu.teacher_service import teacher_worker

def prepare(source,out):
    config=json.loads((source/'train/run_config.json').read_text());base=source/'train/checkpoints/round_000000'
    rows=[]
    for p in sorted((source/'train/rollouts/round_0000').glob('shard-*.jsonl')):rows.extend(json.loads(x) for x in p.read_text().splitlines())
    rows=sorted(rows,key=lambda r:hashlib.sha256(('cosine:20260917:'+r['source_id']).encode()).hexdigest())[:16]
    pool={r['id']:r for r in [json.loads(x) for x in Path(config['train_data']).read_text().splitlines()]};tok=tr.load_tokenizer(str(base))
    selected=[]
    for i,r in enumerate(rows):
        row=pool[r['source_id']];views=build_prompt_views(tok,row['messages'],row['gold_answer'])
        assert views['causal_prompt_ids']==r['causal_prompt_ids']
        selected.append(dict(r,hindsight_prompt_ids=views['hindsight_prompt_ids'],uid=i,teacher_scored=True))
    out.mkdir(parents=True,exist_ok=True);path=out/'records.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in selected))
    models=[dict(round=n,checkpoint=str(source/f'train/checkpoints/round_{n:06d}')) for n in [0,4,8,12]]
    for m in models:assert Path(m['checkpoint'],'lulu_state.json').exists()
    tr.atomic_json(out/'manifest.json',dict(source=str(source),base=str(base),teacher_model=config['teacher_model'],train_data=config['train_data'],models=models,
        records_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),trajectories=16,tokens=sum(len(r['response_ids']) for r in selected),
        scope='Same fixed Base-generated 16-rollout audit batch at saved post-update checkpoints 0/4/8/12. Not reconstruction of historical pre-update batches at steps1/4/8/12.',
        selection='lowest SHA256 cosine:20260917:source_id within all64 first-round rollouts; independent of correctness, length, score and outcomes',
        normalization='all16 prompts equally; reasoning-token mean within each rollout; existing global-token control/reference',optimizer_steps=0,new_rollouts=0))

def reference_worker(a,gpu,records,out,conn):
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),RANK='0',LOCAL_RANK='0',WORLD_SIZE='1');torch.set_num_threads(1)
        model=tr.load_student(a,checkpoint_dir=Path(a.model),trainable=True);model.gradient_checkpointing_disable();model.requires_grad_(False).eval();tr.disable_dropout(model)
        tok=tr.load_tokenizer(a.model);dest=Path(out)/'reference';dest.mkdir(exist_ok=True)
        with torch.no_grad():
            for r in records:tr.atomic_torch(dest/f'{r["uid"]:03d}.pt',tr.selected_hidden(model,tok,[r],'causal_prompt_ids',a)[0].cpu())
        conn.send(dict(op='ready',role='reference',head=tr.head_state(model)))
        assert conn.recv()['op']=='stop'
    except BaseException as exc:conn.send(dict(op='error',role='reference',error=str(exc),traceback=traceback.format_exc()));raise
    finally:conn.close()

def student_worker(a,gpu,checkpoint,round_,records,out,conn):
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu),RANK='0',LOCAL_RANK='0',WORLD_SIZE='1');torch.set_num_threads(1)
        a=copy.copy(a);a.round=round_;a.output_dir=str(Path(out)/f'checkpoint_{round_:02d}');a.gradient_cosines=True
        model=tr.load_student(a,checkpoint_dir=Path(checkpoint),trainable=True);model.gradient_checkpointing_disable();tr.disable_dropout(model);model.eval();tok=tr.load_tokenizer(checkpoint)
        head=copy.deepcopy(model.get_output_embeddings()).requires_grad_(False);cache=copy.deepcopy(records)
        with torch.no_grad():
            for r in cache:
                r['student_hidden']=tr.selected_hidden(model,tok,[r],'causal_prompt_ids',a)[0].cpu()
                r['hindsight_hidden']=tr.selected_hidden(model,tok,[r],'hindsight_prompt_ids',a)[0].cpu()
        conn.send(dict(op='ready',role='student',round=round_))
        message=conn.recv();assert message['op']=='compute'
        teacher=head_from_state(message['teacher_head'],torch.device('cuda'));reference=head_from_state(message['reference_head'],torch.device('cuda'))
        for r in cache:
            r['teacher_hidden']=tr.load_tensor_file(Path(out)/'teacher'/f'{r["uid"]:03d}.pt')
            r['reference_hidden']=tr.load_tensor_file(Path(out)/'reference'/f'{r["uid"]:03d}.pt')
        step=tr.DistillationStep(model,head,teacher,tok,a);step.reference_head=reference
        from lulu.stable import prepare_weights
        from lulu.gradient_diagnostics import component_gradient_norms
        stats=prepare_weights(step,cache,a,0,1);model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        optimizer=torch.optim.SGD(model.parameters(),lr=1e-6)
        result=component_gradient_norms(step,step,optimizer,cache,a,1,sum(len(r['positions']) for r in cache))
        result.update(checkpoint=checkpoint,completed_updates=round_,audit_trajectories=len(cache),reasoning_tokens=stats['reasoning_tokens'],mean_ren_weight=stats['rho_sum']/stats['reasoning_tokens'],optimizer_steps=0)
        assert not optimizer.state
        tr.atomic_json(Path(out)/f'cosines_{round_:02d}.json',result);conn.send(dict(op='complete',round=round_,result=result))
        assert conn.recv()['op']=='stop'
    except BaseException as exc:conn.send(dict(op='error',role='cosine_student',error=str(exc),traceback=traceback.format_exc()));raise
    finally:conn.close()

def run(out,gpus):
    assert len(gpus)==7 and len(set(gpus))==7, "Need seven distinct idle GPUs"
    manifest=json.loads((out/'manifest.json').read_text());assert hashlib.sha256((out/'records.jsonl').read_bytes()).hexdigest()==manifest['records_sha256']
    records=[json.loads(s) for s in (out/'records.jsonl').read_text().splitlines()]
    a=tr.parser().parse_args(['--train-data',manifest['train_data'],'--output-dir',str(out),'--model',manifest['base'],'--teacher-model',manifest['teacher_model'],
        '--method','ren_balanced','--lora-rank','0','--master-weights-fp32','--score-batch-size','1','--train-micro-batch-size','1','--logit-chunk-size','128',
        '--max-new-tokens','16384','--max-prompt-tokens','4096','--max-sequence-tokens','20480','--teacher-tp-mode','eager-local','--gradient-norm-every','1','--gradient-cosines'])
    tr.validate_args(a);workers=Workers(3600);began=time.monotonic()
    def progress(phase,**kw):tr.atomic_json(out/'live_progress.json',dict(status=phase,elapsed_seconds=time.monotonic()-began,**kw))
    try:
        progress('loading_and_student_scoring');port=_port();teacher=workers.launch(teacher_worker,(a,0,2,gpus[5:],'127.0.0.1',port));workers.launch(teacher_worker,(a,1,2,gpus[5:],'127.0.0.1',port),connected=False)
        ref=workers.launch(reference_worker,(a,gpus[4],records,str(out)))
        students=[workers.launch(student_worker,(a,gpus[i],m['checkpoint'],m['round'],records,str(out))) for i,m in enumerate(manifest['models'])]
        ready=workers.expect(teacher,'ready');teacher_head=ready['teacher_head'];(out/'teacher').mkdir(exist_ok=True)
        for i,r in enumerate(records):
            teacher.send(dict(op='score',round=0,request_id=r['index'],records=teacher_payload([r])))
            value=workers.expect(teacher,'scored')['records'][0]['teacher_hidden'];tr.atomic_torch(out/'teacher'/f'{r["uid"]:03d}.pt',value)
            progress('teacher_scoring',completed=i+1,total=len(records))
        reference_head=workers.expect(ref,'ready')['head']
        for conn in students:workers.expect(conn,'ready');conn.send(dict(op='compute',teacher_head=teacher_head,reference_head=reference_head))
        pending=list(students);results=[]
        while pending:
            conn,result=workers.receive(pending);assert result['op']=='complete';pending.remove(conn);results.append(result['result']);progress('gradient_cosines',completed_checkpoints=len(results),total_checkpoints=4)
        tr.atomic_json(out/'summary.json',dict(status='complete',scope=manifest['scope'],elapsed_seconds=time.monotonic()-began,results=sorted(results,key=lambda x:x['completed_updates'])))
        progress('complete')
    finally:workers.close()

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment');p.add_argument('--output-dir',required=True);p.add_argument('--run',action='store_true');p.add_argument('--gpus',default='0,1,2,3,5,6,7');a=p.parse_args();out=Path(a.output_dir).resolve()
    if a.run:run(out,a.gpus.split(','))
    else:prepare(Path(a.experiment).resolve(),out)
if __name__=='__main__':main()
