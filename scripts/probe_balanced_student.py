"""Small real-model checks for causal/live equality and vLLM sleep/weight refresh."""
import argparse,copy,json,os,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--source-rollouts',required=True)
    p.add_argument('--output-dir',required=True);p.add_argument('--gpu',default='0');p.add_argument('--mode',choices=['causal','sleep'],default='causal')
    args=p.parse_args();os.environ.update(CUDA_VISIBLE_DEVICES=args.gpu,OMP_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1')
    import torch
    from lulu import training as tr
    torch.set_num_threads(1)
    root=Path(args.output_dir).resolve();root.mkdir(parents=True,exist_ok=True)
    settings=['--train-data','unused','--output-dir',str(root),'--model',args.model,'--lora-rank','0','--master-weights-fp32',
        '--method','ren_balanced','--logit-chunk-size','128','--rollout-backend','vllm','--rollout-vllm-sleep',
        '--rollout-vllm-memory','.30','--rollout-vllm-max-seqs','16','--max-new-tokens','64']
    a=tr.parser().parse_args(settings)
    with open(Path(args.source_rollouts)/'shard-000.jsonl') as f:rows=[json.loads(next(f)),json.loads(next(f))]
    tok=tr.load_tokenizer(a.model)
    if args.mode=='sleep':
        checkpoint=root/'checkpoints/round_000000';checkpoint.mkdir(parents=True,exist_ok=True)
        for src in Path(a.model).iterdir():
            if src.is_file():
                dst=checkpoint/src.name
                if not dst.exists():dst.symlink_to(src.resolve())
        (checkpoint/'lulu_state.json').write_text(json.dumps(dict(completed_rounds=0,completed_updates=0)))
        latest=checkpoint.parent/'latest'
        if not latest.exists():latest.symlink_to(checkpoint.name)
        from lulu.vllm_rollout import RolloutClient
        client=RolloutClient(a,args.gpu,0)
        try:
            records=[dict(index=0,causal_prompt_ids=rows[0]['causal_prompt_ids'])]
            first,_=client.generate(records,0);sleep=client.sleep();second,_=client.generate(records,0);client.sleep()
            result=dict(status='passed',model=a.model,sleep_ack=sleep,first_tokens=len(first[0]),second_tokens=len(second[0]),
                weight_sync=client.last_result.get('weight_sync'),first_weight_refresh_verified=True,
                same_seed_ids_equal=first==second)
            assert first[0] and second[0]
        finally:client.close()
        tr.atomic_json(root/'sleep_results.json',result);print(json.dumps(result),flush=True);return
    for r in rows:
        n=len(r['response_ids']);r['positions']=list(range(min(128,n)))+list(range(max(128,n-128),n))
    model=tr.load_student(a,Path(a.model),True);head=model.get_output_embeddings();frozen=copy.deepcopy(head).requires_grad_(False)
    model.eval();model.gradient_checkpointing_disable()
    with torch.inference_mode():
        cached=[tr.selected_hidden(model,tok,[r],'causal_prompt_ids',a)[0].cpu().clone() for r in rows]
    model.train();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    results=[]
    from torch.utils.checkpoint import checkpoint
    for r,c in zip(rows,cached):
        model.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats()
        with tr.autocast_context(a):
            live=tr.selected_hidden(model,tok,[r],'causal_prompt_ids',a)[0]
            loss=live.sum()*0.;tv=0.;maxerr=0.
            for start in range(0,len(live),128):
                with torch.no_grad():
                    z=frozen(c[start:start+128].cuda()).float();q=z.softmax(-1)
                def kl(states,target):
                    logp=head(states).float().log_softmax(-1)
                    return (target*(target.clamp_min(1e-38).log()-logp)).sum()
                loss=loss+checkpoint(kl,live[start:start+128],q,use_reentrant=False)/len(live)
                # Use the differentiable head first: a no_grad autocast cast cached
                # first would change checkpoint recomputation metadata.
                with torch.no_grad():
                    zz=head(live[start:start+128]).float();maxerr=max(maxerr,float((zz-z).abs().max()))
                    tv+=float((zz.softmax(-1)-q).abs().sum()/2)/len(live)
        loss.backward();norm=sum(float(p.grad.float().norm().double().square()) for p in model.parameters() if p.grad is not None)**.5
        result=dict(index=r['index'],response_tokens=len(r['response_ids']),positions=len(r['positions']),path='eval_batch1',
            kl=float(loss.detach()),tv=tv,max_logit_error=maxerr,noop_gradient_norm=norm,
            peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,model=a.model)
        results.append(result);print(json.dumps(result),flush=True)
        assert tv<=1e-7 and abs(result['kl'])<=1e-7 and norm<=1e-4,result
        del live,loss
    tr.atomic_json(root/'results.json',results)

if __name__=='__main__':main()
