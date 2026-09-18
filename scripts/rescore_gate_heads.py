"""Reuse C/H/W/T hidden states; obtain signed KL gaps in training BF16 and FP32 heads."""
from pathlib import Path
import argparse,os,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--gpu',default='0');a=p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES']=a.gpu
    import torch
    from lulu import training as tr
    from lulu.semantic_audit import load_output_weight
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    source=Path(a.source);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(x) for x in (source/'records.jsonl').read_text().splitlines()];manifest=json.loads((source/'manifest.json').read_text())
    teacher=load_output_weight(manifest['teacher_model'],'cuda');checkpoint=None;max_errors={'bf16':0.,'fp32':0.}
    for r in rows:
        if checkpoint!=r['checkpoint']:
            if checkpoint is not None:del student
            checkpoint=r['checkpoint'];student=load_output_weight(checkpoint,'cuda')
        cache=tr.load_tensor_file(r['hidden_path']);wrong=tr.load_tensor_file(source/'wrong_hidden'/f'rollout_{r["uid"]:03d}.pt')
        assert cache['positions'].tolist()==wrong['positions'].tolist()==r['positions']
        with np.load(source/'scores'/f'rollout_{r["uid"]:03d}.npz') as z:metadata={k:z[k].copy() for k in z.files}
        certified=Path(r['hidden_path']).parent.parent/'scores'/f'rollout_{r["uid"]:03d}.npz'
        with np.load(certified) as z:old_bf16=z['old_weight'].copy()
        for precision in ['bf16','fp32']:
            chunks={k:[] for k in ['gold_gap','wrong_gap','gold_old_weight','wrong_old_weight','gold_teacher_kl','gold_hindsight_kl','wrong_hindsight_kl']}
            with torch.inference_mode():
                for start in range(0,len(r['positions']),32):
                    stop=start+32
                    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=precision=='bf16'):
                        logits=[torch.nn.functional.linear(cache[k][start:stop].cuda().float(),head) for k,head in [('causal',student),('hindsight',student),('teacher',teacher)]]
                        logits.append(torch.nn.functional.linear(wrong['hidden'][start:stop].cuda().float(),student))
                    c,h,t,w=[v.float().log_softmax(-1) for v in logits];q=t.exp();gold=(q*(h-c)).sum(-1);bad=(q*(w-c)).sum(-1)
                    dc=(q*(t-c)).sum(-1);dh=(q*(t-h)).sum(-1);dw=(q*(t-w)).sum(-1)
                    assert float(((gold-bad)-(dw-dh)).abs().max())<5e-5
                    values=[gold,bad,gold.clamp_min(0)/(1+gold.clamp_min(0)),bad.clamp_min(0)/(1+bad.clamp_min(0)),dc,dh,dw]
                    for k,v in zip(chunks,values):chunks[k].append(v.cpu().numpy())
            data={**metadata,**{k:np.concatenate(v) for k,v in chunks.items()}}
            expected=old_bf16 if precision=='bf16' else metadata['gold_old_weight']
            err=float(np.max(np.abs(data['gold_old_weight']-expected)));max_errors[precision]=max(max_errors[precision],err)
            assert err<2e-5,(precision,r['uid'],err)
            dest=out/precision;dest.mkdir(exist_ok=True);np.savez_compressed(dest/f'rollout_{r["uid"]:03d}.npz',**data)
        tr.atomic_json(out/'live_progress.json',dict(completed_rollouts=r['uid']+1,total_rollouts=48,reconstruction_errors=max_errors))
    tr.atomic_json(out/'summary.json',dict(status='complete',rollouts=48,positions=sum(len(r['positions']) for r in rows),reconstruction_errors=max_errors,new_backbone_forwards=0,new_generations=0))
if __name__=='__main__':main()
