"""Exact hindsight-resolved Teacher mismatch and global-token diagnostics.

No support restriction or target mixture: the Teacher is the only target.
Top-K occurs only in the old-alpha diagnostic, never in the resolved score.
"""
from __future__ import annotations
import csv
import math
from pathlib import Path
import time
import numpy as np
import torch
import torch.distributed as dist


@torch.no_grad()
def mismatch_scores(causal_logits, hindsight_logits, teacher_logits, diagnostic_top_k=32):
    values=(causal_logits,hindsight_logits,teacher_logits)
    if any(x.shape!=causal_logits.shape or x.device!=causal_logits.device or not x.is_floating_point() for x in values):
        raise ValueError('Causal, hindsight and Teacher logits must have matching shapes/devices and floating dtype')
    if causal_logits.ndim<2 or diagnostic_top_k<1:
        raise ValueError('Need position/vocabulary dimensions and a positive diagnostic Top-K')
    dtype=torch.float64 if any(x.dtype==torch.float64 for x in values) else torch.float32
    c,h,t=[x.detach().to(dtype).log_softmax(-1) for x in values]
    q=t.exp()
    dc=(q*(t-c)).sum(-1)
    dh=(q*(t-h)).sum(-1)
    # Algebraically DC-DH; cancel Teacher entropy before floating-point summation.
    resolved=(q*(h-c)).sum(-1)
    ids=hindsight_logits.detach().topk(min(diagnostic_top_k,h.shape[-1]),dim=-1).indices
    alpha=q.gather(-1,ids).sum(-1)
    scores={'causal_kl':dc,'hindsight_kl':dh,'resolved_mismatch':resolved,
            'raw_weight':resolved.clamp_min(0),'old_alpha':alpha}
    if any(not bool(torch.isfinite(x).all()) for x in scores.values()):
        raise FloatingPointError('Non-finite resolved mismatch score')
    return scores


def concentration(values, fractions=(.01,.05,.1,.2,.5,1.)):
    values=np.asarray(values,dtype=np.float64)
    if values.ndim!=1 or np.any(values < -1e-6) or not np.isfinite(values).all():
        raise ValueError('Concentration requires finite nonnegative position scores')
    ordered=np.sort(np.maximum(values,0))[::-1]
    total=float(ordered.sum());cumulative=ordered.cumsum()
    curve={str(f):float(cumulative[min(len(ordered),max(1,math.ceil(f*len(ordered))))-1]/total)
           if total and len(ordered) else 0. for f in fractions}
    return {'positions':len(values),'positive_fraction':float(np.mean(values>0)) if len(values) else 0.,
            'sum':total,'mean':float(values.mean()) if len(values) else 0.,
            'max':float(values.max()) if len(values) else 0.,
            'quantiles':dict(zip(('p50','p90','p99','p999'),np.quantile(values,[.5,.9,.99,.999]).tolist())) if len(values) else {},
            'effective_positions':float(total*total/np.square(values).sum()) if total else 0.,
            'top_fraction_mass':curve}


@torch.no_grad()
def prepare_weights(step_model, records, a, rank, world):
    """Score once per frozen snapshot, then normalize over ALL DDP positions."""
    started=time.monotonic()
    from lulu import training as tr
    device=tr.device_for(a)
    pieces={key:[] for key in ('causal_kl','hindsight_kl','resolved_mismatch','raw_weight','old_alpha','trajectory_index','position')}
    for record in records:
        n=len(record['positions'])
        if 'hindsight_hidden' not in record or 'teacher_hidden' not in record:
            raise ValueError('ren_resolved needs both Hindsight and Teacher hidden states')
        chunks=[]
        for start in range(0,n,a.logit_chunk_size):
            stop=min(n,start+a.logit_chunk_size)
            with tr.autocast_context(a):
                c=step_model.frozen_head(record['student_hidden'][start:stop].to(device))
                h=step_model.frozen_head(record['hindsight_hidden'][start:stop].to(device))
                t=step_model.teacher_head(record['teacher_hidden'][start:stop].to(device))
            scores=mismatch_scores(c,h,t,a.top_k)
            chunks.append(scores['raw_weight'].cpu())
            for key,value in scores.items():pieces[key].append(value.cpu().numpy())
        record['resolved_weight']=torch.cat(chunks) if chunks else torch.empty(0,dtype=torch.float32)
        pieces['trajectory_index'].append(np.full(n,record['index'],dtype=np.int32))
        pieces['position'].append(np.asarray(record['positions'],dtype=np.int32))
        # These two large caches are unnecessary during Teacher-only backward.
        record.pop('hindsight_hidden')
        record.pop('student_hidden')
    arrays={key:np.concatenate(value) if value else np.empty(0,dtype=np.float32) for key,value in pieces.items()}
    totals=torch.tensor([float(arrays['raw_weight'].sum(dtype=np.float64)),len(arrays['raw_weight']),
                         sum(bool(r['positions']) for r in records)],device=device,dtype=torch.float64)
    if world>1:dist.all_reduce(totals)
    weight_sum,positions,active=totals.tolist()
    diagnostics={}
    gathered=[None]*world if rank==0 else None
    if world>1:dist.gather_object(arrays,gathered,dst=0)
    elif rank==0:gathered=[arrays]
    if rank==0:
        full={key:np.concatenate([part[key] for part in gathered]) for key in arrays}
        if len(full['raw_weight'])!=int(positions):raise RuntimeError('Position diagnostics do not match global reduction')
        diagnostics={'round':a.round,'before_update':True,'reasoning_positions':int(positions),
            'weight_sum':weight_sum,'mean_weight':weight_sum/max(positions,1),
            'causal_kl_mean':float(full['causal_kl'].mean()) if positions else 0.,
            'hindsight_kl_mean':float(full['hindsight_kl'].mean()) if positions else 0.,
            'resolved_mean':float(full['resolved_mismatch'].mean()) if positions else 0.,
            'identity_max_abs_error':float(np.max(np.abs(full['resolved_mismatch']-(full['causal_kl']-full['hindsight_kl'])))) if positions else 0.,
            'concentration':{key:concentration(full[key]) for key in ('causal_kl','old_alpha','raw_weight')}}
        path=Path(a.output_dir)/'diagnostics'/f'round_{a.round:04d}';path.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path/'position_scores.npz',**full)
        tr.atomic_json(path/'summary.json',diagnostics)
        with (path/'concentration.csv').open('w') as stream:
            writer=csv.writer(stream);writer.writerow(['top_fraction','causal_kl_mass','old_alpha_mass','resolved_weight_mass'])
            fractions=np.linspace(.01,1,100).tolist()
            curves={key:concentration(full[key],fractions)['top_fraction_mass'] for key in ('causal_kl','old_alpha','raw_weight')}
            for fraction in fractions:writer.writerow([fraction,*[curves[key][str(fraction)] for key in curves]])
    return {'weight_sum':weight_sum,'reasoning_tokens':int(positions),'supervised_trajectories':int(active),
            'weight_scoring_seconds':time.monotonic()-started,'diagnostics':diagnostics}
