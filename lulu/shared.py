"""Full-vocabulary, hindsight-supported positive Teacher probability correction.

The target moves at most to the Teacher coordinatewise. Only recipients are
hindsight-certified; donors follow the Teacher. Target TV is not a bound on the
actual neural-network update after parameter sharing, Adam or other losses.
"""
from __future__ import annotations
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from lulu.resolved import concentration
from lulu.stable import prompt_reasoning_scales


@torch.no_grad()
def shared_positive_target(causal, hindsight, teacher, *, return_diagnostics=False):
    """q = C + [min(T,H)-C]+ - (m/M)[C-T]+, detached FP32/FP64.

    Inputs are normalized full-vocabulary probabilities, not logits or top-k
    approximations. There is no epsilon denominator, threshold, scalar gate or
    weight normalization. Clamp the ratio only against floating-point overshoot
    of the mathematical m <= M bound; do not renormalize the resulting target.
    """
    if causal.shape != hindsight.shape or causal.shape != teacher.shape or causal.ndim < 2:
        raise ValueError('C/H/T must have matching [..., vocabulary] probability shapes')
    if any(p.device != causal.device or not p.is_floating_point() for p in (causal,hindsight,teacher)):
        raise ValueError('C/H/T must be floating-point probabilities on the same device')
    dtype = torch.float64 if any(p.dtype == torch.float64 for p in (causal,hindsight,teacher)) else torch.float32
    c,h,t=(p.detach().to(dtype) for p in (causal,hindsight,teacher))
    recipient=(torch.minimum(t,h)-c).clamp_min(0)
    donor=(c-t).clamp_min(0)
    m=recipient.sum(-1);M=donor.sum(-1)
    ratio=torch.where(M>0,m/torch.where(M>0,M,torch.ones_like(M)),torch.zeros_like(M)).clamp(0,1)
    target=c+recipient-ratio.unsqueeze(-1)*donor
    if not return_diagnostics:
        return target
    tiny=torch.finfo(dtype).tiny
    diagnostics=dict(shared_mass=m,teacher_tv=M,shared_fraction=ratio,
        target_tv=.5*(target-c).abs().sum(-1),
        target_causal_kl=(target*(target.clamp_min(tiny).log()-c.clamp_min(tiny).log())).sum(-1),
        teacher_causal_kl=(t*(t.clamp_min(tiny).log()-c.clamp_min(tiny).log())).sum(-1),
        recipient_actions=(recipient>0).sum(-1),
        normalization_error=(target.sum(-1)-1).abs(),
        coordinate_bound_error=torch.maximum((target-torch.maximum(c,t)).clamp_min(0).amax(-1),
                                             (torch.minimum(c,t)-target).clamp_min(0).amax(-1)))
    return target,diagnostics


@torch.no_grad()
def prepare_shared(step, records, a, rank, world):
    """Audit the frozen target once; retain hidden caches, never dense targets.

    C/H/T hidden tensors live on CPU; exact full-vocabulary targets are rebuilt
    in small position chunks during the checkpointed update. This avoids a
    response_length x vocabulary target cache and preserves the round snapshot.
    """
    from lulu import training as tr
    started=time.monotonic();device=tr.device_for(a)
    keys=('shared_mass','teacher_tv','shared_fraction','target_tv','target_causal_kl',
          'teacher_causal_kl','recipient_actions','normalization_error','coordinate_bound_error',
          'trajectory_index','position','truncated')
    pieces={key:[] for key in keys}
    for record in records:
        n=len(record['positions'])
        if any(key not in record for key in ('student_hidden','hindsight_hidden','teacher_hidden','reference_hidden','reasoning_mask')):
            raise ValueError('Shared correction requires frozen C/H/T/reference caches')
        if len(record['reasoning_mask']) != n or record['positions'] != list(range(len(record['response_ids']))):
            raise ValueError('Shared correction must supervise every response position in order')
        mask=torch.tensor(record['reasoning_mask'],dtype=torch.bool)
        for start in range(0,n,a.logit_chunk_size):
            stop=min(n,start+a.logit_chunk_size);keep=mask[start:stop]
            if not keep.any():continue
            with tr.autocast_context(a):
                c=step.frozen_head(record['student_hidden'][start:stop][keep].to(device))
                h=step.frozen_head(record['hindsight_hidden'][start:stop][keep].to(device))
                t=step.teacher_head(record['teacher_hidden'][start:stop][keep].to(device))
            _,scores=shared_positive_target(c.float().softmax(-1),h.float().softmax(-1),t.float().softmax(-1),return_diagnostics=True)
            for key,value in scores.items():pieces[key].append(value.cpu().numpy())
            positions=np.arange(start,stop,dtype=np.int32)[keep.numpy()]
            pieces['position'].append(positions)
            pieces['trajectory_index'].append(np.full(len(positions),record['index'],dtype=np.int32))
            pieces['truncated'].append(np.full(len(positions),bool(record.get('truncated',False)),dtype=bool))
        # Keep C/H: q^ReN cannot be reconstructed from one scalar per state.
    arrays={key:np.concatenate(value) if value else np.empty(0,dtype=np.float32) for key,value in pieces.items()}
    totals=torch.tensor([len(arrays['shared_mass']),sum(len(r['positions']) for r in records),len(records),
        arrays['shared_mass'].sum(dtype=np.float64)],dtype=torch.float64,device=device)
    if world>1:dist.all_reduce(totals)
    reasoning_tokens,loss_tokens,trajectories,mass_sum=totals.tolist()
    prompt_count=prompt_reasoning_scales(records,loss_tokens,a,world)
    split=getattr(a,'reasoning_diagnostic_split',0);horizon={}
    if split:
        values=torch.tensor([[int(sel.sum()),arrays['shared_mass'][sel].sum(dtype=np.float64),
            arrays['target_causal_kl'][sel].sum(dtype=np.float64)]
            for sel in (arrays['position']<split,arrays['position']>=split)],dtype=torch.float64,device=device)
        if world>1:dist.all_reduce(values)
        for name,(count,mass,kl) in zip(('early','late'),values.tolist()):
            # Legacy coordinator names explicitly refer to target KL, never m*KL.
            horizon[name]=dict(reasoning_tokens=int(count),weight_sum=mass,mean_weight=mass/count if count else None,
                mean_shared_mass=mass/count if count else None,mean_target_causal_kl=kl/count if count else None)
    gathered=[None]*world if rank==0 else None
    if world>1:dist.gather_object(arrays,gathered,dst=0)
    elif rank==0:gathered=[arrays]
    diagnostics={}
    if rank==0:
        full={key:np.concatenate([part[key] for part in gathered]) for key in keys}
        if any(not np.isfinite(value).all() for value in full.values()):
            raise FloatingPointError('Non-finite shared correction diagnostics')
        mass_error=float(full['normalization_error'].max(initial=0))
        bound_error=float(full['coordinate_bound_error'].max(initial=0))
        tv_error=float(np.abs(full['target_tv']-full['shared_mass']).max(initial=0))
        if max(mass_error,bound_error,tv_error)>2e-5:
            raise FloatingPointError(f'Shared target invariant failed: mass={mass_error}, bounds={bound_error}, TV={tv_error}')
        diagnostics=dict(round=a.round,before_update=True,method='ren_shared',
            target='pC + [min(qT,pH)-pC]+ - (m/M)[pC-qT]+; M=0 -> pC',
            normalization='reasoning-token mean within rollout, rollout mean within prompt, prompt mean; control/reference global token mean',
            extra_scalar_weight=False,correction_mass_renormalization=False,topk_target_approximation=False,
            reasoning_positions=int(reasoning_tokens),loss_tokens=int(loss_tokens),prompt_count=prompt_count,
            shared_mass_mean=float(full['shared_mass'].mean()) if reasoning_tokens else 0.,
            teacher_tv_mean=float(full['teacher_tv'].mean()) if reasoning_tokens else 0.,
            shared_fraction_mean=float(full['shared_fraction'].mean()) if reasoning_tokens else 0.,
            target_causal_kl_mean=float(full['target_causal_kl'].mean()) if reasoning_tokens else 0.,
            teacher_causal_kl_mean=float(full['teacher_causal_kl'].mean()) if reasoning_tokens else 0.,
            zero_correction_fraction=float(np.mean(full['shared_mass']==0)) if reasoning_tokens else 0.,
            max_normalization_error=mass_error,max_coordinate_bound_error=bound_error,max_tv_identity_error=tv_error,
            reference_kl_coef=a.reference_kl_coef,reasoning_horizon=horizon,
            concentration={key:concentration(full[key]) for key in ('shared_mass','teacher_tv','target_causal_kl')})
        directory=Path(a.output_dir)/'diagnostics'/f'round_{a.round:04d}';directory.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(directory/'position_scores.npz',**full);tr.atomic_json(directory/'summary.json',diagnostics)
    return dict(reasoning_ablation='ren',reasoning_target='shared_positive_probability_correction',
        weight_sum=loss_tokens,extra_reasoning_scalar_weight=False,shared_mass_sum=mass_sum,
        reasoning_horizon=horizon,reasoning_diagnostic_split=split,prompt_count=prompt_count,
        mean_reasoning_tokens_per_rollout=reasoning_tokens/max(trajectories,1),
        reasoning_tokens=int(reasoning_tokens),control_tokens=int(loss_tokens-reasoning_tokens),
        loss_tokens=int(loss_tokens),supervised_trajectories=int(trajectories),
        weight_scoring_seconds=time.monotonic()-started,diagnostics=diagnostics)
