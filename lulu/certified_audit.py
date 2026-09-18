"""Offline probability-space correction audit. Not registered for training."""
from __future__ import annotations
import math
import torch


@torch.no_grad()
def certified_diagnostics(causal_logits,hindsight_logits,teacher_logits,actual_ids=None,
                          epsilon=1e-8,ratio_norm_floor=1e-7):
    values=(causal_logits,hindsight_logits,teacher_logits)
    if causal_logits.ndim!=2 or any(x.shape!=causal_logits.shape or x.device!=causal_logits.device or not x.is_floating_point() for x in values):
        raise ValueError('Need matching floating [position,vocabulary] logits')
    if not math.isfinite(epsilon) or epsilon<=0 or not math.isfinite(ratio_norm_floor) or ratio_norm_floor<=0:
        raise ValueError('Positive finite numerical tolerances required')
    dtype=torch.float64 if any(x.dtype==torch.float64 for x in values) else torch.float32
    c,h,t=[x.detach().to(dtype).log_softmax(-1) for x in values]
    p,ph,q=c.exp(),h.exp(),t.exp()
    ut,uh=q-p,ph-p
    t2=ut.square().sum(-1);h2=uh.square().sum(-1);dot=(ut*uh).sum(-1)
    lam=(dot/(t2+epsilon)).clamp(0,1)
    # Epsilon alone regularizes lambda. No extra norm/sparsity gate changes it.
    target=(1-lam[:,None])*p+lam[:,None]*q
    stable_gradient=-lam[:,None]*ut
    explicit_gradient=p-target
    norm_t=t2.sqrt();norm_h=h2.sqrt()
    valid=norm_t>ratio_norm_floor
    ratio=explicit_gradient.norm(dim=-1)/norm_t.clamp_min(torch.finfo(dtype).tiny)
    stable_ratio=stable_gradient.norm(dim=-1)/norm_t.clamp_min(torch.finfo(dtype).tiny)
    cos_valid=(norm_t>ratio_norm_floor)&(norm_h>ratio_norm_floor)
    cosine=torch.where(cos_valid,dot/(norm_t*norm_h).clamp_min(ratio_norm_floor**2),0).clamp(-1,1)
    movement=(torch.special.xlogy(target,target)-target*c).sum(-1).clamp_min(0)
    movement=torch.where(lam==0,0,movement)
    dc=(q*(t-c)).sum(-1).clamp_min(0);dh=(q*(t-h)).sum(-1).clamp_min(0)
    gap=(q*(h-c)).sum(-1);oldw=gap.clamp_min(0)/(1+gap.clamp_min(0))
    # Independently construct the mixture in FP64 for a cancellation-resistant
    # check. Inputs/coefficients are exactly those from the production calculation.
    p64,q64,lam64=p.double(),q.double(),lam.double()
    target64=(1-lam64[:,None])*p64+lam64[:,None]*q64
    actual64=p64-target64;expected64=lam64[:,None]*(p64-q64)
    norm64=(p64-q64).norm(dim=-1)
    ratio64=actual64.norm(dim=-1)/norm64.clamp_min(torch.finfo(torch.float64).tiny)
    result=dict(lambda_=lam,lambda_unclipped=dot/(t2+epsilon),dot_product=dot,
        teacher_delta_squared=t2,hindsight_delta_squared=h2,probability_cosine=cosine,alignment_valid=cos_valid,
        gradient_ratio_valid=valid,gradient_ratio_fp32=ratio,gradient_ratio_stable=stable_ratio,
        gradient_ratio_fp64=ratio64,gradient_ratio_error_fp64=(ratio64-lam64).abs(),
        gradient_identity_error_fp32=(explicit_gradient-stable_gradient).norm(dim=-1),
        gradient_identity_error_fp64=(actual64-expected64).norm(dim=-1),
        gradient_bound_excess_fp64=(actual64.norm(dim=-1)-norm64).clamp_min(0),
        target_sum_error=(target.sum(-1)-1).abs(),target_min_probability=target.min(-1).values,
        certified_kl=movement,causal_kl=dc,hindsight_kl=dh,resolved_mismatch=gap,
        old_weight=oldw,old_weighted_kl=oldw*dc,
        certified_gradient_norm=stable_gradient.norm(dim=-1),vanilla_gradient_norm=norm_t,
        old_weighted_gradient_norm=oldw*norm_t,
        teacher_actual_logprob=t.gather(1,torch.as_tensor(actual_ids,device=c.device,dtype=torch.long)[:,None]).squeeze(1) if actual_ids is not None else torch.zeros_like(lam),
        causal_entropy=-(p*c).sum(-1),teacher_entropy=-(q*t).sum(-1))
    # Reuse exact full probabilities to quantify epsilon's effect on actual
    # gradient magnitude, not just on an unstable coefficient at tiny norms.
    for label,eps in [('1e-10',1e-10),('1e-6',1e-6),('1e-4',1e-4)]:
        alt=(dot/(t2+eps)).clamp(0,1)
        result['lambda_eps_'+label]=alt
        result['gradient_norm_eps_'+label]=alt*norm_t
    if any(not bool(v.isfinite().all()) for v in result.values()):
        raise FloatingPointError('Nonfinite certified-correction diagnostic')
    return result
