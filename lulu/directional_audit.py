"""Offline full-vocabulary projection diagnostics; not a training objective."""
from __future__ import annotations
import math
import numpy as np
import torch


def sample_reasoning_positions(mask, seed, per_bin=64, edges=(0,2048,4096,6144,8192)):
    if per_bin < 1 or edges[0] != 0 or edges[-1] < len(mask):
        raise ValueError('Invalid position sampling budget or horizon')
    rng=np.random.default_rng(seed)
    positions=[]; expansion=[]; bins=[]
    eligible=np.flatnonzero(np.asarray(mask,dtype=bool))
    for b,(lo,hi) in enumerate(zip(edges,edges[1:])):
        population=eligible[(eligible>=lo)&(eligible<hi)]
        if not len(population):continue
        selected=np.sort(rng.choice(population,min(per_bin,len(population)),replace=False))
        positions.extend(selected.tolist());bins.extend([b]*len(selected))
        expansion.extend([len(population)/len(selected)]*len(selected))
    return positions, expansion, bins


@torch.no_grad()
def projection_diagnostics(causal_logits,hindsight_logits,teacher_logits,actual_ids=None,
                           epsilon=1e-8,variance_floor=1e-10):
    """Centered p_C Fisher-metric projection, evaluated at unwarped temperature 1.

    FP32 reductions on production logits; FP64 inputs stay FP64 for analytic tests.
    Degenerate cosine is encoded as zero PLUS a separate validity flag. It must
    be reported separately, not silently interpreted as orthogonality.
    """
    tensors=(causal_logits,hindsight_logits,teacher_logits)
    if not math.isfinite(epsilon) or epsilon<=0 or not math.isfinite(variance_floor) or variance_floor<=0:
        raise ValueError('Positive finite numerical tolerances required')
    if causal_logits.ndim!=2 or any(x.shape!=causal_logits.shape or x.device!=causal_logits.device or not x.is_floating_point() for x in tensors):
        raise ValueError('Expected matching [positions,vocabulary] floating logits')
    dtype=torch.float64 if any(x.dtype==torch.float64 for x in tensors) else torch.float32
    c,h,t=[x.detach().to(dtype).log_softmax(-1) for x in tensors]
    p,q=c.exp(),t.exp()
    dt,dh=t-c,h-c
    dt=dt-(p*dt).sum(-1,keepdim=True)
    dh=dh-(p*dh).sum(-1,keepdim=True)
    vt=(p*dt.square()).sum(-1);vh=(p*dh.square()).sum(-1)
    cov=(p*dt*dh).sum(-1)
    valid=(vt>variance_floor)&(vh>variance_floor)
    cosine=torch.where(valid,cov/(vt*vh).sqrt().clamp_min(variance_floor),0).clamp(-1,1)
    beta_raw=cov/(vt+epsilon)
    beta=torch.where(vt>variance_floor,beta_raw.clamp(0,1),0)
    star=((1-beta[:,None])*c+beta[:,None]*t).log_softmax(-1)
    s=star.exp()
    dc=(q*(t-c)).sum(-1).clamp_min(0)
    hindsight_kl=(q*(t-h)).sum(-1).clamp_min(0)
    gap=(q*(h-c)).sum(-1)
    old_w=gap.clamp_min(0)/(1+gap.clamp_min(0))
    movement=(s*(star-c)).sum(-1).clamp_min(0)
    # Enforce the exact zero-target boundary against softmax roundoff.
    movement=torch.where(beta==0,0,movement)
    ratio_valid=dc>variance_floor
    ratio=torch.where(ratio_valid,movement/dc.clamp_min(variance_floor),0)
    gc,gn=p-q,p-s
    ngc=gc.norm(dim=-1);ngn=gn.norm(dim=-1)
    result=dict(cosine=cosine,alignment_valid=valid,beta=beta,beta_raw=beta_raw,
                covariance=cov,teacher_variance=vt,hindsight_variance=vh,
                causal_kl=dc,hindsight_kl=hindsight_kl,resolved_mismatch=gap,
                old_weight=old_w,old_weighted_kl=old_w*dc,
                projected_kl=movement,movement_ratio=ratio,movement_ratio_valid=ratio_valid,
                causal_entropy=-(p*c).sum(-1),teacher_entropy=-(q*t).sum(-1),
                projected_entropy=-(s*star).sum(-1),
                projected_logit_gradient_norm=torch.where(beta==0,0,ngn),
                vanilla_logit_gradient_norm=ngc,old_weighted_logit_gradient_norm=old_w*ngc,
                projected_vs_teacher_gradient_cosine=torch.where((ngc*ngn)>variance_floor,
                    (gc*gn).sum(-1)/(ngc*ngn).clamp_min(variance_floor),0))
    if actual_ids is not None:
        ids=torch.as_tensor(actual_ids,dtype=torch.long,device=c.device)
        if ids.shape!=(len(c),) or bool(((ids<0)|(ids>=c.shape[-1])).any()):
            raise ValueError('Invalid observed response token IDs')
        result.update(teacher_actual_logprob=t.gather(1,ids[:,None]).squeeze(1),
                      causal_actual_logprob=c.gather(1,ids[:,None]).squeeze(1),
                      hindsight_actual_logprob=h.gather(1,ids[:,None]).squeeze(1))
    if any(not bool(x.isfinite().all()) for x in result.values()):
        raise FloatingPointError('Nonfinite directional diagnostic')
    return result
