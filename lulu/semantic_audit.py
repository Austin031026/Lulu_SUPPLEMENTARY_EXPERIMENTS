"""Read-only outcome-specificity and matched continuation audit helpers."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import torch


def stable_seed(*values):
    return int.from_bytes(hashlib.sha256(':'.join(map(str,values)).encode()).digest()[:4],'big')


def load_output_weight(checkpoint,device='cpu'):
    from safetensors import safe_open
    path=Path(checkpoint);config=json.loads((path/'config.json').read_text())
    index=path/'model.safetensors.index.json'
    mapping=json.loads(index.read_text())['weight_map'] if index.exists() else None
    keys=['lm_head.weight']
    if config.get('tie_word_embeddings'):keys.append('model.embed_tokens.weight')
    if mapping:
        key=next((key for key in keys if key in mapping),None)
        if key is None:raise ValueError(f'No output head or tied embedding in {checkpoint}')
        filename=mapping[key]
        if Path(filename).name!=filename:raise ValueError('Unsafe shard path')
        with safe_open(path/filename,framework='pt',device='cpu') as f:weight=f.get_tensor(key)
    else:
        with safe_open(path/'model.safetensors',framework='pt',device='cpu') as f:
            key=next((key for key in keys if key in f.keys()),None)
            if key is None:raise ValueError('Missing output weight')
            weight=f.get_tensor(key)
    return weight.to(device=device,dtype=torch.float32).contiguous()


@torch.no_grad()
def probability_probe(causal_logits,hindsight_logits,teacher_logits,epsilon=1e-8):
    c,h,t=[v.float().log_softmax(-1) for v in [causal_logits,hindsight_logits,teacher_logits]]
    p,ph,q=c.exp(),h.exp(),t.exp();ut,uh=q-p,ph-p
    t2=ut.square().sum(-1);dot=(ut*uh).sum(-1)
    lam=(dot/(t2+epsilon)).clamp(0,1)
    gap=(q*(h-c)).sum(-1);raw=gap.clamp_min(0)
    return dict(lambda_=lam,dot_product=dot,teacher_delta_squared=t2,
        teacher_kl=(q*(t-c)).sum(-1).clamp_min(0),student_entropy=-(p*c).sum(-1),
        student_max_probability=p.max(-1).values,teacher_tv=(q-p).abs().sum(-1)*.5,
        gradient_norm=lam*t2.sqrt(),old_weight=raw/(1+raw))


def residual_coupling(p,q,eta=.2):
    """Exact residual decomposition for P versus (1-eta)P+eta Q.

    Common mass cancels in the treatment effect under identical continuation
    kernels. Returns changed mass and the two normalized conditional residuals.
    """
    if not 0<eta<=1:raise ValueError('eta must be in (0,1]')
    p=np.asarray(p,dtype=np.float64);q=np.asarray(q,dtype=np.float64)
    if p.ndim!=1 or p.shape!=q.shape or not np.isfinite(p).all() or not np.isfinite(q).all() or np.any(p<0) or np.any(q<0) or p.sum()<=0 or q.sum()<=0:raise ValueError('Invalid categorical distributions')
    p=p/p.sum();q=q/q.sum();minus=np.maximum(p-q,0);plus=np.maximum(q-p,0)
    tv=.5*np.abs(p-q).sum()
    if tv==0:return 0.,np.zeros_like(p),np.zeros_like(p)
    return float(eta*tv),minus/minus.sum(),plus/plus.sum()


def sample_categorical(p,u):
    if not 0<=u<1:raise ValueError('Uniform variate must lie in [0,1)')
    cdf=np.cumsum(np.asarray(p,dtype=np.float64));cdf/=cdf[-1]
    return min(int(np.searchsorted(cdf,u,side='right')),len(p)-1)


def continuation_budget(causal_prompt_length,position,forced_tokens=1,total_response_horizon=32768,
                        max_model_len=40960,safety_margin=128):
    consumed=position+forced_tokens
    available=min(total_response_horizon-consumed,max_model_len-causal_prompt_length-consumed-safety_margin)
    if position<0 or available<1:raise ValueError('No continuation budget; never truncate the prompt')
    return available
