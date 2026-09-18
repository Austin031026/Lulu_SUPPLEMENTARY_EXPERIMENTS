"""Infrequent exact full-batch component gradients; no extra model copies."""
from __future__ import annotations
import contextlib
import time
import torch
import torch.distributed as dist


def sum_gradients(parameters, world, bucket_elements=8_000_000):
    """All-reduce summed local gradients using bounded temporary buffers."""
    if world == 1:
        return
    # Split even the large embedding matrix, rather than allocating another
    # full 1.7B-gradient vector alongside the resident rollout engine.
    for parameter in parameters:
        if parameter.grad is None:
            parameter.grad=torch.zeros_like(parameter)
        flat=parameter.grad.reshape(-1)
        for start in range(0,flat.numel(),bucket_elements):
            dist.all_reduce(flat[start:start+bucket_elements])


def component_gradient_norms(step, distributed, optimizer, records, args, world, denominator):
    """Exact gradient of each global objective term, before clipping/Adam.

    Reuse frozen targets and weights. Serial backward passes keep only one
    component gradient resident. Control/reference norms include their coefficients.
    """
    started=time.monotonic();parameters=[p for p in step.student.parameters() if p.requires_grad]
    norms={};archives={};dots={}
    pairwise=getattr(args,'gradient_cosines',False)
    primary=world==1 or dist.get_rank()==0
    try:
        components=[(0,'reasoning'),(1,'control'),(2,'reference')]
        if getattr(args,'reasoning_diagnostic_split',0):
            components.extend([(3,'reasoning_early'),(4,'reasoning_late')])
        for index,label in components:
            optimizer.zero_grad(set_to_none=True)
            step.loss_component_mode=index
            for start in range(0,len(records),args.train_micro_batch_size):
                batch=records[start:start+args.train_micro_batch_size]
                context=distributed.no_sync() if world>1 else contextlib.nullcontext()
                with context:
                    (distributed(batch)/denominator).backward()
            sum_gradients(parameters,world)
            squared=torch.zeros((),device=parameters[0].device,dtype=torch.float64)
            for p in parameters:
                if p.grad is not None:
                    # FP32 norm computation; accumulate squared norms in FP64.
                    squared+=p.grad.float().norm().double().square()
            norms[label]=float(squared.sqrt())
            if pairwise and primary and label in ('reasoning','control','reference'):
                for previous,saved in archives.items():
                    dot=torch.zeros((),device=parameters[0].device,dtype=torch.float64)
                    for parameter,old in zip(parameters,saved):
                        if parameter.grad is None or old is None:continue
                        flat=parameter.grad.detach().reshape(-1)
                        for start in range(0,flat.numel(),2_000_000):
                            now=flat[start:start+2_000_000].double()
                            before=old[start:start+2_000_000].to(device=now.device,dtype=torch.float64)
                            dot+=torch.dot(now,before)
                    dots[previous+'__'+label]=float(dot)
                if label!='reference':
                    archives[label]=[p.grad.detach().reshape(-1).to(device='cpu',copy=True) if p.grad is not None else None for p in parameters]
    finally:
        step.loss_component_mode=None
        optimizer.zero_grad(set_to_none=True)
    archives.clear()
    if pairwise and world>1:
        payload=[dots if primary else None];dist.broadcast_object_list(payload,src=0);dots=payload[0]
    cosines={}
    for name,value in dots.items():
        first,second=name.split('__');den=norms[first]*norms[second]
        cosines[name]=max(-1.,min(1.,value/den)) if den>1e-20 else None
    alignment={'component_gradient_dots':dots,'component_gradient_cosines':cosines} if pairwise else {}
    if 'reasoning_early' in norms:
        early,late=norms['reasoning_early'],norms['reasoning_late']
        dot=(norms['reasoning']**2-early**2-late**2)/2
        alignment.update({'reasoning_gradient_early_late_dot':dot,
            'reasoning_gradient_early_late_cosine':max(-1.,min(1.,dot/(early*late))) if early*late else None,
            'reasoning_gradient_normalization':'Each bin retains full-rollout reasoning denominator and original prompt/rollout weights; norms are global after all-reduce, before clipping/Adam'})
    return {**alignment,'component_gradient_norms':norms,'component_gradient_norm_seconds':time.monotonic()-started,
            'component_gradient_norm_scope':'exact full global batch, all trainable parameters, weighted control/reference terms, before clipping/Adam'}
