"""Frozen Qwen3 TP inference using local shards and explicit synchronous sums.

Transformers performs native checkpoint sharding once. We then remove its
DTensor forward dispatch: Q/K/V and MLP column shards stay local; attention and
MLP row projections explicitly all-reduce. No full Teacher replica is created.
"""
from __future__ import annotations
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F


class RowParallelLinear(nn.Module):
    def __init__(self, module, group=None):
        super().__init__()
        self.weight=module.weight
        self.bias=module.bias
        self.in_features=self.weight.shape[1]
        self.out_features=self.weight.shape[0]
        self.group=group

    def forward(self, value):
        result=F.linear(value,self.weight,None)
        dist.all_reduce(result,group=self.group,async_op=False)
        return result if self.bias is None else result+self.bias


def materialize_qwen3_local_tp(model, group=None):
    from torch.distributed.tensor import DTensor,Shard
    if model.config.model_type != 'qwen3':
        raise ValueError('Local eager TP currently supports Qwen3 only')
    world=dist.get_world_size(group)
    if world<2 or model.config.num_attention_heads%world or model.config.num_key_value_heads%world:
        raise ValueError('Qwen3 attention and KV heads must divide the TP world')
    row_names=[];sharded=0
    for name,module in model.named_modules():
        if name.endswith(('.self_attn.o_proj','.mlp.down_proj')):
            if not isinstance(module,nn.Linear) or not isinstance(module.weight,DTensor):
                raise ValueError(f'Expected a native row-sharded linear at {name}')
            if module.weight.placements != (Shard(1),):
                raise ValueError(f'Unexpected rowwise placement at {name}: {module.weight.placements}')
            row_names.append(name)
        # Only a dedicated, frozen Teacher is supported (no accelerate hooks,
        # adapters, training, or unrelated application hooks in this model).
        module._forward_pre_hooks.clear();module._forward_hooks.clear()
        module._forward_pre_hooks_with_kwargs.clear();module._forward_hooks_with_kwargs.clear()
        module._forward_hooks_always_called.clear()
        for key,parameter in list(module._parameters.items()):
            if isinstance(parameter,DTensor):
                if any(isinstance(p,Shard) for p in parameter.placements):sharded+=1
                module._parameters[key]=nn.Parameter(parameter.to_local().detach().contiguous(),requires_grad=False)
        for key,buffer in list(module._buffers.items()):
            if isinstance(buffer,DTensor):module._buffers[key]=buffer.to_local().detach().contiguous()
        if isinstance(module,nn.Linear):
            module.out_features,module.in_features=module.weight.shape
    if len(row_names)!=2*model.config.num_hidden_layers:
        raise ValueError('Incomplete Qwen3 attention/MLP row sharding')
    for name in row_names:
        parent,leaf=name.rsplit('.',1)
        setattr(model.get_submodule(parent),leaf,RowParallelLinear(model.get_submodule(name),group))
    if any(isinstance(p,DTensor) for p in model.parameters()):
        raise RuntimeError('DTensor parameters remain in eager Teacher')
    model.requires_grad_(False).eval()
    model._lulu_tp_runtime=dict(mode='eager_local_shards',world=world,
                               row_parallel_layers=len(row_names),sharded_parameters=sharded)
    return model._lulu_tp_runtime
