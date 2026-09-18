"""Validate local-shard Qwen3 TP against dense logits, including GQA and padding."""
import copy
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Shard
from transformers import Qwen3Config,Qwen3ForCausalLM
from lulu.eager_tp import materialize_qwen3_local_tp


def worker(rank,rendezvous,output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',rank=rank,world_size=2,init_method=f'file://{rendezvous}')
    try:
        torch.manual_seed(71)
        model=Qwen3ForCausalLM(Qwen3Config(vocab_size=31,hidden_size=32,intermediate_size=64,
            num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
            max_position_embeddings=2048,attention_dropout=0.,tie_word_embeddings=False,
            pad_token_id=0,bos_token_id=1,eos_token_id=2)).eval().requires_grad_(False)
        dense=copy.deepcopy(model)
        mesh=init_device_mesh('cpu',(2,))
        row=('.self_attn.o_proj','.mlp.down_proj')
        col=('.self_attn.q_proj','.self_attn.k_proj','.self_attn.v_proj','.mlp.gate_proj','.mlp.up_proj')
        for name,module in model.named_modules():
            if name.endswith(row+col):
                axis=1 if name.endswith(row) else 0
                module.weight=torch.nn.Parameter(distribute_tensor(module.weight.detach(),mesh,[Shard(axis)]),requires_grad=False)
                # Reproduce the native hook presence; materialization must remove it.
                module.register_forward_pre_hook(lambda *x: (_ for _ in ()).throw(AssertionError('native hook survived')))
        info=materialize_qwen3_local_tp(model)
        assert info['row_parallel_layers']==4 and info['sharded_parameters']==14
        assert sum(p.numel() for p in model.parameters())<sum(p.numel() for p in dense.parameters())
        errors=[]
        for batch,length in [(2,31),(1,1024),(2,17)]:
            ids=torch.arange(batch*length).reshape(batch,length)%29+1
            mask=torch.ones_like(ids)
            if batch>1:mask[-1,-5:]=0;ids[-1,-5:]=0
            with torch.inference_mode():
                actual=model(input_ids=ids,attention_mask=mask,use_cache=False).logits
                expected=dense(input_ids=ids,attention_mask=mask,use_cache=False).logits
            torch.testing.assert_close(actual,expected,atol=3e-7,rtol=2e-5)
            errors.append(float((actual-expected).abs().max()))
        if rank==0:torch.save({'errors':errors,'runtime':info},output)
    finally:dist.destroy_process_group()


def test_local_tp_gqa_logits_equal_dense_across_shapes_and_padding(tmp_path):
    torch.multiprocessing.spawn(worker,args=(str(tmp_path/'rendezvous'),str(tmp_path/'result.pt')),nprocs=2,join=True)
    data=torch.load(tmp_path/'result.pt',weights_only=True)
    assert max(data['errors'])<3e-7
