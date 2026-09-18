"""Memory controls must preserve ordinary AdamW updates and resumable state."""
from types import SimpleNamespace
from pathlib import Path
import copy
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from lulu.persistent import student_optimizer, consolidate_optimizer
from lulu import training


def _worker(rank, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo',init_method='file://'+rendezvous,rank=rank,world_size=2)
    try:
        torch.manual_seed(42)
        model=torch.nn.Sequential(torch.nn.Linear(3,7),torch.nn.Tanh(),torch.nn.Linear(7,2))
        reference=copy.deepcopy(model)
        args=SimpleNamespace(optimizer_state_sharding=True,learning_rate=1e-3,weight_decay=.01)
        opt=student_optimizer(model,args,2)
        full=torch.optim.AdamW(reference.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay,foreach=False)
        ddp=DDP(model,gradient_as_bucket_view=True)
        for step in range(3):
            x=torch.arange(24,dtype=torch.float32).reshape(8,3)/30+step*.1
            target=torch.arange(16,dtype=torch.float32).reshape(8,2)/17
            opt.zero_grad(set_to_none=True);full.zero_grad(set_to_none=True)
            ((ddp(x[rank::2])-target[rank::2])**2).mean().backward()
            ((reference(x)-target)**2).mean().backward()
            opt.step();full.step()
            for p,q in zip(model.parameters(),reference.parameters()):
                torch.testing.assert_close(p,q,rtol=2e-6,atol=2e-7)
            consolidate_optimizer(opt,args)
            if rank==0:
                state=opt.state_dict()
                torch.save(state,Path(output)/'optimizer.pt')
                for key,values in full.state_dict()['state'].items():
                    for field,value in values.items():
                        torch.testing.assert_close(state['state'][key][field],value,rtol=2e-5,atol=2e-7)
            dist.barrier()
            if step==1:
                # All ranks load the rank-zero complete state; each owns its partition.
                opt=student_optimizer(model,args,2)
                opt.load_state_dict(torch.load(Path(output)/'optimizer.pt',weights_only=False))
        if rank==0: (Path(output)/'passed').write_text('3 matched AdamW steps including a consolidated-state resume')
    finally:dist.destroy_process_group()


def test_sharded_adam_matches_full_and_resumes(tmp_path):
    mp.spawn(_worker,args=(str(tmp_path/'rendezvous'),str(tmp_path)),nprocs=2,join=True)
    assert (tmp_path/'passed').exists()


def test_memory_flags_require_compatible_backend():
    a=training.parser().parse_args(['--train-data','unused','--output-dir','unused','--optimizer-state-sharding'])
    with pytest.raises(ValueError,match='full-parameter'):training.validate_args(a)
    a=training.parser().parse_args(['--train-data','unused','--output-dir','unused','--rollout-vllm-sleep'])
    with pytest.raises(ValueError,match='vLLM rollout'):training.validate_args(a)
