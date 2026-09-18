"""Exact target math, global DDP normalization, zero-step and full-weight refresh."""
import copy
import json
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from lulu import persistent, training
from lulu.resolved import mismatch_scores, concentration, prepare_weights
from lulu.checkpoints import CheckpointManager
from lulu.weight_sync import sync_hindsight_checkpoint, checkpoint_tensors
from lulu.vllm_rollout import snapshot_path
from test_lulu_training import _student, _records
from test_lulu_persistent import _args
from test_lulu_checkpoints import FakeModel, FakeTokenizer, FakeOptimizer


def test_resolved_score_identity_detachment_and_no_topk_dependence():
    torch.manual_seed(33)
    c,h,t=[torch.randn(23,41,dtype=torch.float64,requires_grad=True) for _ in range(3)]
    scores=mismatch_scores(c,h,t,2)
    q=t.softmax(-1)
    expected=(q*(h.log_softmax(-1)-c.log_softmax(-1))).sum(-1)
    torch.testing.assert_close(scores['resolved_mismatch'], expected)
    torch.testing.assert_close(scores['resolved_mismatch'], scores['causal_kl']-scores['hindsight_kl'])
    assert all(not value.requires_grad for value in scores.values())
    torch.testing.assert_close(scores['raw_weight'], mismatch_scores(c,h,t,30)['raw_weight'])
    assert bool((scores['raw_weight']==0).any())
    assert bool((scores['raw_weight']>0).any())
    assert torch.equal(mismatch_scores(c,c,t)['raw_weight'], torch.zeros(23,dtype=torch.float64))


def fixture(root, zero=False):
    model=_student()
    records,head,teacher,tok=_records(copy.deepcopy(model).requires_grad_(False).eval(),'ren_resolved',include_empty=True)
    records=[records[1],records[0],records[2]]
    if zero:
        for r in records:r['hindsight_hidden']=r['student_hidden'].clone()
    args=_args(method='ren_resolved',cpu=True,dtype='float32',global_batch_prompts=3,lora_rank=0,
        max_sequence_tokens=128,max_prompt_tokens=64,learning_rate=1e-4,logit_chunk_size=2,top_k=3,
        output_dir=str(root))
    step=training.DistillationStep(model,head,teacher,tok,args)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=.1)
    return args,model,records,step,optimizer


def ddp_worker(rank, rendezvous, root, zero):
    torch.set_num_threads(1)
    args,model,records,step,opt=fixture(Path(root)/'distributed',zero)
    dist.init_process_group('gloo',init_method=f'file://{rendezvous}',rank=rank,world_size=2)
    try:
        wrapped=DDP(step,broadcast_buffers=False)
        dummy=dict(causal_prompt_ids=[3,4],response_ids=[5],positions=[],snapshot_round=0,dummy=True)
        metrics=persistent.update_records(step,wrapped,opt,records[rank::2],dummy,args,rank,2)
        if rank==0:torch.save({'state':model.state_dict(),'optimizer':opt.state_dict(),'metrics':metrics},Path(root)/'actual.pt')
    finally:dist.destroy_process_group()


@pytest.mark.parametrize('zero',[False,True])
def test_global_token_weighted_ddp_matches_single_batch_and_zero_skips_optimizer(tmp_path,zero):
    args,model,records,step,opt=fixture(tmp_path/'reference',zero)
    original=copy.deepcopy(model.state_dict())
    scores=prepare_weights(step,records,args,0,1)
    if not zero:
        denominator=scores['weight_sum']+args.resolved_weight_epsilon*scores['reasoning_tokens']
        reference=step(records)/denominator
        reference.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),args.max_grad_norm)
        opt.step()
    torch.multiprocessing.spawn(ddp_worker,args=(str(tmp_path/'gloo'),str(tmp_path),zero),nprocs=2,join=True)
    actual=torch.load(tmp_path/'actual.pt',weights_only=False)
    assert actual['metrics']['skipped_update']==zero
    if zero:
        assert not actual['optimizer']['state']
        for name,p in original.items():torch.testing.assert_close(actual['state'][name],p,rtol=0,atol=0)
    else:
        assert actual['metrics']['forward_kl']==pytest.approx(reference.item(),rel=3e-5)
        for index,(name,p) in enumerate(model.named_parameters()):
            torch.testing.assert_close(actual['optimizer']['state'][index]['exp_avg'],opt.state[p]['exp_avg'],atol=6e-9,rtol=5e-4)
            meaningful=p.grad.abs()>1e-6
            torch.testing.assert_close(actual['state'][name][meaningful],p.detach()[meaningful],atol=2e-6,rtol=3e-5)
    assert actual['metrics']['reasoning_tokens']==6


def test_zero_weight_skips_existing_adam_momentum_and_weight_decay(tmp_path):
    args,model,records,step,opt=fixture(tmp_path,True)
    sum(p.sum() for p in model.parameters()).backward();opt.step();opt.zero_grad()
    before=copy.deepcopy(model.state_dict()); optimizer=copy.deepcopy(opt.state_dict())
    result=persistent.update_records(step,step,opt,records,records[0],args,0,1)
    assert result['skipped_update']
    for name,p in model.state_dict().items():torch.testing.assert_close(p,before[name],rtol=0,atol=0)
    for index,state in optimizer['state'].items():
        for name,value in state.items():torch.testing.assert_close(opt.state_dict()['state'][index][name],value,rtol=0,atol=0)


def test_round_checkpoints_resume_even_when_updates_are_skipped(tmp_path):
    manager=CheckpointManager(tmp_path,save_every=20,retain_steps=[2],index_unit='rounds')
    for r,u in [(0,0),(1,1),(2,1),(3,2)]:
        manager.save(FakeModel(),FakeTokenizer(),FakeOptimizer(u),r,{'completed_rounds':r,'completed_updates':u},final=r==3)
    assert sorted(p.name for p in manager.directory.iterdir() if p.is_dir() and not p.is_symlink())==['round_000000','round_000002','round_000003']
    assert manager.latest_metadata()['completed_updates']==2
    assert manager.latest_metadata()['completed_rounds']==3
    with pytest.raises(RuntimeError,match='differs'):snapshot_path(tmp_path,2)
    assert Path(snapshot_path(tmp_path,3)).name=='round_000003'


def test_full_checkpoint_refresh_changes_all_parameters_without_rebuilding_model(tmp_path):
    model=_student(); new=copy.deepcopy(model)
    with torch.no_grad():
        for p in new.parameters():p.add_(.5)
    new.save_pretrained(tmp_path,safe_serialization=True)
    (tmp_path/'lulu_state.json').write_text(json.dumps({'completed_rounds':1}))
    identity=id(model)
    sync_hindsight_checkpoint(model,tmp_path,1)
    assert id(model)==identity
    assert not any(p.requires_grad for p in model.parameters())
    for name,p in model.state_dict().items():torch.testing.assert_close(p,new.state_dict()[name])
    with pytest.raises(ValueError,match='different round'):sync_hindsight_checkpoint(model,tmp_path,0)


def test_concentration_measures_fraction_of_global_positions():
    result=concentration([0,0,0,4])
    assert result['positive_fraction']==.25
    assert result['effective_positions']==1
    assert result['top_fraction_mass']['0.01']==1


def test_native_bf16_teacher_forward_does_not_enter_student_autocast(monkeypatch):
    import contextlib
    calls=[]
    monkeypatch.setattr(training,'autocast_context',lambda a: calls.append(True) or contextlib.nullcontext())
    model=_student().to(torch.bfloat16)
    from types import SimpleNamespace
    args=SimpleNamespace(max_sequence_tokens=128,cpu=False,dtype='bfloat16')
    records=[{'causal_prompt_ids':[3,4],'response_ids':[5,6],'positions':[0,1]}]
    with torch.no_grad():
        training.selected_hidden(model,SimpleNamespace(pad_token_id=0),records,'causal_prompt_ids',args)
    assert not calls
    model.float()
    with torch.no_grad():
        training.selected_hidden(model,SimpleNamespace(pad_token_id=0),records,'causal_prompt_ids',args)
    assert calls==[True]
