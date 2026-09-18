"""Probability invariants and the actual checkpointed, prompt-balanced update."""
import copy
from pathlib import Path
import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from lulu import training,persistent
from lulu.shared import shared_positive_target
from lulu.stable import prepare_weights
from test_balanced import balanced_fixture


def test_hand_computed_recipients_donors_and_logit_gradient():
    c=torch.tensor([[.4,.3,.2,.1]],dtype=torch.float64,requires_grad=True)
    h=torch.tensor([[.3,.2,.25,.25]],dtype=torch.float64,requires_grad=True)
    t=torch.tensor([[.2,.2,.3,.3]],dtype=torch.float64,requires_grad=True)
    q,s=shared_positive_target(c,h,t,return_diagnostics=True)
    expected=torch.tensor([[4/15,7/30,.25,.25]],dtype=torch.float64)
    torch.testing.assert_close(q,expected)
    assert s['shared_mass'].item()==pytest.approx(.2)
    assert s['teacher_tv'].item()==pytest.approx(.3)
    assert s['target_tv'].item()==pytest.approx(.2)
    assert not q.requires_grad
    logits=c.detach().log().requires_grad_()
    loss=(q*(q.log()-logits.log_softmax(-1))).sum();loss.backward()
    torch.testing.assert_close(logits.grad,c.detach()-q,atol=1e-14,rtol=1e-12)
    assert all(p.grad is None for p in (c,h,t))


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_target_invariants_random_full_vocabulary(dtype):
    gen=torch.Generator().manual_seed(55)
    c,h,t=[torch.randn(37,1009,generator=gen,dtype=dtype).softmax(-1) for _ in range(3)]
    q,s=shared_positive_target(c,h,t,return_diagnostics=True)
    atol=4e-7 if dtype==torch.float32 else 1e-14
    torch.testing.assert_close(q.sum(-1),torch.ones(37,dtype=dtype),atol=atol,rtol=0)
    assert bool((q>=0).all())
    assert bool((q>=torch.minimum(c,t)-atol).all()) and bool((q<=torch.maximum(c,t)+atol).all())
    increase=(q-c).clamp_min(0)
    torch.testing.assert_close(increase,torch.minimum((t-c).clamp_min(0),(h-c).clamp_min(0)),atol=atol,rtol=1e-5)
    torch.testing.assert_close(s['target_tv'],s['shared_mass'],atol=atol,rtol=1e-5)
    torch.testing.assert_close(shared_positive_target(c,c,t),c,atol=0,rtol=0)
    torch.testing.assert_close(shared_positive_target(c,h,c),c,atol=0,rtol=0)
    torch.testing.assert_close(shared_positive_target(c,t,t),t,atol=atol,rtol=1e-5)


def test_tiny_mass_is_not_epsilon_attenuated_and_zero_probabilities_stay_finite():
    c=torch.tensor([[.5,.5,0.],[1.,0.,0.]],dtype=torch.float64)
    t=torch.tensor([[.5-1e-12,.5+1e-12,0.],[0.,1.,0.]],dtype=torch.float64)
    q,s=shared_positive_target(c,t,t,return_diagnostics=True)
    torch.testing.assert_close(q,t,atol=1e-15,rtol=0)
    assert s['shared_fraction'][0]==pytest.approx(1.)
    assert all(torch.isfinite(x).all() for x in s.values())


def shared_fixture(root):
    a,model,records,step,opt=balanced_fixture(root)
    a.method='ren_shared';a.reasoning_diagnostic_split=2;a.gradient_cosines=True
    return a,model,records,step,opt


def dense_components(step,records,a):
    from collections import Counter
    counts=Counter(r['source_id'] for r in records);n=sum(len(r['positions']) for r in records)
    states=training.selected_hidden(step.student,step.tok,records,'causal_prompt_ids',a)
    zero=states[0].sum()*0.;reason,control,reference,early,late=zero,zero,zero,zero,zero
    for r,h in zip(records,states):
        logits=step.student.get_output_embeddings()(h);logp=logits.log_softmax(-1)
        with torch.no_grad():
            c=step.frozen_head(r['student_hidden']).softmax(-1)
            h=step.frozen_head(r['hindsight_hidden']).softmax(-1)
            t=step.teacher_head(r['teacher_hidden']).softmax(-1)
            # Independent direct formula, no production target helper.
            increase=torch.minimum((t-c).clamp_min(0),(h-c).clamp_min(0))
            decrease=(c-t).clamp_min(0);mass=increase.sum(-1,keepdim=True)
            q=c+increase-mass*decrease/decrease.sum(-1,keepdim=True)
            logref=step.reference_head(r['reference_hidden']).log_softmax(-1)
        mask=torch.tensor(r['reasoning_mask'],dtype=torch.bool)
        if mask.sum():
            losses=(q*(q.log()-logp)).sum(-1)*mask/(len(counts)*counts[r['source_id']]*mask.sum())
            reason=reason+losses.sum();early=early+losses[:2].sum();late=late+losses[2:].sum()
        control=control+((t*(t.log()-logp)).sum(-1)*(~mask)).sum()/n
        reference=reference+(logp.exp()*(logp-logref)).sum()/n*a.reference_kl_coef
    return torch.stack((reason,control,reference,early,late))


def shared_worker(rank,rendezvous,root):
    a,model,records,step,opt=shared_fixture(Path(root)/'ddp')
    dist.init_process_group('gloo',init_method=f'file://{rendezvous}',rank=rank,world_size=2)
    try:
        wrapper=DDP(step,broadcast_buffers=False,gradient_as_bucket_view=True)
        dummy=dict(causal_prompt_ids=[3,4],response_ids=[2],positions=[],snapshot_round=0,dummy=True)
        metrics=persistent.update_records(step,wrapper,opt,records[rank::2],dummy,a,rank,2)
        if rank==0:torch.save(dict(state=model.state_dict(),metrics=metrics),Path(root)/'actual.pt')
    finally:dist.destroy_process_group()


def test_chunked_backward_uneven_ddp_matches_independent_dense_objective(tmp_path):
    a,model,records,step,opt=shared_fixture(tmp_path/'dense')
    expected=dense_components(step,records,a)
    before=copy.deepcopy(model.state_dict())
    frozen=[{k:v.clone() for k,v in head.state_dict().items()} for head in (step.frozen_head,step.teacher_head,step.reference_head)]
    stats=prepare_weights(step,records,a,0,1)
    assert records[0]['reasoning_loss_scale']==pytest.approx(9/12)
    assert records[1]['reasoning_loss_scale']==pytest.approx(9/8)
    assert records[2]['reasoning_loss_scale']==0
    assert not stats['extra_reasoning_scalar_weight']
    assert all('student_hidden' in r and 'hindsight_hidden' in r for r in records)
    torch.testing.assert_close(step(records)/9,expected[:3].sum(),atol=3e-7,rtol=3e-5)
    norms={}
    for label,value in zip(('reasoning','control','reference','reasoning_early','reasoning_late'),expected):
        gradients=torch.autograd.grad(value,list(model.parameters()),retain_graph=True,allow_unused=True)
        norms[label]=sum(float(g.double().square().sum()) for g in gradients if g is not None)**.5
    expected[:3].sum().backward();opt.step()
    assert any(not torch.equal(v,before[k]) for k,v in model.state_dict().items())
    torch.multiprocessing.spawn(shared_worker,args=(str(tmp_path/'gloo'),str(tmp_path)),nprocs=2,join=True)
    actual=torch.load(tmp_path/'actual.pt',weights_only=False);metrics=actual['metrics']
    for name,v in model.state_dict().items():torch.testing.assert_close(v,actual['state'][name],atol=3e-7,rtol=3e-5)
    for name,norm in norms.items():assert metrics['component_gradient_norms'][name]==pytest.approx(norm,rel=2e-4,abs=1e-8)
    for name,value in zip(('ren_loss','answer_stop_loss','reference_penalty'),expected[:3]):
        assert metrics[name]==pytest.approx(float(value),rel=1e-4,abs=3e-7)
    assert metrics['reasoning_horizon']['early']['reasoning_objective_contribution']==pytest.approx(float(expected[3]),rel=2e-4,abs=1e-7)
    for head,original in zip((step.frozen_head,step.teacher_head,step.reference_head),frozen):
        for k,v in head.state_dict().items():torch.testing.assert_close(v,original[k],atol=0,rtol=0)


def test_zero_shared_mass_has_zero_initial_reasoning_gradient(tmp_path):
    a,model,records,step,opt=shared_fixture(tmp_path)
    for r in records:r['hindsight_hidden']=r['student_hidden'].clone()
    prepare_weights(step,records,a,0,1)
    step.loss_component_mode=0
    (step(records)/9).backward()
    norm=sum(float(p.grad.square().sum()) for p in model.parameters() if p.grad is not None)**.5
    assert norm<2e-6
