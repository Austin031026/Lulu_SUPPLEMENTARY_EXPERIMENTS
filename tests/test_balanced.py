"""Exact hierarchical normalization, component gradients and dev selection."""
import copy,json
from pathlib import Path
import pytest,torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from lulu import training,persistent
from lulu.stable import prepare_weights,bounded_absolute_weight
from lulu.validation import merge_validation
from test_stable import fixture


def balanced_fixture(root):
    args,model,records,step,opt=fixture(root)
    args.method='ren_balanced';args.gradient_norm_every=1
    # Two unequal rollouts for prompt A, one empty-reasoning rollout for B.
    for r,prompt in zip(records,['A','A','B']):r['source_id']=prompt
    records[0]['truncated']=True;records[1]['truncated']=False;records[2]['truncated']=False
    return args,model,records,step,opt


def dense_components(step,records,args):
    from collections import Counter
    counts=Counter(r['source_id'] for r in records);n=sum(len(r['positions']) for r in records)
    states=training.selected_hidden(step.student,step.tok,records,'causal_prompt_ids',args)
    zero=states[0].sum()*0.;reason,control,reference=zero,zero,zero
    for r,h in zip(records,states):
        logits=step.student.get_output_embeddings()(h);logp=logits.log_softmax(-1)
        q=step.teacher_head(r['teacher_hidden']).softmax(-1).detach();ref=step.reference_head(r['reference_hidden']).detach().log_softmax(-1)
        kl=(q*(q.log()-logp)).sum(-1);mask=torch.tensor(r['reasoning_mask'],dtype=torch.bool)
        if mask.sum():reason=reason+(r['stable_rho']*kl*mask).sum()/(len(counts)*counts[r['source_id']]*mask.sum())
        control=control+(kl*(~mask)).sum()/n
        reference=reference+(logp.exp()*(logp-ref)).sum()/n*args.reference_kl_coef
    return torch.stack((reason,control,reference))


def balanced_worker(rank,rendezvous,root):
    args,model,records,step,opt=balanced_fixture(Path(root)/'ddp')
    args.reasoning_diagnostic_split=2
    args.gradient_cosines=True
    dist.init_process_group('gloo',init_method=f'file://{rendezvous}',rank=rank,world_size=2)
    try:
        wrapper=DDP(step,broadcast_buffers=False,gradient_as_bucket_view=True)
        dummy=dict(causal_prompt_ids=[3,4],response_ids=[2],positions=[],snapshot_round=0,dummy=True)
        metrics=persistent.update_records(step,wrapper,opt,records[rank::2],dummy,args,rank,2)
        if rank==0:torch.save({'state':model.state_dict(),'metrics':metrics},Path(root)/'actual.pt')
    finally:dist.destroy_process_group()


def test_absolute_weight_is_bounded_detached_and_not_renormalized():
    g=torch.tensor([-1.,0.,.001,.1,1.,20.],requires_grad=True)
    w=bounded_absolute_weight(g)
    torch.testing.assert_close(w,torch.tensor([0.,0.,.001/1.001,.1/1.1,.5,20/21]))
    assert not w.requires_grad and 0<=w.min() and w.max()<1


def test_balanced_formula_and_exact_full_gradient_diagnostics_across_uneven_ddp(tmp_path):
    a,model,records,step,opt=balanced_fixture(tmp_path/'dense')
    prepare_weights(step,records,a,0,1)
    assert records[0]['reasoning_loss_scale']==pytest.approx(9/(2*2*3))
    assert records[1]['reasoning_loss_scale']==pytest.approx(9/(2*2*2))
    assert records[2]['reasoning_loss_scale']==0
    expected=dense_components(step,records,a)
    torch.testing.assert_close(step(records)/9,expected.sum(),atol=3e-7,rtol=3e-5)
    parameters=list(model.parameters());norms=[];vectors=[]
    for component in expected:
        grads=torch.autograd.grad(component,parameters,retain_graph=True,allow_unused=True)
        norms.append(sum(float(g.double().square().sum()) for g in grads if g is not None)**.5)
        vectors.append(torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1).double() for p,g in zip(parameters,grads)]))
    expected.sum().backward();opt.step()
    torch.multiprocessing.spawn(balanced_worker,args=(str(tmp_path/'gloo'),str(tmp_path)),nprocs=2,join=True)
    actual=torch.load(tmp_path/'actual.pt',weights_only=False)
    for name,tensor in model.state_dict().items():torch.testing.assert_close(tensor,actual['state'][name],atol=3e-7,rtol=3e-5)
    metrics=actual['metrics'];assert metrics['objective_loss']==pytest.approx(float(expected.sum()),rel=4e-5)
    for name,value in zip(['reasoning','control','reference'],norms):
        assert metrics['component_gradient_norms'][name]==pytest.approx(value,rel=8e-5,abs=1e-8)
    for i,j,key in [(0,1,'reasoning__control'),(0,2,'reasoning__reference'),(1,2,'control__reference')]:
        cosine=float(vectors[i]@vectors[j])/(norms[i]*norms[j])
        assert metrics['component_gradient_cosines'][key]==pytest.approx(cosine,rel=2e-4,abs=1e-7)
    assert metrics['ren_loss']==pytest.approx(float(expected[0]),rel=3e-5)
    assert metrics['reasoning_horizon']['early']['reasoning_tokens']==4
    assert metrics['reasoning_horizon']['late']['reasoning_tokens']==1
    assert sum(b['reasoning_objective_contribution'] for b in metrics['reasoning_horizon'].values())==pytest.approx(float(expected[0]),rel=3e-5)
    assert 0<=metrics['capped_reasoning_loss_share']<=1


def test_reason_denominator_ignores_control_length_and_zero_weight_token_count(tmp_path):
    a,_,records,step,_=balanced_fixture(tmp_path)
    prepare_weights(step,records,a,0,1)
    # Multiplying the global token denominator rescales the internal numerator,
    # leaving the prompt-balanced reason term invariant.
    from lulu.stable import prompt_reasoning_scales
    before=[r['reasoning_loss_scale']/9 for r in records]
    prompt_reasoning_scales(records,900,a,1)
    assert [r['reasoning_loss_scale']/900 for r in records]==pytest.approx(before)
    # Zero weights do not remove the corresponding reasoning position.
    records[0]['stable_rho'].zero_();prompt_reasoning_scales(records,9,a,1)
    assert records[0]['reasoning_loss_scale']==pytest.approx(9/(2*2*3))


def test_dev_selection_uses_accuracy_then_earlier_round_and_validates_coverage(tmp_path):
    for round_index,correct in [(0,[True,False]),(4,[True,True]),(8,[True,True])]:
        checkpoint=tmp_path/'checkpoints'/f'round_{round_index:06d}';checkpoint.mkdir(parents=True)
        directory=tmp_path/'validation'/f'round_{round_index:06d}';directory.mkdir(parents=True)
        for rank in range(2):
            row={'index':rank,'source_id':str(rank),'correct':correct[rank],'checkpoint':str(checkpoint),'hit_cap':False,'response_tokens':10}
            (directory/f'shard-{rank:03d}.jsonl').write_text(json.dumps(row)+'\n')
        merge_validation(tmp_path,round_index,2,2)
    assert json.loads((tmp_path/'validation/selected.json').read_text())['round']==4
    (tmp_path/'validation/round_000008/shard-001.jsonl').write_text('')
    with pytest.raises(RuntimeError,match='coverage'):merge_validation(tmp_path,8,2,2)


def test_resident_dev_worker_uses_greedy_fresh_snapshot_and_keeps_all_examples(monkeypatch,tmp_path):
    from types import SimpleNamespace
    from lulu import validation
    data=tmp_path/'dev.jsonl';data.write_text(''.join(json.dumps({'id':str(i),'messages':[{'role':'user','content':f'question {i}'}],'gold_answer':'42'})+'\n' for i in range(3)))
    args=SimpleNamespace(validation_data=str(data),validation_max_examples=3,output_dir=str(tmp_path))
    monkeypatch.setattr(validation,'build_prompt_views',lambda *a,**kw:{'causal_prompt_ids':[10,20]})
    class Client:
        def generate(self,records,index,*,greedy):
            assert greedy and index==4
            assert [r['index'] for r in records]==[0,2]
            for r in records:r['rollout_checkpoint']=str(tmp_path/'checkpoints/round_000004')
            return [[1,2],[1,2]],['stop','length']
    model=SimpleNamespace(eval=lambda:None,gradient_checkpointing_disable=lambda:None)
    tok=SimpleNamespace(decode=lambda *a,**kw:r'Final \boxed{42}')
    value=validation.validate_student(model,tok,args,0,2,Client(),4)
    assert value['examples']==2
    rows=[json.loads(line) for line in (tmp_path/'validation/round_000004/shard-000.jsonl').read_text().splitlines()]
    assert all(r['correct'] for r in rows) and rows[1]['hit_cap']
