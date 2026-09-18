"""Stable ReN structural boundaries, exact objective and distributed reductions."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from lulu import training,persistent
from lulu.data import structured_response_regions
from lulu.stable import bounded_weight, reference_reverse_kl, prepare_weights
from lulu.stability import collapse_decision
from test_lulu_training import _student
from test_lulu_persistent import _args


class Markers:
    all_special_ids=[1,2,3]
    def encode(self,text,add_special_tokens=False):
        return {'<think>':[1],'</think>':[2]}[text]
    def decode(self,ids,**kwargs):
        pieces={1:'<think>',2:'</think>',3:'<|im_end|>',10:' the answer is ',11:'\\boxed{42} but check again ',12:'42'}
        return ''.join(pieces.get(i,'reason ') for i in ids)


def test_structural_mask_keeps_answer_like_text_in_thinking_and_all_control_tokens():
    response=[1,10,11,9,2,12,3]
    regions=structured_response_regions(Markers(),response)
    assert regions['reasoning']==[False,True,True,True,False,False,False]
    assert all(r != c for r,c in zip(regions['reasoning'],regions['control']))
    assert structured_response_regions(Markers(),[10,11],prompt_ids=[1])['reasoning']==[True,True]
    assert structured_response_regions(Markers(),[1,10,11])['reasoning']==[False,True,True]
    assert structured_response_regions(Markers(),[12,3])['control']==[True,True]


def test_bounded_weights_are_detached_and_never_mean_normalized():
    dc=torch.tensor([2.,2.,1.,0.],requires_grad=True)
    r=torch.tensor([1.,-1.,3.,0.],requires_grad=True)
    rho=bounded_weight(dc,r,1e-6)
    torch.testing.assert_close(rho,torch.tensor([.5,0.,1.,0.]))
    assert not rho.requires_grad and rho.mean()<1


def test_reverse_kl_has_student_gradient_and_no_reference_gradient():
    torch.manual_seed(31)
    live=torch.randn(4,11,dtype=torch.float64,requires_grad=True)
    reference=torch.randn(4,11,dtype=torch.float64,requires_grad=True)
    result=reference_reverse_kl(live,reference).mean()
    expected=(live.softmax(-1)*(live.log_softmax(-1)-reference.detach().log_softmax(-1))).sum(-1).mean()
    torch.testing.assert_close(result,expected)
    result.backward()
    assert live.grad.abs().sum()>0 and reference.grad is None
    torch.testing.assert_close(reference_reverse_kl(reference,reference),torch.zeros(4,dtype=torch.float64),atol=1e-14,rtol=0)


def fixture(root,zero_rho=False):
    model=_student(); reference=copy.deepcopy(model).requires_grad_(False).eval()
    with torch.no_grad(): model.model.layers[0].self_attn.v_proj.weight.add_(.03)
    causal=copy.deepcopy(model).requires_grad_(False).eval()
    args=_args(method='ren_stable',cpu=True,dtype='float32',lora_rank=0,global_batch_prompts=3,
               max_sequence_tokens=128,max_prompt_tokens=64,output_dir=str(root),round=0,
               learning_rate=.001,logit_chunk_size=2,top_k=3,max_grad_norm=1e6)
    tok=SimpleNamespace(pad_token_id=0)
    records=[dict(index=0,causal_prompt_ids=[3,4],hindsight_prompt_ids=[3,4,20],response_ids=[6,7,8,9,2],reasoning_mask=[True,True,True,False,False]),
             dict(index=1,causal_prompt_ids=[3,5,11],hindsight_prompt_ids=[3,5,11,24],response_ids=[12,13,2],reasoning_mask=[True,True,False]),
             dict(index=2,causal_prompt_ids=[3,4],hindsight_prompt_ids=[3,4,20],response_ids=[2],reasoning_mask=[False])]
    for r in records:r.update(positions=list(range(len(r['response_ids']))),snapshot_round=0,teacher_scored=True)
    head=copy.deepcopy(causal.get_output_embeddings()).requires_grad_(False)
    teacher=torch.nn.Linear(16,31,bias=False).requires_grad_(False)
    torch.nn.init.normal_(teacher.weight,std=.15)
    with torch.no_grad():
        c=training.selected_hidden(causal,tok,records,'causal_prompt_ids',args)
        h=training.selected_hidden(causal,tok,records,'hindsight_prompt_ids',args)
        ref=training.selected_hidden(reference,tok,records,'causal_prompt_ids',args)
        for r,ch,hh,rh in zip(records,c,h,ref):
            r.update(student_hidden=ch.detach().clone(),hindsight_hidden=(ch if zero_rho else hh).detach().clone(),
                     reference_hidden=rh.detach().clone(),teacher_hidden=torch.randn_like(ch))
            if zero_rho:r['reasoning_mask']=[True]*len(r['positions'])
    step=training.DistillationStep(model,head,teacher,tok,args)
    step.reference_head=copy.deepcopy(reference.get_output_embeddings()).requires_grad_(False)
    opt=torch.optim.SGD(model.parameters(),lr=args.learning_rate)
    return args,model,records,step,opt


def dense(step,records,args):
    states=training.selected_hidden(step.student,step.tok,records,'causal_prompt_ids',args)
    result=0.
    for r,h in zip(records,states):
        logits=step.student.get_output_embeddings()(h)
        q=step.teacher_head(r['teacher_hidden']).softmax(-1).detach()
        ref=step.reference_head(r['reference_hidden']).detach().log_softmax(-1)
        logp=logits.log_softmax(-1)
        kl=(q*(q.log()-logp)).sum(-1)
        reasoning=torch.tensor(r['reasoning_mask'],dtype=torch.bool)
        w=torch.where(reasoning,r['stable_rho'],1.)
        result=result+(w*kl+args.reference_kl_coef*(logp.exp()*(logp-ref)).sum(-1)).sum()
    return result/sum(len(r['positions']) for r in records)


def ddp_worker(rank,rendezvous,root,zero):
    args,model,records,step,opt=fixture(Path(root)/'distributed',zero)
    dist.init_process_group('gloo',init_method=f'file://{rendezvous}',rank=rank,world_size=2)
    try:
        wrapped=DDP(step,broadcast_buffers=False)
        dummy=dict(causal_prompt_ids=[3,4],response_ids=[2],positions=[],snapshot_round=0,dummy=True)
        metrics=persistent.update_records(step,wrapped,opt,records[rank::2],dummy,args,rank,2)
        if rank==0:torch.save({'model':model.state_dict(),'metrics':metrics},Path(root)/'actual.pt')
    finally:dist.destroy_process_group()


@pytest.mark.parametrize('zero',[False,True])
def test_stable_ddp_matches_dense_all_token_objective_with_uneven_shards(tmp_path,zero):
    args,model,records,step,opt=fixture(tmp_path/'reference',zero)
    frozen=copy.deepcopy(step.reference_head.state_dict())
    stats=prepare_weights(step,records,args,0,1)
    assert stats['loss_tokens']==9
    if zero:assert stats['rho_sum']==0
    expected=dense(step,records,args)
    torch.testing.assert_close(step(records)/9,expected,atol=2e-7,rtol=2e-5)
    expected.backward();opt.step()
    torch.multiprocessing.spawn(ddp_worker,args=(str(tmp_path/'gloo'),str(tmp_path),zero),nprocs=2,join=True)
    actual=torch.load(tmp_path/'actual.pt',weights_only=False)
    assert actual['metrics']['objective_loss']==pytest.approx(expected.item(),rel=3e-4,abs=2e-7)
    assert actual['metrics']['reference_kl']>0
    assert not actual['metrics']['skipped_update']
    for name,tensor in model.state_dict().items():
        torch.testing.assert_close(tensor,actual['model'][name],atol=2e-7,rtol=2e-5)
    for name,tensor in frozen.items():torch.testing.assert_close(step.reference_head.state_dict()[name],tensor,atol=0,rtol=0)


def test_collapse_guard_needs_sustained_joint_changes():
    healthy=dict(hit_cap_fraction=.5,repetition_above_half_fraction=.02,mean_response_tokens=6500)
    bad=dict(hit_cap_fraction=.92,repetition_above_half_fraction=.6,mean_response_tokens=8100)
    assert not collapse_decision([healthy]*4+[bad])[0]
    assert collapse_decision([healthy]*3+[bad]*2)[0]
    assert not collapse_decision([healthy]*5)[0]


def test_hindsight_refresh_never_refreshes_reference(monkeypatch,tmp_path):
    from lulu import hindsight_service as service
    from test_lulu_hindsight_service import _ScriptedPipe
    args,model,records,step,opt=fixture(tmp_path)
    args.initial_checkpoint=None
    reference=_student().eval()
    original_ref=copy.deepcopy(reference.state_dict())
    loaded=iter([model,reference]);calls=[]
    def load(*a,**kw):
        calls.append(kw.get('checkpoint_dir'));return next(loaded)
    monkeypatch.setattr(service,'load_student',load)
    monkeypatch.setattr(service,'load_tokenizer',lambda _:step.tok)
    state={n:p.detach().clone() for n,p in model.named_parameters()}
    next_state={n:p+.02 for n,p in state.items()}
    payload=[{k:r[k] for k in ('causal_prompt_ids','hindsight_prompt_ids','response_ids','positions')} for r in records]
    pipe=_ScriptedPipe([{'op':'sync','round':0,'state':state},
                       {'op':'score','round':0,'request_id':0,'records':payload},
                       {'op':'sync','round':1,'state':next_state},
                       {'op':'score','round':1,'request_id':1,'records':payload}, {'op':'stop'}])
    service.hindsight_worker(args,[],pipe)
    assert len(calls)==2 and calls[1]==tmp_path/'checkpoints/round_000000'
    first=pipe.responses[2]['records'];second=pipe.responses[4]['records']
    for a,b in zip(first,second):
        torch.testing.assert_close(a['reference_hidden'],b['reference_hidden'],rtol=0,atol=0)
        assert not torch.allclose(a['hindsight_hidden'],b['hindsight_hidden'])
    for name,tensor in original_ref.items():torch.testing.assert_close(reference.state_dict()[name],tensor,rtol=0,atol=0)


@pytest.mark.parametrize("method", ["ren_stable", "ren_balanced", "ren_shared"])
def test_cpu_full_pipeline_collect_score_update_checkpoint_and_diagnostics(tmp_path, method):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    model=_student()
    model.config.eos_token_id=3;model.generation_config.eos_token_id=3
    vocab={'<pad>':0,'<think>':1,'</think>':2,'<|im_end|>':3,'<unk>':4}
    vocab.update({f'word{i}':i for i in range(5,31)})
    backend=Tokenizer(WordLevel(vocab,unk_token='<unk>'));backend.pre_tokenizer=Whitespace()
    tok=PreTrainedTokenizerFast(tokenizer_object=backend,pad_token='<pad>',eos_token='<|im_end|>',unk_token='<unk>',additional_special_tokens=['<think>','</think>'])
    tok.chat_template="{% for m in messages %}{{ m['content'] }} {% endfor %}{% if add_generation_prompt %}<think>{% endif %}"
    source=tmp_path/'student';teacher=tmp_path/'teacher'
    model.save_pretrained(source);tok.save_pretrained(source)
    with torch.no_grad():model.model.layers[0].self_attn.v_proj.weight.add_(.1)
    model.save_pretrained(teacher);tok.save_pretrained(teacher)
    data=tmp_path/'data.jsonl'
    data.write_text(''.join(json.dumps(dict(id=f'p{i}',messages=[dict(role='user',content=f'word{i+5} problem')],gold_answer='42'))+'\n' for i in range(2)))
    args=_args(method=method,gradient_norm_every=1 if method=='ren_balanced' else 0,cpu=True,dtype='float32',lora_rank=0,model=str(source),teacher_model=str(teacher),
               train_data=str(data),output_dir=str(tmp_path/'run'),rounds=2,global_batch_prompts=2,
               max_new_tokens=8,max_prompt_tokens=100,max_sequence_tokens=128,top_k=3,logit_chunk_size=3,
               rollout_batch_size=2,score_batch_size=1,worker_timeout=90,learning_rate=1e-4)
    persistent.run_persistent(args)
    state=json.loads((tmp_path/'run/checkpoints/latest/lulu_state.json').read_text())
    assert state['completed_rounds']==state['completed_updates']==2
    history=json.loads((tmp_path/'run/analysis/stability_history.json').read_text())
    assert len(history)==2 and all(r['loss_token_fraction']==1 for r in history)
    assert (tmp_path/'run/analysis/stability.png').is_file()
    assert not (tmp_path/'run/early_stop.json').exists()
