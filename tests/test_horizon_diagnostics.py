import copy
import pytest
import torch
from lulu import persistent,training
from lulu.stable import prepare_weights
from test_balanced import balanced_fixture


def test_horizon_gradients_use_full_reasoning_denominator_and_leave_update_unchanged(tmp_path):
    a,model,records,step,opt=balanced_fixture(tmp_path/'instrumented')
    a.reasoning_diagnostic_split=2
    original_records=copy.deepcopy(records)
    plain_model=copy.deepcopy(model);plain_step=copy.deepcopy(step);plain_step.student=plain_model
    plain_args=copy.copy(a);plain_args.gradient_norm_every=0;plain_args.reasoning_diagnostic_split=0
    plain_args.output_dir=str(tmp_path/'plain');plain_step.args=plain_args
    plain_opt=torch.optim.SGD(plain_model.parameters(),lr=plain_args.learning_rate)
    stats=prepare_weights(step,records,a,0,1)
    assert stats['reasoning_horizon']['early']['reasoning_tokens']==4
    assert stats['reasoning_horizon']['late']['reasoning_tokens']==1
    states=training.selected_hidden(model,step.tok,records,'causal_prompt_ids',a)
    early,late=states[0].sum()*0.,states[0].sum()*0.
    for r,h in zip(records,states):
        mask=torch.tensor(r['reasoning_mask']);n=int(mask.sum())
        if not n:continue
        p=model.get_output_embeddings()(h).log_softmax(-1)
        q=step.teacher_head(r['teacher_hidden']).softmax(-1).detach()
        loss=(q*(q.log()-p)).sum(-1)*r['stable_rho']*mask/(2*2*n)
        early=early+loss[:2].sum();late=late+loss[2:].sum()
    expected={}
    for name,value in [('reasoning',early+late),('reasoning_early',early),('reasoning_late',late)]:
        grads=torch.autograd.grad(value,list(model.parameters()),retain_graph=True,allow_unused=True)
        expected[name]=sum(float(g.double().square().sum()) for g in grads if g is not None)**.5
    dummy=dict(causal_prompt_ids=[3,4],response_ids=[2],positions=[],snapshot_round=0,dummy=True)
    # update_records reconstructs weights from untouched caches.
    metrics=persistent.update_records(step,step,opt,copy.deepcopy(original_records),dummy,a,0,1)
    plain=persistent.update_records(plain_step,plain_step,plain_opt,copy.deepcopy(original_records),dummy,plain_args,0,1)
    for name,norm in expected.items():assert metrics['component_gradient_norms'][name]==pytest.approx(norm,rel=1e-4,abs=1e-8)
    assert metrics['reasoning_horizon']['early']['reasoning_objective_contribution']==pytest.approx(float(early),rel=1e-4)
    assert metrics['reasoning_horizon']['late']['reasoning_objective_contribution']==pytest.approx(float(late),rel=1e-4)
    assert sum(v['reasoning_objective_contribution'] for v in metrics['reasoning_horizon'].values())==pytest.approx(metrics['ren_loss'],rel=1e-5)
    for name,tensor in model.state_dict().items():torch.testing.assert_close(tensor,plain_model.state_dict()[name],atol=2e-7,rtol=2e-5)


def test_final_only_parser_excludes_draft_numbers_and_unfinished_thinking():
    from lulu import thinking_final_parser as parser
    text='I think the answer is 1. Consider two examples. </think>\n### Final Answer\nAnswer: 504'
    assert parser.extract_answer(text,'math500')=='504'
    assert parser.math_equal(parser.extract_answer(text,'math500'),'504')
    assert not parser.math_equal(parser.extract_answer('draft \\boxed{504}','math500'),'504')
    assert parser.extract_answer('draft answer: A </think> Final answer: D','gpqa_diamond')=='__CHOICE__D'
