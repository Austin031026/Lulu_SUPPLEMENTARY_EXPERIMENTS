"""Recipe correctness: exact frozen score/live loss, coefficient gradients and fail-closed step."""
import copy
import pytest
import torch
from lulu import training, persistent, stable
from test_balanced import balanced_fixture, dense_components


def test_control_coefficient_changes_only_control_gradient_and_loss(tmp_path):
    a,model,records,step,opt=balanced_fixture(tmp_path)
    a.control_loss_coef=.5;a.match_causal_update=True
    stats=stable.prepare_weights(step,records,a,0,1)
    parts=dense_components(step,records,a)
    expected=parts[0]+.5*parts[1]+parts[2]
    torch.testing.assert_close(step(records)/9,expected,atol=3e-7,rtol=3e-5)
    assert stats['reasoning_loss_score_expected']==pytest.approx(float(parts[0].detach()),abs=1e-7)
    step.loss_component_mode=1
    torch.testing.assert_close(step(records)/9,.5*parts[1],atol=3e-7,rtol=3e-5)
    grad=torch.autograd.grad(step(records)/9,list(model.parameters()),allow_unused=True)
    reference=torch.autograd.grad(.5*parts[1],list(model.parameters()),allow_unused=True)
    for g,r in zip(grad,reference):
        if g is not None and r is not None:torch.testing.assert_close(g,r,atol=2e-7,rtol=2e-4)


def test_mismatch_stops_before_parameter_or_optimizer_mutation(tmp_path,monkeypatch):
    a,model,records,step,opt=balanced_fixture(tmp_path)
    a.match_causal_update=True;a.control_loss_coef=.5;a.gradient_norm_every=0
    before=copy.deepcopy(model.state_dict())
    original=stable.prepare_weights
    def corrupt(*args,**kwargs):
        stats=original(*args,**kwargs);stats['reasoning_loss_score_expected']+=1.;return stats
    monkeypatch.setattr(stable,'prepare_weights',corrupt)
    with pytest.raises(RuntimeError,match='before optimizer step'):
        persistent.update_records(step,step,opt,records,{},a,0,1)
    for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],atol=0,rtol=0)
    assert not opt.state


def test_matched_recipe_commits_and_logs_weighted_control(tmp_path):
    a,model,records,step,opt=balanced_fixture(tmp_path)
    a.match_causal_update=True;a.control_loss_coef=.25;a.gradient_norm_every=0
    metrics=persistent.update_records(step,step,opt,records,{},a,0,1)
    assert metrics['completed_updates']==1
    assert metrics['reasoning_score_live_abs_error']<1e-7
    assert metrics['control_penalty']==pytest.approx(.25*metrics['answer_stop_loss'])
    assert metrics['objective_loss']==pytest.approx(metrics['ren_loss']+metrics['control_penalty']+metrics['reference_penalty'],abs=1e-7)


def test_matched_recipe_rejects_different_update_batch_shape():
    a=training.parser().parse_args(['--train-data','unused','--output-dir','unused','--method','ren_balanced',
        '--lora-rank','0','--match-causal-update','--train-micro-batch-size','2'])
    with pytest.raises(ValueError,match='microbatch=1'):training.validate_args(a)
