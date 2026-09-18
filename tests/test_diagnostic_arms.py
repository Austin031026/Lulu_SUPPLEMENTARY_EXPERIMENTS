import copy
import pytest
import torch
from lulu import training
from lulu.stable import prepare_weights
from lulu.gradient_diagnostics import component_gradient_norms
from test_balanced import balanced_fixture,dense_components

@pytest.mark.parametrize('arm',['ren','vanilla','none'])
def test_arms_keep_control_reference_and_exact_cosines(tmp_path,arm):
    a,model,records,step,opt=balanced_fixture(tmp_path)
    a.reasoning_ablation=arm;a.gradient_cosines=True
    before={k:v.detach().clone() for k,v in model.state_dict().items()}
    prepare_weights(step,records,a,0,1)
    for r in records:
        if arm=='vanilla':torch.testing.assert_close(r['stable_rho'],torch.tensor(r['reasoning_mask']).float())
        if arm=='none':assert r['stable_rho'].count_nonzero()==0
    expected=dense_components(step,records,a);total=sum(len(r['positions']) for r in records)
    if arm=='none':assert float(expected[0])==0
    torch.testing.assert_close(step(records)/total,expected.sum(),atol=3e-7,rtol=3e-5)
    vectors={}
    for name,value in zip(['reasoning','control','reference'],expected):
        gradients=torch.autograd.grad(value,list(model.parameters()),retain_graph=True,allow_unused=True)
        vectors[name]=torch.cat([(g if g is not None else torch.zeros_like(p)).reshape(-1).double() for p,g in zip(model.parameters(),gradients)])
    result=component_gradient_norms(step,step,opt,records,a,1,total)
    for pair,dot in result['component_gradient_dots'].items():
        x,y=pair.split('__');v,w=vectors[x],vectors[y];den=float(v.norm()*w.norm())
        assert dot==pytest.approx(float(v@w),rel=2e-4,abs=1e-10)
        actual=result['component_gradient_cosines'][pair]
        if den<1e-20:assert actual is None
        else:assert actual==pytest.approx(float(v@w)/den,rel=2e-4,abs=1e-7)
    for name,v in model.state_dict().items():torch.testing.assert_close(v,before[name],rtol=0,atol=0)
    assert not opt.state and all(p.grad is None for p in model.parameters())

def test_arms_require_matching_backbone():
    a=training.parser().parse_args(['--train-data','unused','--output-dir','unused','--method','vanilla_opd','--reasoning-ablation','vanilla'])
    with pytest.raises(ValueError,match='same ren_balanced'):training.validate_args(a)
