import pytest
import torch
from lulu.certified_audit import certified_diagnostics as score


def test_autograd_gradient_identity_and_single_update_equivalence():
    torch.manual_seed(19)
    c,h,t=[torch.randn(7,19,dtype=torch.float64,requires_grad=True) for _ in range(3)]
    d=score(c,h,t)
    lam=d['lambda_'];p=c.detach().softmax(-1);q=t.detach().softmax(-1)
    target=((1-lam[:,None])*p+lam[:,None]*q).detach()
    live=c.detach().clone().requires_grad_()
    grad=torch.autograd.grad(-(target*live.log_softmax(-1)).sum(),live)[0]
    old=torch.autograd.grad(-(lam[:,None]*q*live.log_softmax(-1)).sum(),live)[0]
    torch.testing.assert_close(grad,lam[:,None]*(p-q),atol=1e-14,rtol=1e-12)
    torch.testing.assert_close(grad,old,atol=1e-14,rtol=1e-12)
    assert torch.all(grad.norm(dim=-1)<= (p-q).norm(dim=-1)+1e-14)
    assert all(not x.requires_grad for x in d.values())


def test_known_probability_projection_and_boundaries():
    p=torch.tensor([[.5,.3,.2]],dtype=torch.float64)
    q=torch.tensor([[.2,.3,.5]],dtype=torch.float64)
    for coeff in [0.,.1,.25,.5,1.]:
        h=(1-coeff)*p+coeff*q
        d=score(p.log(),h.log(),q.log(),epsilon=1e-14)
        assert d['lambda_'].item()==pytest.approx(coeff,abs=1e-12)
        assert d['gradient_identity_error_fp64'].item()<1e-14
        assert d['gradient_ratio_fp64'].item()==pytest.approx(d['lambda_'].item(),abs=1e-12)
    opposite=torch.tensor([[.65,.3,.05]],dtype=torch.float64)
    assert score(p.log(),opposite.log(),q.log())['lambda_'].item()==0
    zero=score(p.log(),q.log(),p.log())
    assert zero['lambda_'].item()==0 and not zero['gradient_ratio_valid'].item()
    assert zero['certified_kl'].item()==0


def test_fp32_saturation_finite_and_gradient_errors_bounded():
    c=torch.tensor([[40.,-20.,-30.],[0.,-20.,-40.],[0.,0.,0.]],dtype=torch.float32)
    h=c+torch.tensor([[.1,0.,0.],[.01,.2,.1],[.2,-.1,.4]])
    t=c+torch.tensor([[.3,-.1,.2],[.1,.5,.2],[.3,-.2,.5]])
    d=score(c,h,t)
    assert all(torch.isfinite(v).all() for v in d.values())
    assert torch.all(d['gradient_identity_error_fp64']<1e-14)
    assert torch.all(d['gradient_identity_error_fp32']<2e-7)


def test_refreshed_endpoints_can_converge_to_teacher():
    initial=torch.tensor([.8,.2],dtype=torch.float64)
    teacher=torch.tensor([.2,.8],dtype=torch.float64)
    lam=.1;p=initial.clone()
    for _ in range(80):p=(1-lam)*p+lam*teacher
    torch.testing.assert_close(p,teacher+(1-lam)**80*(initial-teacher))
    assert (p-teacher).norm() < .001*(initial-teacher).norm()


def test_gradient_need_not_be_smaller_than_old_weighted_ren():
    p=torch.tensor([[.5,.5]],dtype=torch.float64)
    q=torch.tensor([[.6,.4]],dtype=torch.float64)
    h=torch.tensor([[.52,.48]],dtype=torch.float64)
    d=score(p.log(),h.log(),q.log())
    assert d['certified_gradient_norm'].item()>d['old_weighted_gradient_norm'].item()
