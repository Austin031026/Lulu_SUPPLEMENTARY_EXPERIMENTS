import numpy as np
import pytest
import torch
from lulu.directional_audit import projection_diagnostics as score, sample_reasoning_positions


def test_exact_projection_and_geometric_movement():
    c=torch.tensor([[.3,-.4,1.,.2]],dtype=torch.float64)
    t=torch.tensor([[1.2,.1,-.6,.4]],dtype=torch.float64)
    for b in (0.,.25,.7,1.):
        h=(1-b)*c+b*t
        d=score(c,h,t,epsilon=1e-14)
        torch.testing.assert_close(d['beta'],torch.tensor([b],dtype=torch.float64),atol=1e-12,rtol=1e-12)
        target=h.softmax(-1)
        expected=(target*(h.log_softmax(-1)-c.log_softmax(-1))).sum(-1)
        torch.testing.assert_close(d['projected_kl'],expected,atol=1e-12,rtol=1e-12)
        assert 0<=d['movement_ratio'].item()<=1+1e-12
    assert score(c,c,t)['beta'].item()==0
    assert not score(c,c,t)['alignment_valid'].item()
    assert score(c,2*c-t,t)['beta'].item()==0
    assert score(c,2*c-t,t)['cosine'].item()==pytest.approx(-1.)
    assert score(c,t,c)['beta'].item()==0
    assert not score(c,t,c)['alignment_valid'].item()


def test_centering_offsets_and_detachment():
    torch.manual_seed(2)
    c,h,t=[torch.randn(8,97,dtype=torch.float64,requires_grad=True) for _ in range(3)]
    d=score(c,h,t,torch.arange(8))
    offsets=[torch.randn(8,1,dtype=torch.float64)*100 for _ in range(3)]
    other=score(c+offsets[0],h+offsets[1],t+offsets[2],torch.arange(8))
    for k in d:torch.testing.assert_close(d[k],other[k])
    assert not any(v.requires_grad for v in d.values())
    assert torch.all(d['projected_kl']<=d['causal_kl']+1e-12)
    torch.testing.assert_close(d['resolved_mismatch'],d['causal_kl']-d['hindsight_kl'])


def test_position_sampling_expansion_recovers_reasoning_denominators():
    mask=np.ones(8100,dtype=bool);mask[30:77]=False;mask[8090:]=False
    positions,weights,bins=sample_reasoning_positions(mask,seed=17)
    assert len(positions)==256 and positions==sorted(set(positions))
    assert all(mask[p] for p in positions)
    assert sum(weights)==pytest.approx(mask.sum())
    assert (positions,weights,bins)==sample_reasoning_positions(mask,seed=17)
    assert sample_reasoning_positions([False]*10,seed=1)==([],[],[])
    assert sample_reasoning_positions([True]*3,seed=1)==([0,1,2],[1.,1.,1.],[0,0,0])
