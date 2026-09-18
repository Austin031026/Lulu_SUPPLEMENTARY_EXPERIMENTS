import torch
from lulu.allocation import allocate_reasoning_weights


def test_uniform_preserves_rollout_weight_mass():
    base=torch.tensor([0.1,0.0,0.4,0.2,0.0])
    dc=torch.tensor([1.,2.,3.,4.,5.])
    mask=torch.tensor([1,0,1,1,0],dtype=torch.bool)
    out=allocate_reasoning_weights(base,dc,mask,arm='uniform',seed=42,round_index=0,source_id='x',record_index=0)
    assert out[~mask].count_nonzero()==0
    torch.testing.assert_close(out[mask],torch.full((3,),base[mask].mean()))
    torch.testing.assert_close(out.sum(),base[mask].sum())


def test_shuffled_preserves_exact_multiset_and_is_deterministic():
    base=torch.tensor([0.1,0.2,0.3,0.4])
    dc=torch.arange(4,dtype=torch.float32)
    mask=torch.ones(4,dtype=torch.bool)
    a=allocate_reasoning_weights(base,dc,mask,arm='shuffled',seed=7,round_index=3,source_id='p',record_index=9)
    b=allocate_reasoning_weights(base,dc,mask,arm='shuffled',seed=7,round_index=3,source_id='p',record_index=9)
    torch.testing.assert_close(a,b)
    torch.testing.assert_close(torch.sort(a).values,torch.sort(base).values)


def test_causal_matched_assigns_largest_weight_to_largest_causal_gap():
    base=torch.tensor([0.4,0.1,0.3,0.2])
    dc=torch.tensor([1.,4.,2.,3.])
    mask=torch.ones(4,dtype=torch.bool)
    out=allocate_reasoning_weights(base,dc,mask,arm='causal_matched',seed=0,round_index=0,source_id='p',record_index=0)
    torch.testing.assert_close(out,torch.tensor([0.1,0.4,0.2,0.3]))
    torch.testing.assert_close(torch.sort(out).values,torch.sort(base).values)


def test_none_vanilla_opsd_behave_as_expected():
    base=torch.tensor([0.1,0.2,0.3])
    dc=torch.ones(3)
    mask=torch.tensor([1,0,1],dtype=torch.bool)
    kw=dict(seed=0,round_index=0,source_id='p',record_index=0)
    assert allocate_reasoning_weights(base,dc,mask,arm='none',**kw).count_nonzero()==0
    for arm in ('vanilla','opsd'):
        out=allocate_reasoning_weights(base,dc,mask,arm=arm,**kw)
        torch.testing.assert_close(out,torch.tensor([1.,0.,1.]))
