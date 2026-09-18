import numpy as np
import pytest
from lulu.semantic_audit import residual_coupling,sample_categorical,continuation_budget


def test_residual_estimator_exactly_recovers_original_intervention_effect():
    p=np.array([.5,.3,.2]);q=np.array([.2,.6,.2]);eta=.2
    success=np.array([.1,.8,.4])
    mass,control,intervention=residual_coupling(p,q,eta)
    expected=np.dot((1-eta)*p+eta*q-p,success)
    assert mass*np.dot(intervention-control,success)==pytest.approx(expected)
    assert np.dot(control,intervention)==pytest.approx(0)
    assert mass==pytest.approx(eta*.5*np.abs(q-p).sum())


def test_residual_estimator_with_arbitrary_success_and_random_distributions():
    rng=np.random.default_rng(19)
    for _ in range(20):
        p=rng.dirichlet(np.ones(100));q=rng.dirichlet(np.ones(100));v=rng.uniform(size=100)
        m,c,i=residual_coupling(p,q,.2)
        assert m*np.dot(i-c,v)==pytest.approx(.2*np.dot(q-p,v),abs=1e-14)
    m,c,i=residual_coupling(p,p,.2);assert m==0 and not np.any(c) and not np.any(i)


def test_context_budget_counts_prefix_and_forced_token_without_truncation():
    assert continuation_budget(500,8000)==24767
    assert continuation_budget(10000,8000)==22831
    assert sample_categorical([.2,.3,.5],0)==0
    assert sample_categorical([.2,.3,.5],.21)==1
    with pytest.raises(ValueError):continuation_budget(40900,8000)
