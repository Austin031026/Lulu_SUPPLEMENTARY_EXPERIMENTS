import copy
import pytest
import torch
from test_stable import fixture
from lulu.stable import prepare_weights
from lulu.persistent import allocate_roles
from lulu.hindsight_service import score_hindsight


def balanced_fixture(root):
    a, model, records, step, opt = fixture(root)
    a.method = 'ren_balanced'
    for row, source in zip(records, ['A', 'A', 'B']):
        row.update(source_id=source, truncated=False)
    return a, model, records, step, opt


@pytest.mark.parametrize('arm', ['ren', 'vanilla', 'opsd'])
def test_cost_path_keeps_weights_loss_and_gradients(tmp_path, arm):
    torch.manual_seed(42)
    a, model, records, step, _ = balanced_fixture(tmp_path)
    a.reasoning_ablation = arm
    a.match_causal_update = True
    fast_records = copy.deepcopy(records)
    prepare_weights(step, records, a, 0, 1)
    loss = step(records)
    old_gradients = torch.autograd.grad(loss, list(model.parameters()))
    a.cost_mode = True
    if arm == 'vanilla':
        for row in fast_records:
            row.pop('hindsight_hidden')
    if arm == 'opsd':
        for row in fast_records:
            row.pop('teacher_hidden')
        step.teacher_head = None
    prepare_weights(step, fast_records, a, 0, 1)
    fast_loss = step(fast_records)
    gradients = torch.autograd.grad(fast_loss, list(model.parameters()))
    for old, new in zip(records, fast_records):
        torch.testing.assert_close(old['stable_rho'], new['stable_rho'], atol=1e-7, rtol=1e-5)
    torch.testing.assert_close(loss, fast_loss, atol=1e-6, rtol=1e-5)
    for old, new in zip(old_gradients, gradients):
        torch.testing.assert_close(old, new, atol=1e-6, rtol=1e-5)


def test_vanilla_hindsight_does_not_touch_model(tmp_path):
    a, _, _, _, _ = balanced_fixture(tmp_path)
    a.cost_mode = True
    a.reasoning_ablation = 'vanilla'
    assert score_hindsight(None, None, [{}], a) == [{}]
    a.reasoning_ablation = 'opsd'
    assert allocate_roles(a)['teacher'] is None
