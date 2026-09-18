import numpy as np
import pytest
import torch
from lulu.hindsight_prompt_audit import diagnostics, sample_positions, prompt_ids, PROMPTS
from lulu.data import build_prompt_views


def test_overlap_distinguishes_distance_from_teacher_support():
    c = torch.tensor([[.4,.3,.2,.1]], dtype=torch.float64)
    t = torch.tensor([[.2,.2,.3,.3]], dtype=torch.float64)
    # A larger H shift can be entirely unhelpful to Teacher-positive recipients.
    aligned = torch.tensor([[.3,.2,.25,.25]], dtype=torch.float64)
    unaligned = torch.tensor([[.8,.1,.05,.05]], dtype=torch.float64)
    a = diagnostics(c.log(), aligned.log(), t.log(), [False,False,True,False])
    b = diagnostics(c.log(), unaligned.log(), t.log(), [False,False,True,False])
    assert a['shared_mass'].item() == pytest.approx(.2)
    assert a['shared_structural_mass'].item() == pytest.approx(.05)
    assert b['shared_mass'].item() == 0
    assert b['tv_h_c'].item() > a['tv_h_c'].item()
    z = diagnostics(c.log(), c.log(), t.log(), [False]*4)
    assert z['shared_mass'].item() == 0


def test_sampling_weights_reconstruct_region_and_bin_populations():
    mask = np.arange(8192) % 5 != 0
    rows = sample_positions(mask, 43, 17, 7)
    assert rows == sample_positions(mask, 43, 17, 7)
    assert len(set(r[0] for r in rows)) == len(rows)
    for region in (0,1):
        for b in range(4):
            assert sum(w for p,w,r,k in rows if r == region and k == b) == pytest.approx(
                np.sum(mask[2048*b:2048*(b+1)] == region))


def test_current_prompt_is_identical_and_messages_not_modified():
    class Tokenizer:
        def apply_chat_template(self, messages, **kw):
            assert kw == dict(tokenize=False, add_generation_prompt=True, enable_thinking=True)
            return repr(messages)
        def encode(self, text, **kw):
            return list(text.encode())
    tok = Tokenizer(); messages = [{'role':'user','content':'Question'}]
    current = build_prompt_views(tok, messages, ' 42 ')['hindsight_prompt_ids']
    assert prompt_ids(tok, messages, ' 42 ', 'current') == current
    assert messages == [{'role':'user','content':'Question'}]
    assert len({tuple(prompt_ids(tok,messages,'42',name)) for name in PROMPTS}) == 3
