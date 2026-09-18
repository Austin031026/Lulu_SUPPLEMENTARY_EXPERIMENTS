"""Fixed-prefix prompt comparison. These diagnostics are not training losses."""
from __future__ import annotations

import numpy as np
import torch
from lulu.data import HINDSIGHT_CONTEXT

PROMPTS = {
    'current': HINDSIGHT_CONTEXT,
    'minimal': 'The verified final answer is:\n<verified_final_answer>\n{answer}\n</verified_final_answer>\n\n'
               'Use this answer as given and reason step by step to derive it from the original problem.',
    'opsd': 'The verified final answer is:\n<verified_final_answer>\n{answer}\n</verified_final_answer>\n\n'
            'Using your own reasoning, derive this same final answer step by step from the original problem.',
}


def prompt_ids(tokenizer, messages, answer, name):
    view = [dict(m) for m in messages]
    if not view or view[-1]['role'] != 'user' or not str(answer).strip():
        raise ValueError('Need a final user message and a verified answer')
    view[-1]['content'] += '\n\n' + PROMPTS[name].format(answer=str(answer).strip())
    text = tokenizer.apply_chat_template(view, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    return tokenizer.encode(text, add_special_tokens=False)


def sample_positions(reasoning_mask, seed, reasoning_per_bin=128, control_per_bin=32):
    """Shared stratified probability sample, with inverse inclusion weights."""
    mask = np.asarray(reasoning_mask, dtype=bool)
    if len(mask) > 8192:
        raise ValueError('This audit is for existing 8k rollouts')
    rng = np.random.default_rng(seed)
    rows = []
    for region, budget in [(True, reasoning_per_bin), (False, control_per_bin)]:
        if budget < 1:
            raise ValueError('Positive position budget required')
        for b, lo in enumerate(range(0, 8192, 2048)):
            population = np.flatnonzero((mask == region) & (np.arange(len(mask)) >= lo) & (np.arange(len(mask)) < lo+2048))
            if not len(population):
                continue
            chosen = rng.choice(population, min(budget, len(population)), replace=False)
            rows.extend((int(p), len(population)/len(chosen), int(region), b) for p in chosen)
    return sorted(rows)


def structural_vocabulary(tokenizer, size):
    """Explicit lexical proxy only: special, whitespace, punctuation/symbol tokens.

    Tokens containing letters, numbers, or replacement characters are excluded.
    Mathematical punctuation is included; this is NOT a semantic style detector.
    """
    special = set(tokenizer.all_special_ids)
    mask = []
    for i in range(size):
        s = tokenizer.decode([i], skip_special_tokens=False)
        mask.append(i in special or bool(s) and '\ufffd' not in s and not any(c.isalnum() for c in s))
    return np.asarray(mask, dtype=bool)


@torch.no_grad()
def diagnostics(causal_logits, hindsight_logits, teacher_logits, structural_mask):
    dtype = torch.float64 if causal_logits.dtype == torch.float64 else torch.float32
    lc, lh, lt = [z.to(dtype).log_softmax(-1) for z in (causal_logits, hindsight_logits, teacher_logits)]
    c, h, t = lc.exp(), lh.exp(), lt.exp()
    up_t, up_h = (t-c).clamp_min(0), (h-c).clamp_min(0)
    shared = torch.minimum(up_t, up_h)
    structural = torch.as_tensor(structural_mask, dtype=torch.bool, device=c.device)
    if structural.shape != (c.shape[-1],):
        raise ValueError('Vocabulary mask shape mismatch')
    result = dict(kl_h_c=(h*(lh-lc)).sum(-1), kl_c_h=(c*(lc-lh)).sum(-1),
                  tv_h_c=.5*(h-c).abs().sum(-1), teacher_tv=up_t.sum(-1),
                  shared_mass=shared.sum(-1), shared_structural_mass=shared[:, structural].sum(-1),
                  structural_l1=(h-c).abs()[:, structural].sum(-1),
                  teacher_positive_h_uplift=(up_h*(up_t > 0)).sum(-1),
                  supported_teacher_positive_mass=(up_t*(up_h > 0)).sum(-1))
    result['shared_nonstructural_mass'] = result['shared_mass']-result['shared_structural_mass']
    if any(not bool(v.isfinite().all()) for v in result.values()):
        raise FloatingPointError('Nonfinite prompt comparison')
    return result
