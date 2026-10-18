"""Essential matched-backbone score preparation, with identical live losses.

No hindsight forward/score for vanilla; no Teacher forward/score for OPSD.
Control tokens in matched OPSD use the privileged Student, just as in the
historical stable_forward implementation. The frozen reference stays in all arms.
"""
import time
import torch
import torch.distributed as dist

from lulu.objective import forward_kl
from lulu.stable import bounded_absolute_weight, prompt_reasoning_scales
from lulu.allocation import allocate_reasoning_weights, allocation_invariants


@torch.no_grad()
def prepare_cost_weights(step, records, a, rank, world):
    from lulu import training as tr
    from lulu.cost import synchronize
    synchronize()
    started = time.monotonic()
    arm = a.reasoning_ablation
    device = tr.device_for(a)
    for record in records:
        n = len(record['positions'])
        if record['positions'] != list(range(len(record['response_ids']))) or len(record['reasoning_mask']) != n:
            raise ValueError('Cost mode requires full-response positions and masks')
        weights = torch.zeros(n)
        target_scores = torch.zeros(n)
        causal_entropy = torch.zeros(n) if arm == 'entropy_matched' else None
        for start in range(0, n, a.logit_chunk_size):
            stop = min(n, start + a.logit_chunk_size)
            mask = torch.tensor(record['reasoning_mask'][start:stop], dtype=torch.bool, device=device)
            if not mask.any():
                continue
            with tr.autocast_context(a):
                c = step.frozen_head(record['student_hidden'][start:stop].to(device))[mask]
                if arm == 'opsd':
                    target = step.frozen_head(record['hindsight_hidden'][start:stop].to(device))[mask]
                else:
                    target = step.teacher_head(record['teacher_hidden'][start:stop].to(device))[mask]
                if arm in ('ren', 'low_kl_matched', 'entropy_matched'):
                    h = step.frozen_head(record['hindsight_hidden'][start:stop].to(device))[mask]
            dtype = torch.float64 if c.dtype == torch.float64 else torch.float32
            t_log = target.to(dtype).log_softmax(-1)
            c_log = c.to(dtype).log_softmax(-1)
            q = t_log.exp()
            dc = (q * (t_log-c_log)).sum(-1) if arm != 'opsd' else forward_kl(c, target.float().softmax(-1), reduction='none')
            w = (bounded_absolute_weight((q * (h.to(dtype).log_softmax(-1)-c_log)).sum(-1))
                 if arm in ('ren', 'low_kl_matched', 'entropy_matched') else torch.ones_like(dc))
            if not torch.isfinite(dc).all() or not torch.isfinite(w).all():
                raise FloatingPointError('Non-finite cost-mode scores')
            cpu_mask = mask.cpu()
            weights[start:stop][cpu_mask] = w.cpu()
            target_scores[start:stop][cpu_mask] = dc.cpu()
            if causal_entropy is not None:
                entropy = -(c_log.exp() * c_log).sum(-1)
                if not torch.isfinite(entropy).all():
                    raise FloatingPointError('Non-finite causal Student entropy')
                causal_entropy[start:stop][cpu_mask] = entropy.cpu()
        if arm in ('low_kl_matched', 'entropy_matched'):
            reasoning = torch.tensor(record['reasoning_mask'], dtype=torch.bool)
            applied = allocate_reasoning_weights(
                weights, target_scores, reasoning, causal_entropy=causal_entropy,
                arm=arm, seed=a.seed, round_index=a.round,
                source_id=record.get('source_id', record['index']), record_index=record['index'])
            if not allocation_invariants(weights, applied, reasoning)['same_multiset']:
                raise RuntimeError(f'{arm} changed the per-trajectory HinD weight multiset')
            weights = applied
        record['stable_rho'] = weights
        record['scored_reasoning_kl_sum'] = float((weights.double() * target_scores.double()).sum())
        record.pop('student_hidden')
        if arm != 'opsd':
            record.pop('hindsight_hidden', None)
    totals = torch.tensor([sum(len(r['positions']) for r in records),
                           sum(sum(r['reasoning_mask']) for r in records), len(records),
                           sum(float(r['stable_rho'].sum()) for r in records)], dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(totals)
    tokens, reasoning, trajectories, weight_sum = totals.tolist()
    prompts = prompt_reasoning_scales(records, tokens, a, world)
    expected = torch.tensor(sum(r['scored_reasoning_kl_sum'] * r['reasoning_loss_scale'] for r in records), dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(expected)
    synchronize()
    return dict(loss_tokens=int(tokens), reasoning_tokens=int(reasoning), control_tokens=int(tokens-reasoning),
                supervised_trajectories=int(trajectories), prompt_count=prompts,
                mean_reasoning_tokens_per_rollout=reasoning/max(trajectories, 1),
                reasoning_loss_score_expected=float(expected)/max(tokens, 1),
                applied_reasoning_weight_sum=weight_sum, weight_sum=weight_sum+tokens-reasoning,
                causal_update_matched=a.match_causal_update, reasoning_ablation=arm,
                weight_scoring_seconds=time.monotonic()-started,
                diagnostics={'scope': 'essential cost-mode scoring; no optional HinD diagnostics'})
