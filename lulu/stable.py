"""Bounded ReN, full response control supervision and fixed-reference reverse KL.

This adopts StableOPD's reference divergence, not its golden-data mixture or
policy-gradient estimator. All distillation terms use exact full vocabularies.
"""
from __future__ import annotations
import time
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from lulu.resolved import mismatch_scores, concentration
from lulu.objective import forward_kl


def bounded_weight(causal_kl, resolved_mismatch, epsilon=1e-6):
    """Stop-gradient resolved fraction; there is no empirical mean rescaling."""
    if epsilon <= 0:
        raise ValueError('Ratio epsilon must be positive')
    return (resolved_mismatch.detach() / (causal_kl.detach().clamp_min(0) + epsilon)).clamp(0, 1)


def bounded_absolute_weight(resolved_mismatch):
    """Outcome-resolved mismatch, without threshold or empirical renormalization."""
    gap = resolved_mismatch.detach().clamp_min(0)
    return gap / (1 + gap)


def prompt_reasoning_scales(records, global_tokens, args, world):
    """N/(B * M_i * |R_ij|), since the outer coordinator divides by N.

    Empty reasoning regions contribute zero, but their prompt/rollout remains
    in B/M and their control/reference terms remain intact. Group across ranks.
    """
    def key(record):
        return str(record.get('source_id', record['index'] // args.rollouts_per_prompt))
    local = Counter(key(r) for r in records)
    gathered = [None] * world
    if world > 1:
        dist.all_gather_object(gathered, dict(local))
    else:
        gathered = [local]
    counts = Counter()
    for part in gathered:
        counts.update(part)
    if not counts:
        raise ValueError('No prompts in balanced update')
    for record in records:
        reasoning = sum(record['reasoning_mask'])
        record['reasoning_loss_scale'] = (global_tokens / (len(counts) * counts[key(record)] * reasoning)
                                           if reasoning else 0.)
        record['prompt_rollouts'] = counts[key(record)]
    return len(counts)


def reference_reverse_kl(student_logits, reference_logits):
    dtype = torch.float64 if student_logits.dtype == torch.float64 else torch.float32
    log_student = student_logits.to(dtype).log_softmax(-1)
    log_reference = reference_logits.detach().to(dtype).log_softmax(-1)
    return (log_student.exp() * (log_student-log_reference)).sum(-1)


@torch.no_grad()
def prepare_weights(step, records, a, rank, world):
    """Score ReN once, apply the requested allocation control, and persist RQ diagnostics."""
    from lulu import training as tr
    from lulu.allocation import allocate_reasoning_weights, allocation_invariants
    if a.method == 'ren_shared':
        from lulu.shared import prepare_shared
        return prepare_shared(step, records, a, rank, world)
    started = time.monotonic()
    device = tr.device_for(a)
    arm = getattr(a, 'reasoning_ablation', 'ren')
    keys = ('causal_kl', 'hindsight_kl', 'resolved_mismatch', 'raw_weight', 'old_alpha',
            'bounded_weight', 'applied_weight', 'opsd_causal_kl', 'trajectory_index', 'position')
    pieces = {key: [] for key in keys}
    local_trajectories = []
    for record in records:
        n = len(record['positions'])
        if any(key not in record for key in ('student_hidden', 'hindsight_hidden', 'teacher_hidden', 'reference_hidden', 'reasoning_mask')):
            raise ValueError('Stable ReN requires C/H/T/reference caches and structural regions')
        if len(record['reasoning_mask']) != n or record['positions'] != list(range(len(record['response_ids']))):
            raise ValueError('Stable ReN must supervise every sampled response token in order')
        reasoning = torch.tensor(record['reasoning_mask'], dtype=torch.bool)
        base_rho = torch.zeros(n, dtype=torch.float32)
        scored_dc = torch.zeros(n, dtype=torch.float32)
        scored_opsd = torch.zeros(n, dtype=torch.float32)
        per_record = {key: [] for key in ('causal_kl','hindsight_kl','resolved_mismatch','raw_weight','old_alpha','bounded_weight','opsd_causal_kl','position')}
        for start in range(0, n, a.logit_chunk_size):
            stop = min(n, start+a.logit_chunk_size)
            keep = reasoning[start:stop]
            if not keep.any():
                continue
            with tr.autocast_context(a):
                if getattr(a,'match_causal_update',False):
                    # Match live head GEMM dimensions; masking before the head can
                    # change BF16 kernels and dominate these deliberately weak scores.
                    c_full = step.frozen_head(record['student_hidden'][start:stop].to(device))
                    h_full = step.frozen_head(record['hindsight_hidden'][start:stop].to(device))
                    t_full = step.teacher_head(record['teacher_hidden'][start:stop].to(device))
                    c, h, t = c_full[keep.to(device)], h_full[keep.to(device)], t_full[keep.to(device)]
                else:
                    c = step.frozen_head(record['student_hidden'][start:stop][keep].to(device))
                    h = step.frozen_head(record['hindsight_hidden'][start:stop][keep].to(device))
                    t = step.teacher_head(record['teacher_hidden'][start:stop][keep].to(device))
            scores = mismatch_scores(c, h, t, a.top_k)
            scores['bounded_weight'] = (bounded_absolute_weight(scores['resolved_mismatch']) if a.method == 'ren_balanced'
                                        else bounded_weight(scores['causal_kl'], scores['resolved_mismatch'], a.stable_ratio_epsilon))
            # Matched-backbone OPSD uses the privileged Student as the direct target.
            # This scalar is only for the pre-update score/live correctness guard.
            h_target = h.float().softmax(-1)
            scores['opsd_causal_kl'] = forward_kl(c, h_target, reduction='none').detach()
            scored_dc[start:stop][keep] = scores['causal_kl'].cpu()
            scored_opsd[start:stop][keep] = scores['opsd_causal_kl'].cpu()
            base_rho[start:stop][keep] = scores['bounded_weight'].cpu()
            positions = np.arange(start,stop,dtype=np.int32)[keep.numpy()]
            for key in ('causal_kl','hindsight_kl','resolved_mismatch','raw_weight','old_alpha','bounded_weight','opsd_causal_kl'):
                values = scores[key].cpu().numpy()
                pieces[key].append(values); per_record[key].append(values)
            pieces['position'].append(positions); per_record['position'].append(positions)
            pieces['trajectory_index'].append(np.full(len(positions),record['index'],dtype=np.int32))

        applied = allocate_reasoning_weights(base_rho, scored_dc, reasoning, arm=arm,
            seed=a.seed, round_index=a.round, source_id=record.get('source_id', record['index']), record_index=record['index'])
        record['stable_rho'] = applied
        target_kl = scored_opsd if arm == 'opsd' else scored_dc
        record['scored_reasoning_kl_sum'] = float((applied.double()*target_kl.double()).sum())
        selected = reasoning.numpy()
        pieces['applied_weight'].append(applied[reasoning].cpu().numpy())
        invariant = allocation_invariants(base_rho, applied, reasoning)
        causal_selected = scored_dc[reasoning].double()
        applied_selected = applied[reasoning].double()
        base_selected = base_rho[reasoning].double()
        trajectory = dict(
            index=int(record['index']), source_id=str(record.get('source_id', record['index'])),
            response_tokens=int(n), reasoning_tokens=int(reasoning.sum()), truncated=bool(record.get('truncated', False)),
            prompt_rollouts=None, reasoning_loss_scale=None,
            base_weight_sum=float(base_selected.sum()), applied_weight_sum=float(applied_selected.sum()),
            base_weight_mean=float(base_selected.mean()) if base_selected.numel() else 0.,
            applied_weight_mean=float(applied_selected.mean()) if applied_selected.numel() else 0.,
            causal_kl_mean=float(causal_selected.mean()) if causal_selected.numel() else 0.,
            causal_kl_sum=float(causal_selected.sum()),
            base_weighted_teacher_kl_sum=float((base_selected*causal_selected).sum()),
            applied_weighted_teacher_kl_sum=float((applied_selected*causal_selected).sum()),
            allocation_invariants=invariant,
        )
        local_trajectories.append(trajectory)
        record.pop('student_hidden')
        # Direct OPSD needs the cached privileged states during the live loss.
        if arm != 'opsd':
            record.pop('hindsight_hidden')

    arrays = {key: np.concatenate(value) if value else np.empty(0,dtype=np.float32) for key,value in pieces.items()}
    totals = torch.tensor([float(arrays['bounded_weight'].sum(dtype=np.float64)),
                           float(arrays['applied_weight'].sum(dtype=np.float64)),
                           len(arrays['bounded_weight']), sum(len(r['positions']) for r in records),
                           sum(bool(r['positions']) for r in records)],dtype=torch.float64,device=device)
    if world > 1:
        dist.all_reduce(totals)
    rho_sum, applied_sum, reasoning_tokens, loss_tokens, trajectories = totals.tolist()
    prompt_count = prompt_reasoning_scales(records, loss_tokens, a, world) if a.method == 'ren_balanced' else None
    split = getattr(a, 'reasoning_diagnostic_split', 0)
    horizon = {}
    if split:
        bin_totals=[]
        for selected in (arrays['position'] < split, arrays['position'] >= split):
            applied=arrays['applied_weight'][selected].astype(np.float64)
            dc=arrays['causal_kl'][selected].astype(np.float64)
            bin_totals.append([int(selected.sum()), float(applied.sum()), float((applied*dc).sum())])
        values=torch.tensor(bin_totals,dtype=torch.float64,device=device)
        if world>1: dist.all_reduce(values)
        for name,value in zip(('early','late'),values.tolist()):
            count,weight_sum,weighted_kl_sum=value
            horizon[name]=dict(reasoning_tokens=int(count),weight_sum=weight_sum,
                mean_weight=weight_sum/count if count else None,
                mean_weighted_causal_kl=weighted_kl_sum/count if count else None)
    by_index = {item['index']:item for item in local_trajectories}
    for record in records:
        if record['index'] in by_index:
            item=by_index[record['index']]
            item['prompt_rollouts']=int(record.get('prompt_rollouts',1))
            item['reasoning_loss_scale']=float(record.get('reasoning_loss_scale',0.))
            item['scaled_reasoning_kl_sum']=float(record['scored_reasoning_kl_sum']*record.get('reasoning_loss_scale',1.))
            item['scaled_raw_teacher_kl_sum']=float(item['causal_kl_sum']*record.get('reasoning_loss_scale',1.))
            item['scaled_applied_teacher_kl_sum']=float(item['applied_weighted_teacher_kl_sum']*record.get('reasoning_loss_scale',1.))
    scored_loss = torch.tensor(sum(r['scored_reasoning_kl_sum']*r.get('reasoning_loss_scale',1.) for r in records),dtype=torch.float64,device=device)
    if world > 1: dist.all_reduce(scored_loss)
    expected_reasoning_loss = float(scored_loss)/max(loss_tokens,1)

    gathered = [None]*world if rank == 0 else None
    gathered_traj = [None]*world if rank == 0 else None
    if world > 1:
        dist.gather_object(arrays,gathered,dst=0)
        dist.gather_object(local_trajectories,gathered_traj,dst=0)
    elif rank == 0:
        gathered, gathered_traj = [arrays], [local_trajectories]
    diagnostics = {}
    if rank == 0:
        full = {key:np.concatenate([part[key] for part in gathered]) for key in keys}
        trajectory_rows=[item for part in gathered_traj for item in part]
        assert len(full['bounded_weight']) == int(reasoning_tokens)
        if len(full['applied_weight']) != int(reasoning_tokens):
            raise RuntimeError('Applied allocation diagnostics do not match reasoning positions')

        # Position-bin diagnostics are kept independent of any one paper figure;
        # they support horizon, Teacher-gap and scaling analyses post hoc.
        max_position = int(full['position'].max()) if len(full['position']) else -1
        position_bins = {}
        for low in range(0, max_position+1, 2048):
            high=low+2048; selected=(full['position']>=low)&(full['position']<high); count=int(selected.sum())
            if not count: continue
            base=full['bounded_weight'][selected].astype(np.float64)
            applied=full['applied_weight'][selected].astype(np.float64)
            dc=full['causal_kl'][selected].astype(np.float64)
            delta=full['resolved_mismatch'][selected].astype(np.float64)
            position_bins[f'{low}-{high}']={
                'positions':count,'positive_fraction':float(np.mean(base>0)),
                'base_weight_mean':float(base.mean()),'applied_weight_mean':float(applied.mean()),
                'base_weight_sum':float(base.sum()),'applied_weight_sum':float(applied.sum()),
                'causal_kl_mean':float(dc.mean()),'resolved_mean':float(delta.mean()),
                'applied_weighted_teacher_kl_sum':float((applied*dc).sum()),
                'applied_weighted_teacher_kl_mean':float((applied*dc).mean()),
            }
        base_total=float(full['bounded_weight'].sum(dtype=np.float64)); applied_total=float(full['applied_weight'].sum(dtype=np.float64))
        for item in position_bins.values():
            item['base_weight_mass_share']=item['base_weight_sum']/base_total if base_total else 0.
            item['applied_weight_mass_share']=item['applied_weight_sum']/applied_total if applied_total else 0.

        # Prompt-level contribution concentration is the relevant companion to
        # token-level sparsity when batch size changes.
        from collections import defaultdict
        prompt_contrib=defaultdict(float)
        for x in trajectory_rows:
            # Each rollout contribution already contains the 1/M_i scale, so
            # summing by source_id gives exactly one prompt-level contribution.
            prompt_contrib[x['source_id']] += x.get('scaled_reasoning_kl_sum',0.)/max(loss_tokens,1)
        contributions=np.asarray(list(prompt_contrib.values()),dtype=np.float64)
        contrib_sum=float(contributions.sum())
        prompt_ess=float(contrib_sum*contrib_sum/np.square(contributions).sum()) if np.square(contributions).sum()>0 else 0.
        sorted_contrib=np.sort(np.maximum(contributions,0))[::-1]
        top_n=max(1,int(np.ceil(.1*len(sorted_contrib)))) if len(sorted_contrib) else 0
        top10_prompt_share=float(sorted_contrib[:top_n].sum()/contrib_sum) if contrib_sum>0 and top_n else 0.
        raw_teacher_mass=float(full['causal_kl'].astype(np.float64).sum())
        applied_teacher_mass=float((full['applied_weight'].astype(np.float64)*full['causal_kl'].astype(np.float64)).sum())
        # The training objective is trajectory/prompt balanced, so keep both the
        # token-global retention ratio and the ratio under the *actual* reasoning
        # reduction.  The latter is the scientifically relevant quantity when
        # comparing allocation arms or Teacher strength.
        raw_teacher_mass_balanced=float(sum(x.get('scaled_raw_teacher_kl_sum',0.) for x in trajectory_rows))
        applied_teacher_mass_balanced=float(sum(x.get('scaled_applied_teacher_kl_sum',0.) for x in trajectory_rows))
        retained_token_global=applied_teacher_mass/raw_teacher_mass if raw_teacher_mass else 0.
        retained_balanced=applied_teacher_mass_balanced/raw_teacher_mass_balanced if raw_teacher_mass_balanced else 0.
        diagnostics = {'round':a.round,'before_update':True,'reasoning_positions':int(reasoning_tokens),
                       'loss_tokens':int(loss_tokens),'control_tokens':int(loss_tokens-reasoning_tokens),
                       'rho_sum':rho_sum,'rho_mean':rho_sum/max(reasoning_tokens,1),
                       'applied_weight_sum':applied_sum,'applied_weight_mean':applied_sum/max(reasoning_tokens,1),
                       'causal_update_matched':getattr(a,'match_causal_update',False),
                       'reasoning_loss_score_expected':expected_reasoning_loss,
                       'normalization':('reasoning_tokens_then_rollout_then_prompt; control/reference_global_tokens'
                                        if a.method == 'ren_balanced' else 'global_generated_token_count'),
                       'weight_function':'g/(1+g)' if a.method == 'ren_balanced' else 'clip(g/(DC+epsilon),0,1)',
                       'prompt_count':prompt_count,'position_bins':position_bins,
                       'reasoning_horizon':horizon,'reasoning_diagnostic_split':split,
                       'raw_ren_weight_mean':float(full['raw_weight'].mean()) if reasoning_tokens else 0.,
                       'reference_checkpoint':str(Path(a.output_dir)/'checkpoints/round_000000'),
                       'reference_kl_coef':a.reference_kl_coef,'reasoning_ablation':arm,'applied_allocation':arm,
                       'rho_diagnostic_scope':'bounded ReN score retained for every matched allocation arm',
                       'causal_kl_mean':float(full['causal_kl'].mean()) if reasoning_tokens else 0.,
                       'hindsight_kl_mean':float(full['hindsight_kl'].mean()) if reasoning_tokens else 0.,
                       'resolved_mean':float(full['resolved_mismatch'].mean()) if reasoning_tokens else 0.,
                       'resolved_positive_fraction':float(np.mean(full['resolved_mismatch']>0)) if reasoning_tokens else 0.,
                       # Backward-compatible alias now follows the actual prompt-balanced
                       # reasoning measure; the explicitly named token-global field is also saved.
                       'teacher_supervision_retained_fraction':retained_balanced,
                       'teacher_supervision_retained_fraction_prompt_balanced':retained_balanced,
                       'teacher_supervision_retained_fraction_token_global':retained_token_global,
                       'raw_teacher_kl_mass':raw_teacher_mass,'applied_teacher_kl_mass':applied_teacher_mass,
                       'raw_teacher_kl_mass_prompt_balanced':raw_teacher_mass_balanced,
                       'applied_teacher_kl_mass_prompt_balanced':applied_teacher_mass_balanced,
                       'prompt_reasoning_loss_ess':prompt_ess,'prompt_reasoning_loss_top10_share':top10_prompt_share,
                       'concentration':{key:concentration(full[key]) for key in ('causal_kl','raw_weight','bounded_weight','applied_weight')}}
        path=Path(a.output_dir)/'diagnostics'/f'round_{a.round:04d}'
        path.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path/'position_scores.npz',**full)
        tr.atomic_json(path/'summary.json',diagnostics)
        with (path/'trajectory_scores.jsonl').open('w') as handle:
            for item in sorted(trajectory_rows,key=lambda x:x['index']):
                import json
                handle.write(json.dumps(item,ensure_ascii=False)+'\n')

    return dict(reasoning_loss_score_expected=expected_reasoning_loss,
                causal_update_matched=getattr(a,'match_causal_update',False),
                reasoning_ablation=arm,
                applied_reasoning_weight_sum=applied_sum,
                weight_sum=applied_sum+(loss_tokens-reasoning_tokens), rho_sum=rho_sum,
                reasoning_horizon=horizon, reasoning_diagnostic_split=split,
                prompt_count=prompt_count, mean_reasoning_tokens_per_rollout=reasoning_tokens/max(trajectories,1),
                reasoning_tokens=int(reasoning_tokens),control_tokens=int(loss_tokens-reasoning_tokens),
                loss_tokens=int(loss_tokens),supervised_trajectories=int(trajectories),
                weight_scoring_seconds=time.monotonic()-started,diagnostics=diagnostics)

def stable_forward(step, records):
    """Return a token loss SUM; the coordinator divides once by global N."""
    from lulu import training as tr
    a = step.args
    if step.reference_head is None:
        raise RuntimeError('Frozen initial reference head has not been installed')
    hidden = tr.selected_hidden(step.student,step.tok,records,'causal_prompt_ids',a)
    head = tr.base_model(step.student).get_output_embeddings()
    graph_zero = hidden[0].sum()*0.+head.weight.reshape(-1)[0]*0.
    components = torch.zeros(3,device=hidden[0].device)
    budget = torch.zeros(2,device=hidden[0].device)
    horizon_sums = torch.zeros((2,2),device=hidden[0].device)
    split = getattr(a, 'reasoning_diagnostic_split', 0)
    mode = getattr(step, 'loss_component_mode', None)
    shared = a.method == 'ren_shared'
    opsd = getattr(a, 'reasoning_ablation', 'ren') == 'opsd'
    for record, live in zip(records,hidden):
        n=len(record['positions'])
        if not n or record.get('dummy',False):
            graph_zero=graph_zero+live.sum()*0.+head.weight.reshape(-1)[0]*0.
            continue
        for start in range(0,n,a.logit_chunk_size):
            stop=min(n,start+a.logit_chunk_size)
            if mode == 3 and start >= split: continue
            if mode == 4 and stop <= split: continue
            th=record['teacher_hidden'][start:stop].to(live.device)
            rh=record['reference_hidden'][start:stop].to(live.device)
            rho=(torch.ones(stop-start,device=live.device) if shared else record['stable_rho'][start:stop].to(live.device))
            ch=record['student_hidden'][start:stop].to(live.device) if shared else th[:,:0]
            hh=record['hindsight_hidden'][start:stop].to(live.device) if (shared or opsd) else th[:,:0]
            reasoning=torch.tensor(record['reasoning_mask'][start:stop],dtype=torch.bool,device=live.device)
            early = torch.arange(start,stop,device=live.device) < split if split else torch.zeros(stop-start,dtype=torch.bool,device=live.device)
            if mode == 3: reasoning = reasoning & early
            if mode == 4: reasoning = reasoning & ~early
            def chunk_loss(states, teacher_states, reference_states, weights, mask, early_mask, causal_states, hindsight_states):
                student_logits=head(states)
                zero=student_logits.sum()*0.
                ren, control, ref, early_ren, late_ren = zero, zero, zero, zero, zero
                if mode in (None, 0, 1, 3, 4):
                    with torch.no_grad():
                        teacher_target=step.teacher_head(teacher_states).float().softmax(-1)
                        distill_target=(step.frozen_head(hindsight_states).float().softmax(-1) if opsd else teacher_target)
                    if mode in (None, 1) or not shared:
                        kd=forward_kl(student_logits,distill_target,reduction='none')
                    if mode in (None, 0, 3, 4):
                        if shared:
                            from lulu.shared import shared_positive_target
                            with torch.no_grad():
                                causal=step.frozen_head(causal_states[mask]).float().softmax(-1)
                                hindsight=step.frozen_head(hindsight_states[mask]).float().softmax(-1)
                                target=shared_positive_target(causal,hindsight,teacher_target[mask])
                            # No scalar gate and no correction-mass renormalization.
                            token_kl=forward_kl(student_logits[mask],target,reduction='none')
                            weighted=torch.zeros_like(weights).masked_scatter(mask,token_kl)
                        else:
                            weighted=kd*weights.detach()*mask
                        ren=weighted.sum()
                        if split:
                            early_ren=(weighted*early_mask).sum()
                            late_ren=(weighted*(~early_mask)).sum()
                    if mode in (None, 1): control=(kd*(~mask)).sum()
                if mode in (None, 2):
                    with torch.no_grad(): reference_logits=step.reference_head(reference_states)
                    ref=reference_reverse_kl(student_logits,reference_logits).sum()
                return torch.stack((ren,control,ref,early_ren,late_ren))
            arguments=(live[start:stop],th,rh,rho,reasoning,early,ch,hh)
            values=checkpoint(chunk_loss,*arguments,use_reentrant=False) if a.gradient_checkpointing else chunk_loss(*arguments)
            scale=record.get('reasoning_loss_scale',1.)
            ren=values[0]*scale
            if split: horizon_sums=horizon_sums+torch.stack((values[3:5].detach(),values[3:5].detach()*scale),dim=1)
            budget=budget+torch.stack((values[0].detach(),ren.detach()*bool(record.get('truncated',False))))
            components=components+torch.stack((ren,values[1],values[2]))
    step.last_reasoning_horizon_sums=horizon_sums
    step.last_loss_components=components.detach()
    step.last_budget_components=budget
    if mode is not None:
        component_index=0 if mode in (3,4) else mode
        return graph_zero+components[component_index]*(a.reference_kl_coef if mode==2 else getattr(a,'control_loss_coef',1.) if mode==1 else 1.)
    return graph_zero+components[0]+getattr(a,'control_loss_coef',1.)*components[1]+a.reference_kl_coef*components[2]
