"""Resident Student DDP, synchronized privileged Student, and Teacher TP.

Only microbatches inside one frozen on-policy round overlap. A round barrier
precedes the optimizer step; new rollouts wait for the updated privileged view.
Hidden-state caches stay in each Student process's RAM, never on the filesystem.
"""
from __future__ import annotations

from collections import deque
import contextlib
import copy
import hashlib
import json
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import os
import pickle
from pathlib import Path
import random
import socket
import sys
import threading
import time
import traceback
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from lulu import training as tr
from lulu.checkpoints import CheckpointManager


def allocate_roles(a):
    needs_h = a.method in ('ren_shared', 'ren_balanced', 'ren_stable', 'ren_resolved', 'ren_opd', 'ren_graft', 'opsd', 'union_topk')
    needs_t = a.method != 'opsd'
    if a.cpu:
        return {'student': [], 'hindsight': [] if needs_h else None,
                'teacher': [] if needs_t else None}
    available = tr.gpu_ids(a)
    selected = {}
    for role, raw, needed in [('student', a.student_gpus, True),
                             ('hindsight', a.hindsight_gpus, needs_h),
                             ('teacher', a.teacher_gpus, needs_t)]:
        if not needed:
            if raw != 'auto':
                raise ValueError(f'{role} GPUs are unused by {a.method}; leave auto')
            selected[role] = None
        elif raw != 'auto':
            ids = [x.strip() for x in raw.split(',') if x.strip()]
            if not ids or len(ids) != len(set(ids)) or not set(ids) <= set(available):
                raise ValueError(f'Invalid {role} GPU IDs; must be unique members of --gpus')
            selected[role] = ids
    used = [x for ids in selected.values() if ids for x in ids]
    if len(used) != len(set(used)):
        raise ValueError('Student, hindsight and Teacher GPU groups must be disjoint')
    free = [x for x in available if x not in used]
    for role, count in [('teacher', a.teacher_gpus_per_worker), ('hindsight', 1)]:
        if role not in selected:
            if len(free) < count:
                raise ValueError('Insufficient GPUs for resident roles; specify smaller disjoint groups or --backend staged')
            selected[role] = free[-count:]
            free = free[:-count]
    if 'student' not in selected:
        selected['student'] = free
    if not selected['student'] or (selected['hindsight'] is not None and len(selected['hindsight']) != 1):
        raise ValueError('Need at least one Student GPU and exactly one privileged Student GPU')
    return selected



def resume_scientific_config(config):
    """Allow physical placement changes while keeping role sizes and science fixed."""
    operational = {'train_data', 'output_dir', 'save_every', 'retain_checkpoints',
                   'worker_timeout', 'gpus', 'student_gpus', 'hindsight_gpus', 'teacher_gpus'}
    normalized = {k: v for k, v in config.items() if k not in operational}
    # Explicit assignments can be recovered even if the original devices are busy.
    # allocate_roles also rejects duplicates, overlapping roles and unknown IDs.
    roles = allocate_roles(SimpleNamespace(**config))
    normalized['resume_role_sizes'] = {role: None if ids is None else len(ids)
                                       for role, ids in roles.items()}
    return normalized


def _port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def _configure(ids, rank, world):
    os.environ.update(CUDA_VISIBLE_DEVICES=','.join(ids), LOCAL_RANK=str(rank),
                      RANK=str(rank), WORLD_SIZE=str(world), TOKENIZERS_PARALLELISM='false')
    os.environ.setdefault('NCCL_SOCKET_IFNAME', 'lo')
    os.environ.setdefault('GLOO_SOCKET_IFNAME', 'lo')
    torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))


def trainable_snapshot(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def head_from_state(state, device):
    weight = state['weight'].to(device)
    head = torch.nn.Linear(weight.shape[1], weight.shape[0], bias=state.get('bias') is not None,
                           device=device, dtype=weight.dtype)
    head.weight = torch.nn.Parameter(weight, requires_grad=False)
    if state.get('bias') is not None:
        head.bias = torch.nn.Parameter(state['bias'].to(device), requires_grad=False)
    return head.eval()


def collect_batches(student, tok, rows, a, rank, world, rollout_client=None):
    """Yield local cache and small service payloads as each rollout batch finishes."""
    chosen = tr.schedule(len(rows), a.global_batch_prompts, a.round, a.seed)
    jobs = [(i*a.rollouts_per_prompt+j, rows[k]) for i, k in enumerate(chosen)
            for j in range(a.rollouts_per_prompt)][rank::world]
    eos = student.generation_config.eos_token_id or tok.eos_token_id
    eos_set = set(eos if isinstance(eos, list) else [eos])
    head = tr.base_model(student).get_output_embeddings()
    k = min(a.top_k, head.weight.shape[0])
    student.eval()
    # Generation uses KV cache; gradient checkpointing is enabled only for update.
    if a.gradient_checkpointing:
        student.gradient_checkpointing_disable()
    tr.seed_all(a.seed + 100003*a.round + rank)
    for start in range(0, len(jobs), a.rollout_batch_size):
        records = []
        for index, row in jobs[start:start+a.rollout_batch_size]:
            views = tr.build_prompt_views(tok, row['messages'], row['gold_answer'], enable_thinking=True)
            cp, hp = views['causal_prompt_ids'], views['hindsight_prompt_ids']
            if max(len(cp), len(hp)) > a.max_prompt_tokens or max(len(cp), len(hp))+a.max_new_tokens > a.max_sequence_tokens:
                raise ValueError(f'Prompt {row["id"]} exceeds configured context budget')
            records.append(dict(index=index, source_id=row['id'], causal_prompt_ids=cp,
                                hindsight_prompt_ids=hp, snapshot_round=a.round, gold_answer=row['gold_answer']))
        if rollout_client is not None:
            responses, finish_reasons = rollout_client.generate(records, a.round)
        else:
            batch = tr.padded_batch([r['causal_prompt_ids'] for r in records], tok.pad_token_id,
                                   tr.device_for(a), left=True)
            batch.pop('position_ids')
            with torch.inference_mode():
                output = student.generate(**batch, max_new_tokens=a.max_new_tokens, do_sample=True,
                    temperature=a.temperature, top_p=a.top_p, top_k=a.rollout_top_k, repetition_penalty=1.0,
                    pad_token_id=tok.pad_token_id, eos_token_id=eos, use_cache=True)
            responses = output[:, batch['input_ids'].shape[1]:].tolist()
            finish_reasons = [None]*len(responses)
        for r, response, finish in zip(records, responses, finish_reasons):
            stop = next((i+1 for i, token in enumerate(response) if token in eos_set), len(response))
            r['response_ids'] = response[:stop]
            if a.method in ('ren_stable', 'ren_shared', 'ren_balanced'):
                from lulu.data import structured_response_regions
                regions = structured_response_regions(tok, r['response_ids'], prompt_ids=r['causal_prompt_ids'])
                r['reasoning_mask'] = regions['reasoning']
                r['positions'] = list(range(len(r['response_ids'])))
                r['finish_reason'] = finish or ('stop' if response[:stop] and response[stop-1] in eos_set else 'length')
                r['has_stop_token'] = bool(r['response_ids'] and r['response_ids'][-1] in eos_set)
                if finish == 'stop' and not r['has_stop_token']:
                    raise RuntimeError('vLLM omitted the sampled stop token; refusing incomplete termination supervision')
            else:
                mask = tr.reasoning_token_mask(tok, r['response_ids'], prompt_ids=r['causal_prompt_ids'])
                r['positions'] = [i for i, keep in enumerate(mask) if keep]
            r['truncated'] = (finish == 'length') if finish else not bool(response[:stop] and response[stop-1] in eos_set)
        payloads = []
        causal_batch_size = a.train_micro_batch_size if getattr(a,'match_causal_update',False) else a.score_batch_size
        with torch.inference_mode():
            for offset in range(0, len(records), causal_batch_size):
                part = records[offset:offset+causal_batch_size]
                hidden = tr.selected_hidden(student, tok, part, 'causal_prompt_ids', a)
                for r, h in zip(part, hidden):
                    # Only this worker retains the large causal hidden states.
                    with torch.inference_mode(False):
                        r['student_hidden'] = h.detach().to('cpu', copy=True)
                    payload = {key: r[key] for key in ('index', 'causal_prompt_ids',
                               'hindsight_prompt_ids', 'response_ids', 'positions')}
                    if a.method in ('ren_graft', 'causal_topk', 'union_topk'):
                        topk = [head(h[pos:pos+a.logit_chunk_size]).float().topk(k, -1).indices.cpu()
                                for pos in range(0, len(h), a.logit_chunk_size)]
                        payload['causal_topk_ids'] = torch.cat(topk) if topk else torch.empty((0, k), dtype=torch.long)
                    r.pop('hindsight_prompt_ids')
                    payloads.append(payload)
        yield records, payloads


def checkpoint_manager(a):
    return CheckpointManager(a.output_dir, a.save_every,
        retain_steps=[int(x) for x in getattr(a, 'retain_checkpoints', '').split(',') if x.strip()],
        index_unit='rounds' if a.method in ('ren_resolved', 'ren_stable', 'ren_shared', 'ren_balanced') else 'updates')


def audit_rollouts(records, tok, a, rank):
    """Keep every rollout and verifier result as metadata, without filtering."""
    from lulu import benchmark_parser
    path=Path(a.output_dir)/'rollouts'/f'round_{a.round:04d}'/f'shard-{rank:03d}.jsonl'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as output:
        for record in records:
            text=tok.decode(record['response_ids'],skip_special_tokens=True)
            prediction=benchmark_parser.extract_answer(text,'math500')
            try:
                correct=bool(benchmark_parser.math_equal(prediction,record['gold_answer']))
                error=None
            except Exception as exc:
                correct=None;error=str(exc)
            value={key:record[key] for key in ('index','source_id','snapshot_round','causal_prompt_ids',
                                              'response_ids','positions','truncated','gold_answer')}
            for key in ('reasoning_mask', 'finish_reason', 'has_stop_token'):
                if key in record: value[key] = record[key]
            value.update(response=text,prediction=prediction,correct=correct,verifier_error=error,
                         response_tokens=len(record['response_ids']), rollout_checkpoint=record.get('rollout_checkpoint'))
            output.write(json.dumps(value,ensure_ascii=False)+'\n')


def update_records(step_model, distributed, optimizer, records, dummy, a, rank, world):
    """One global update: resolved-token mean or legacy nonempty-trajectory mean."""
    student = step_model.student
    student.train()
    if a.gradient_checkpointing:
        student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    for record in records:
        if record['snapshot_round'] != a.round:
            raise RuntimeError('Stale Student rollout snapshot')
        if a.method != 'opsd' and not record.get('teacher_scored'):
            raise RuntimeError('Missing Teacher target')
    resolved = a.method == 'ren_resolved'
    stable = a.method in ('ren_stable', 'ren_shared', 'ren_balanced')
    weight_stats = {}
    if stable:
        from lulu.stable import prepare_weights
        weight_stats = prepare_weights(step_model, records, a, rank, world)
    if resolved:
        from lulu.resolved import prepare_weights
        weight_stats = prepare_weights(step_model, records, a, rank, world)
    expected = a.global_batch_prompts*a.rollouts_per_prompt
    local = list(records)
    random.Random(a.seed+1009*a.round+rank).shuffle(local)
    target_count = math.ceil(expected/world)
    local.extend([dict(dummy, dummy=True)]*(target_count-len(local)))
    optimizer.zero_grad(set_to_none=True)
    device = tr.device_for(a)
    loss_sum = torch.zeros((), device=device)
    component_sum = torch.zeros(3, device=device) if stable else None
    budget_sum = torch.zeros(2, device=device) if stable else None
    horizon_sum = torch.zeros((2,2), device=device) if stable else None
    counts = torch.tensor([sum(len(r['positions']) for r in records),
                           sum(bool(r['positions']) for r in records)], device=device, dtype=torch.long)
    started = time.monotonic()
    denominator = weight_stats['loss_tokens'] if stable else expected
    if stable and not denominator:
        raise RuntimeError('No generated tokens for stable distillation')
    if resolved:
        positions = weight_stats['reasoning_tokens']
        if not positions:
            raise RuntimeError('No reasoning tokens in round; inspect thinking markers')
        if weight_stats['weight_sum'] == 0:
            return dict(round=a.round, objective_loss=0., forward_kl=0., grad_norm=0.,
                trajectories=expected, skipped_update=True, update_seconds=time.monotonic()-started,
                **weight_stats)
        denominator = weight_stats['weight_sum'] + a.resolved_weight_epsilon*positions
    every = getattr(a, 'gradient_norm_every', 0)
    if stable and every and (a.round == 0 or (a.round+1) % every == 0):
        from lulu.gradient_diagnostics import component_gradient_norms
        weight_stats.update(component_gradient_norms(step_model, distributed, optimizer, local, a, world, denominator))
    for start in range(0, len(local), a.train_micro_batch_size):
        batch = local[start:start+a.train_micro_batch_size]
        if any(r['snapshot_round'] != a.round for r in batch):
            raise RuntimeError('Stale Student rollout snapshot')
        if a.method != 'opsd' and any(not r.get('teacher_scored') and not r.get('dummy') for r in batch):
            raise RuntimeError('Missing Teacher target')
        last = start+a.train_micro_batch_size >= len(local)
        context = distributed.no_sync() if world > 1 and not last else contextlib.nullcontext()
        with context:
            value = distributed(batch)
            (value*(world/denominator)).backward()
        loss_sum += value.detach()
        if stable:
            component_sum += step_model.last_loss_components
            budget_sum += step_model.last_budget_components
            horizon_sum += step_model.last_reasoning_horizon_sums
    if world > 1:
        dist.all_reduce(loss_sum)
        dist.all_reduce(counts)
        if stable:
            dist.all_reduce(component_sum)
            dist.all_reduce(budget_sum)
            dist.all_reduce(horizon_sum)
    positions, active = counts.tolist()
    if not active:
        raise RuntimeError('No reasoning tokens in round; increase rollout budget or inspect thinking markers')
    if not (resolved or stable):
        for p in student.parameters():
            if p.grad is not None:
                p.grad.mul_(expected/active)
    grad = torch.nn.utils.clip_grad_norm_(student.parameters(), a.max_grad_norm, error_if_nonfinite=True)
    if not torch.isfinite(loss_sum):
        raise FloatingPointError('Non-finite distillation loss')
    if stable and getattr(a,'match_causal_update',False):
        expected_reason = weight_stats['reasoning_loss_score_expected']
        observed_reason = float(component_sum[0]/denominator)
        discrepancy = abs(observed_reason-expected_reason)
        weight_stats.update(reasoning_score_live_abs_error=discrepancy,
            reasoning_score_live_relative_error=discrepancy/max(abs(expected_reason),1e-12))
        if discrepancy > 1e-7 + 1e-4*abs(expected_reason):
            raise RuntimeError(f'Causal score/live mismatch before optimizer step: expected={expected_reason}, observed={observed_reason}')
    optimizer.step()
    objective_value = loss_sum.item()/(denominator if (resolved or stable) else active)
    if stable:
        ren, control, reference = (component_sum/denominator).tolist()
        weight_stats.update(ren_loss=ren, answer_stop_loss=control, reference_kl=reference,
                            reference_penalty=a.reference_kl_coef*reference,
                            control_loss_coef=getattr(a,'control_loss_coef',1.),
                            control_penalty=getattr(a,'control_loss_coef',1.)*control)
        weight_stats.update(reasoning_loss_token_global=float(budget_sum[0]/denominator),
                            reasoning_loss_rollout_balanced=ren,
                            capped_reasoning_loss_share=float(budget_sum[1]/component_sum[0]) if component_sum[0]>0 else 0.)
        if getattr(a, 'reasoning_diagnostic_split', 0):
            for name, (raw, balanced) in zip(('early', 'late'), horizon_sum.tolist()):
                item=weight_stats['reasoning_horizon'][name];count=item['reasoning_tokens']
                item.update(weighted_kl_sum=raw, mean_weighted_kl=raw/count if count else None,
                    reasoning_objective_contribution=balanced/denominator,
                    reasoning_objective_share=balanced/float(component_sum[0]) if component_sum[0]>0 else 0.)
            weight_stats['reasoning_mean_weighted_kl']=float(budget_sum[0])/max(weight_stats['reasoning_tokens'],1)

    return {'round': a.round, 'completed_updates': a.round+1, 'objective_loss': objective_value,
            'forward_kl': objective_value,  # legacy field name kept for older analysis scripts
            'grad_norm': grad.item(), 'reasoning_tokens': positions, 'supervised_trajectories': active,
            'trajectories': expected, 'update_seconds': time.monotonic()-started,
            'skipped_update': False, **weight_stats}


def student_optimizer(student, a, world):
    parameters = [p for p in student.parameters() if p.requires_grad]
    kwargs = dict(lr=a.learning_rate, weight_decay=a.weight_decay)
    if getattr(a, 'optimizer_state_sharding', False):
        if world < 2:
            raise ValueError('Optimizer state sharding requires at least two Student ranks')
        from torch.distributed.optim import ZeroRedundancyOptimizer
        return ZeroRedundancyOptimizer(parameters, optimizer_class=torch.optim.AdamW,
                                       foreach=False, overlap_with_ddp=False, **kwargs)
    return torch.optim.AdamW(parameters, **kwargs)


def consolidate_optimizer(optimizer, a):
    # Every rank participates, before rank zero serializes the ordinary full state.
    if getattr(a, 'optimizer_state_sharding', False):
        optimizer.consolidate_state_dict(to=0)


def student_worker(a, rank, world, ids, port, conn):
    rollout_client = None
    try:
        _configure(ids, rank, world)
        device = tr.device_for(a)
        if world > 1:
            dist.init_process_group('gloo' if a.cpu else 'nccl', init_method=f'tcp://127.0.0.1:{port}',
                                    rank=rank, world_size=world)
        tr.seed_all(a.seed)
        initial = Path(a.initial_checkpoint) if a.initial_checkpoint else None
        student = tr.load_student(a, initial, trainable=True)
        if initial is None and a.lora_rank:
            from peft import LoraConfig, get_peft_model
            student = get_peft_model(student, LoraConfig(r=a.lora_rank, lora_alpha=a.lora_alpha,
                lora_dropout=0., target_modules=a.lora_target_modules.split(','), bias='none', task_type='CAUSAL_LM'))
        if not a.lora_rank and any(not p.requires_grad for p in student.parameters()):
            raise RuntimeError('Full Student training has unexpectedly frozen parameters')
        parameter_counts = {'total': sum(p.numel() for p in student.parameters()),
            'trainable': sum(p.numel() for p in student.parameters() if p.requires_grad),
            'parameter_dtype': str(next(student.parameters()).dtype)}
        tr.disable_dropout(student)
        tok = tr.load_tokenizer(a.model)
        rows = tr.load_prepared_jsonl(a.train_data)
        optimizer = student_optimizer(student, a, world)
        if initial and (initial/'optimizer.pt').exists():
            optimizer.load_state_dict(tr.load_tensor_file(initial/'optimizer.pt'))
        head = tr.base_model(student).get_output_embeddings()
        frozen_head = copy.deepcopy(head).requires_grad_(False) if any(p.requires_grad for p in head.parameters()) else head
        step_model = tr.DistillationStep(student, frozen_head, None, tok, a)
        distributed = DDP(step_model, device_ids=[device.index] if device.type == 'cuda' else None,
                          broadcast_buffers=False, gradient_as_bucket_view=True) if world > 1 else step_model
        manager = checkpoint_manager(a) if rank == 0 else None
        if initial is None:
            consolidate_optimizer(optimizer, a)
        if rank == 0 and initial is None:
            initial = manager.save(student, tok, optimizer, 0,
                {'completed_rounds': 0, 'completed_updates': 0, 'model': a.model, 'method': a.method})
        conn.send({'op': 'ready', 'role': 'student', 'rank': rank,
                   'initial_checkpoint': str(initial) if initial else None, 'parameter_counts': parameter_counts})
        optimizer_updates = 0
        if initial and (initial/'lulu_state.json').exists():
            optimizer_updates = json.loads((initial/'lulu_state.json').read_text()).get('completed_updates', 0)
        cache = {}
        while True:
            command = conn.recv()
            op = command['op']
            if op == 'stop':
                conn.send({'op': 'stopped', 'role': 'student', 'rank': rank})
                break
            if op == 'snapshot':
                conn.send({'op': 'snapshot', 'round': command['round'], 'state': trainable_snapshot(student)})
            elif op in ('teacher_head', 'reference_head'):
                setattr(step_model, op, head_from_state(command['state'], device))
                conn.send({'op': 'head_ready', 'rank': rank})
            elif op == 'collect':
                a.round = command['round']
                optimizer.zero_grad(set_to_none=True)
                if getattr(a, 'rollout_vllm_sleep', False):
                    # Return update-time cached buffers before vLLM wakes on this GPU.
                    torch.cuda.empty_cache()
                cache = {}
                if frozen_head is not head:
                    frozen_head.load_state_dict(head.state_dict())
                started = time.monotonic()
                if a.rollout_backend == 'vllm' and rollout_client is None:
                    from lulu.vllm_rollout import RolloutClient
                    # DDP construction leaves freed bucket/broadcast buffers cached.
                    # vLLM counts the other process's reserved CUDA memory as occupied.
                    torch.cuda.empty_cache()
                    rollout_client = RolloutClient(a, ids[rank], rank)
                for records, payload in collect_batches(student, tok, rows, a, rank, world, rollout_client):
                    cache.update((r['index'], r) for r in records)
                    conn.send({'op': 'batch', 'rank': rank, 'round': a.round, 'records': payload})
                if rollout_client is not None and getattr(a, 'rollout_vllm_sleep', False):
                    rollout_client.sleep()
                if a.method in ('ren_resolved', 'ren_stable', 'ren_shared', 'ren_balanced'):
                    audit_rollouts(list(cache.values()), tok, a, rank)
                conn.send({'op': 'collected', 'rank': rank, 'round': a.round,
                           'seconds': time.monotonic()-started,
                           'response_tokens': sum(len(r['response_ids']) for r in cache.values())})
            elif op == 'validate':
                optimizer.zero_grad(set_to_none=True)
                if rollout_client is None:
                    from lulu.vllm_rollout import RolloutClient
                    torch.cuda.empty_cache()
                    rollout_client=RolloutClient(a,ids[rank],rank)
                from lulu.validation import validate_student
                conn.send(validate_student(student,tok,a,rank,world,rollout_client,command['round']))
            elif op == 'update':
                if command['round'] != a.round:
                    raise RuntimeError('Update command round differs from collected snapshot')
                targets = command['records']
                if set(targets) != set(cache):
                    raise RuntimeError('Missing or duplicate pipeline targets')
                for index, target in targets.items():
                    cache[index].update(target)
                metrics = update_records(step_model, distributed, optimizer, list(cache.values()),
                                         command['dummy'], a, rank, world)
                optimizer_updates += int(not metrics.get('skipped_update', False))
                metrics['completed_updates'] = optimizer_updates
                metrics['completed_rounds'] = a.round+1
                cache.clear()
                consolidate_optimizer(optimizer, a)
                if rank == 0:
                    started = time.monotonic()
                    path = manager.save(student, tok, optimizer, a.round+1,
                        {'completed_rounds': a.round+1, 'completed_updates': optimizer_updates, 'model': a.model, 'method': a.method,
                         'metrics': [metrics]}, final=a.round+1 == a.rounds)
                    metrics['checkpoint_seconds'] = time.monotonic()-started
                    metrics['checkpoint'] = str(path)
                conn.send({'op': 'updated', 'rank': rank, 'round': a.round, 'metrics': metrics})
            else:
                raise ValueError(f'Unknown Student command: {op}')
    except BaseException as exc:
        try:
            conn.send({'op': 'error', 'role': 'student', 'rank': rank,
                       'error': str(exc), 'traceback': traceback.format_exc()})
        except (OSError, EOFError):
            pass
        raise
    finally:
        if rollout_client is not None:
            rollout_client.close()
        if dist.is_initialized():
            dist.destroy_process_group()
        conn.close()


class MemoryChannel:
    """Byte IPC avoids /dev/shm tensor lifetime/capacity limits (4 GiB here)."""
    def __init__(self, connection, timeout=None):
        self.connection = connection
        self.timeout = timeout

    def _transfer(self, operation):
        if self.timeout is None:
            return operation()
        complete, result = threading.Event(), []
        def transfer():
            try:
                result.append((True, operation()))
            except BaseException as error:
                result.append((False, error))
            finally:
                complete.set()
        # A stalled peer can stop midway through a large tensor payload. Bound
        # both send and recv, not just the initial connection-readiness poll.
        thread = threading.Thread(target=transfer, daemon=True)
        thread.start()
        if not complete.wait(self.timeout):
            raise TimeoutError(f'Worker IPC transfer exceeded {self.timeout} seconds')
        succeeded, value = result[0]
        if not succeeded:
            raise value
        return value

    def send(self, message):
        return self._transfer(lambda: self.connection.send_bytes(pickle.dumps(message, protocol=5)))

    def recv(self):
        return self._transfer(lambda: pickle.loads(self.connection.recv_bytes()))

    def fileno(self):
        return self.connection.fileno()

    def close(self):
        self.connection.close()


class Workers:
    """Fail promptly on crashes, with a configurable inactivity timeout."""
    def __init__(self, timeout):
        self.ctx = mp.get_context('spawn')
        self.processes, self.connections = [], []
        self.timeout = timeout

    def launch(self, target, args, *, connected=True):
        parent, child = [MemoryChannel(c) for c in self.ctx.Pipe()] if connected else (None, None)
        if parent:
            parent.timeout = self.timeout
        proc = self.ctx.Process(target=target, args=(*args, child))
        proc.start()
        if child:
            child.close()
            self.connections.append(parent)
        self.processes.append(proc)
        return parent

    def receive(self, connections):
        deadline = time.monotonic()+self.timeout
        while True:
            ready = wait(connections, timeout=min(1, max(0, deadline-time.monotonic())))
            if ready:
                conn = ready[0]
                try:
                    message = conn.recv()
                except (EOFError, OSError) as exc:
                    raise RuntimeError('Persistent worker closed its connection unexpectedly') from exc
                if message.get('op') == 'error':
                    raise RuntimeError(f"{message.get('role')} worker failed: {message.get('error')}\n{message.get('traceback', '')}")
                return conn, message
            failed = [(p.pid, p.exitcode) for p in self.processes if p.exitcode is not None]
            if failed:
                raise RuntimeError(f'Persistent workers exited unexpectedly: {failed}')
            if time.monotonic() >= deadline:
                raise TimeoutError(f'No worker response for {self.timeout} seconds')

    def expect(self, conn, op):
        _, message = self.receive([conn])
        if message['op'] != op:
            raise RuntimeError(f'Expected {op}, received {message}')
        return message

    def close(self):
        # Termination also covers failed TP peers blocked in a collective.
        graceful = sys.exc_info()[0] is None
        if graceful:
            for conn in self.connections:
                conn.timeout = min(2, self.timeout)
                with contextlib.suppress(OSError, EOFError, TimeoutError):
                    conn.send({'op': 'stop'})
        else:
            for proc in self.processes:
                if proc.is_alive():
                    proc.terminate()
        deadline = time.monotonic()+10
        for proc in self.processes:
            proc.join(timeout=max(0, deadline-time.monotonic()))
        for proc in self.processes:
            if proc.is_alive():
                proc.terminate()
        for proc in self.processes:
            proc.join(timeout=3)
        for conn in self.connections:
            conn.close()


def teacher_payload(records):
    # Gold and privileged context must never cross this service boundary.
    return [{key: r[key] for key in ('causal_prompt_ids', 'response_ids', 'positions',
                                    'correction_ids') if key in r} for r in records]


def pipeline_round(a, workers, students, hindsight, teacher):
    started = time.monotonic()
    if hindsight:
        if not a.lora_rank and a.rollout_backend == 'vllm':
            from lulu.vllm_rollout import snapshot_path
            hindsight.send({'op': 'sync_checkpoint', 'round': a.round,
                            'checkpoint': snapshot_path(a.output_dir, a.round)})
        else:
            students[0].send({'op': 'snapshot', 'round': a.round})
            snapshot = workers.expect(students[0], 'snapshot')
            if snapshot.get('round') != a.round:
                raise RuntimeError('Student returned a stale snapshot round')
            hindsight.send({'op': 'sync', 'round': a.round, 'state': snapshot['state']})
            del snapshot
        synced = workers.expect(hindsight, 'synced')
        if synced.get('round') != a.round:
            raise RuntimeError('Privileged Student acknowledged a stale snapshot round')
    sync_seconds = time.monotonic()-started
    for conn in students:
        conn.send({'op': 'collect', 'round': a.round})
    pending_h, pending_t = deque(), deque()
    busy_h = busy_t = None
    collected = set()
    targets = {i: {} for i in range(len(students))}
    # Recognition-weighted ReN needs H Top-K IDs and the full answer-blind
    # Teacher hidden states independently. Fork each batch to both services;
    # only join their results at the update barrier. Sparse graft/projection
    # objectives still require H's selected IDs before Teacher scoring.
    parallel_services = a.method in ('ren_opd', 'ren_resolved', 'ren_stable', 'ren_shared', 'ren_balanced') and hindsight is not None and teacher is not None
    scored_by = {}
    dummy = None
    response_tokens = 0
    service_seconds = {'hindsight': 0., 'teacher': 0.}
    service_counts = {'hindsight': 0, 'teacher': 0}
    scoring_work = {'teacher_sequence_tokens': 0, 'hindsight_sequence_tokens': 0,
                    'teacher_positions': 0, 'hindsight_positions': 0,
                    'causal_prompt_tokens': 0, 'hindsight_prompt_tokens': 0}
    def progress(phase):
        if (Path(a.output_dir)/'run_config.json').exists():
            tr.atomic_json(Path(a.output_dir)/'phase_progress.json',
                {'round': a.round, 'phase': phase, 'collected_student_ranks':len(collected),
                 'student_ranks':len(students), 'scored_trajectories':dict(service_counts),
                 'elapsed_seconds':time.monotonic()-started, 'updated_unix':time.time()})
    progress('rollout_and_scoring')
    def complete(rank, records):
        nonlocal dummy
        for r in records:
            if r['index'] in targets[rank]:
                raise RuntimeError('Duplicate target from scoring service')
            targets[rank][r['index']] = {key: r[key] for key in
                ('correction_ids', 'recognition_ids', 'teacher_probs', 'teacher_hidden', 'hindsight_hidden', 'reference_hidden', 'student_hidden', 'teacher_scored') if key in r}
            if dummy is None:
                dummy = {key: r[key] for key in ('causal_prompt_ids', 'response_ids')}
                # A zero-loss DDP padding record needs a graph, not a long rollout.
                dummy['response_ids'] = dummy['response_ids'][:1]
                dummy.update(positions=[], snapshot_round=a.round, dummy=True)
    while len(collected) < len(students) or pending_h or pending_t or busy_h or busy_t:
        if hindsight and busy_h is None and pending_h:
            busy_h = pending_h.popleft()
            rank, records = busy_h
            hindsight.send({'op': 'score', 'round': a.round, 'request_id': records[0]['index'],
                'records': [{key: r[key] for key in ('hindsight_prompt_ids', 'causal_prompt_ids', 'response_ids',
                            'positions', 'causal_topk_ids') if key in r} for r in records]})
        if teacher and busy_t is None and pending_t:
            busy_t = pending_t.popleft()
            rank, records = busy_t
            teacher.send({'op': 'score', 'round': a.round, 'request_id': records[0]['index'],
                          'records': teacher_payload(records)})
        connections = [conn for i, conn in enumerate(students) if i not in collected]
        if busy_h:
            connections.append(hindsight)
        if busy_t:
            connections.append(teacher)
        if not connections:
            break
        conn, message = workers.receive(connections)
        if message.get('round') != a.round:
            raise RuntimeError('Service returned a stale round')
        if conn in students:
            rank = students.index(conn)
            if message['op'] == 'collected':
                collected.add(rank)
                response_tokens += message['response_tokens']
                progress('rollout_and_scoring')
            elif message['op'] == 'batch':
                records = message['records']
                for r in records:
                    response_n=len(r.get('response_ids',()))
                    causal_n=len(r.get('causal_prompt_ids',()))
                    hindsight_n=len(r.get('hindsight_prompt_ids',()))
                    positions_n=len(r.get('positions',()))
                    scoring_work['causal_prompt_tokens'] += causal_n
                    scoring_work['hindsight_prompt_tokens'] += hindsight_n
                    scoring_work['teacher_sequence_tokens'] += causal_n + response_n
                    scoring_work['hindsight_sequence_tokens'] += hindsight_n + response_n
                    scoring_work['teacher_positions'] += positions_n
                    scoring_work['hindsight_positions'] += positions_n
                if hindsight:
                    pending_h.append((rank, records))
                    if parallel_services:
                        for r in records:
                            key = (rank, r['index'])
                            if key in scored_by:
                                raise RuntimeError('Duplicate target from Student batch')
                            scored_by[key] = set()
                        pending_t.append((rank, records))
                else:
                    for r in records:
                        if a.method == 'causal_topk':
                            r['correction_ids'] = r['causal_topk_ids']
                        r.pop('hindsight_prompt_ids', None)
                    pending_t.append((rank, records))
            else:
                raise RuntimeError(f'Unexpected Student pipeline message: {message["op"]}')
        else:
            is_h = conn == hindsight
            rank, records = busy_h if is_h else busy_t
            if message['op'] != 'scored' or message['request_id'] != records[0]['index'] or len(message['records']) != len(records):
                raise RuntimeError('Mismatched scoring service response')
            for r, result in zip(records, message['records']):
                r.update(result)
                # Teacher may finish before a queued H request even starts.
                # Keep H's prompt until its own response has been received.
                if is_h or not parallel_services:
                    r.pop('hindsight_prompt_ids', None)
            service_seconds['hindsight' if is_h else 'teacher'] += message.get('seconds', 0.)
            service_counts['hindsight' if is_h else 'teacher'] += len(records)
            progress('rollout_and_scoring')
            if parallel_services:
                role = 'hindsight' if is_h else 'teacher'
                if is_h:
                    busy_h = None
                else:
                    busy_t = None
                for r in records:
                    finished = scored_by[(rank, r['index'])]
                    if role in finished:
                        raise RuntimeError('Duplicate target from scoring service')
                    finished.add(role)
                    if not is_h:
                        r['teacher_scored'] = True
                    if len(finished) == 2:
                        complete(rank, [r])
            elif is_h:
                busy_h = None
                if teacher:
                    pending_t.append((rank, records))
                else:
                    complete(rank, records)
            else:
                busy_t = None
                for r in records:
                    r['teacher_scored'] = True
                complete(rank, records)
    expected = a.global_batch_prompts*a.rollouts_per_prompt
    if sum(map(len, targets.values())) != expected or dummy is None:
        raise RuntimeError('Incomplete resident rollout/target pipeline')
    pipeline_seconds = time.monotonic()-started-sync_seconds
    progress('weight_scoring_and_update')
    for rank, conn in enumerate(students):
        conn.send({'op': 'update', 'round': a.round, 'records': targets[rank], 'dummy': dummy})
    updated = set()
    metrics = None
    while len(updated) < len(students):
        conn, message = workers.receive([c for i, c in enumerate(students) if i not in updated])
        rank = students.index(conn)
        if message['op'] != 'updated' or message['round'] != a.round:
            raise RuntimeError('Unexpected update completion')
        updated.add(rank)
        if rank == 0:
            metrics = message['metrics']
    progress('round_complete')
    metrics.update(round_seconds=time.monotonic()-started, pipeline_seconds=pipeline_seconds,
                   snapshot_sync_seconds=sync_seconds, response_tokens=response_tokens,
                   teacher_seconds=service_seconds['teacher'], hindsight_seconds=service_seconds['hindsight'],
                   **scoring_work)
    return metrics


def run_persistent(a):
    if a.update_passes != 1:
        raise ValueError('Persistent backend requires --update-passes 1 for strict on-policy rounds; staged supports snapshot reuse')
    if a.keep_round_cache:
        raise ValueError('Persistent backend uses RAM caches; --keep-round-cache requires --backend staged')
    roles = allocate_roles(a)
    plan = {'backend': 'persistent', 'model': a.model, 'teacher': a.teacher_model,
            'method': a.method, 'roles': roles,
            'teacher_tp_mode': getattr(a, 'teacher_tp_mode', 'native'),
            'teacher_parallelism': ('tensor_parallel' if roles['teacher'] and len(roles['teacher']) > 1
                                    else 'single_process' if roles['teacher'] is not None else None),
            'rollouts_per_round': a.global_batch_prompts*a.rollouts_per_prompt,
            'rounds': a.rounds, 'strict_on_policy': True, 'resident_models': True,
            'context_budget': {'max_new_tokens': a.max_new_tokens,
                               'max_prompt_tokens': a.max_prompt_tokens,
                               'max_sequence_tokens': a.max_sequence_tokens},
            'batches': {'rollout_per_student': a.rollout_batch_size,
                        'score': a.score_batch_size, 'train_micro': a.train_micro_batch_size},
            'pipeline': ('Student rollout/causal -> parallel hindsight and answer-blind Teacher -> joined targets -> DDP update'
                         if a.method in ('ren_opd', 'ren_resolved', 'ren_stable', 'ren_shared', 'ren_balanced') else
                         'Student rollout/causal -> synchronized hindsight -> answer-blind Teacher; then DDP update'),
            'rollout_backend': a.rollout_backend,
            'rollout_sampling': {'temperature': a.temperature, 'top_p': a.top_p, 'top_k': a.rollout_top_k},
            'optimizer_state_sharding': getattr(a, 'optimizer_state_sharding', False),
            'rollout_vllm_sleep': getattr(a, 'rollout_vllm_sleep', False),
            'cache': 'in-memory hidden states; diagnostic scores and rollout metadata persisted', 'save_every': a.save_every, 'latest_every': 1}
    print(json.dumps(plan, indent=2), flush=True)
    if a.dry_run:
        return
    root = Path(a.output_dir).resolve()
    a.output_dir = str(root)
    a.train_data = str(Path(a.train_data).resolve())
    manifest = root/'run_config.json'
    ignored = {'resume', 'phase', 'round', 'dry_run', 'worker_rank', 'worker_world'}
    config = {k: v for k, v in vars(a).items() if k not in ignored}
    config['train_sha256'] = hashlib.sha256(Path(a.train_data).read_bytes()).hexdigest()
    manager = None
    if manifest.exists():
        if not a.resume:
            raise FileExistsError('Run exists; use --resume')
        if resume_scientific_config(json.loads(manifest.read_text())) != resume_scientific_config(config):
            raise ValueError('Resume configuration/data differs from original run')
        manager = checkpoint_manager(a)
        initial = manager.latest_path()
        if initial is None:
            # Initial setup may have failed before a checkpoint was published.
            start_round = 0
        else:
            start_round = manager.latest_metadata()['completed_rounds']
    else:
        if a.resume:
            raise FileNotFoundError('Cannot resume without run_config.json')
        if root.exists() and any(root.iterdir()):
            raise FileExistsError('Output directory must be empty for a new run')
        tr.atomic_json(manifest, config)
        initial, start_round = None, 0
        manager = checkpoint_manager(a)
    if start_round >= a.rounds or (root/'early_stop.json').exists():
        print(f'Completed: {initial}', flush=True)
        return
    a.initial_checkpoint = str(initial) if initial else None
    workers = Workers(a.worker_timeout)
    startup = time.monotonic()
    try:
        from lulu.teacher_service import teacher_worker
        from lulu.hindsight_service import hindsight_worker
        teacher = None
        if roles['teacher'] is not None:
            ids = roles['teacher']
            world, port = max(1, len(ids)), _port()
            for rank in range(world):
                conn = workers.launch(teacher_worker, (a, rank, world, ids, '127.0.0.1', port), connected=rank == 0)
                if rank == 0:
                    teacher = conn
        ids = roles['student']
        world, port = max(1, len(ids)), _port()
        students = [workers.launch(student_worker, (a, rank, world, ids, port)) for rank in range(world)]
        first = workers.expect(students[0], 'ready')
        a.initial_checkpoint = first['initial_checkpoint']
        plan['student_parameters'] = first.get('parameter_counts')
        hindsight = workers.launch(hindsight_worker, (a, roles['hindsight'])) if roles['hindsight'] is not None else None
        for conn in students[1:]:
            workers.expect(conn, 'ready')
        if teacher:
            ready = workers.expect(teacher, 'ready')
            if a.method in ('vanilla_opd', 'ren_opd', 'ren_resolved', 'ren_stable', 'ren_shared', 'ren_balanced'):
                for conn in students:
                    conn.send({'op': 'teacher_head', 'state': ready['teacher_head']})
                for conn in students:
                    workers.expect(conn, 'head_ready')
        if hindsight:
            ready = workers.expect(hindsight, 'ready')
            if a.method in ('ren_stable', 'ren_shared', 'ren_balanced'):
                for conn in students:
                    conn.send({'op': 'reference_head', 'state': ready['reference_head']})
                for conn in students:
                    workers.expect(conn, 'head_ready')
                plan['reference_checkpoint'] = ready['reference_checkpoint']
                plan['reference_kl_coef'] = a.reference_kl_coef
                del ready
        tr.atomic_json(root/'runtime_plan.json', dict(plan, startup_seconds=time.monotonic()-startup))
        if getattr(a,'validation_every',0):
            from lulu.validation import validation_round
            if start_round==0 or start_round%a.validation_every==0:
                validation_round(a,workers,students,start_round)
        for index in range(start_round, a.rounds):
            a.round = index
            metrics = pipeline_round(a, workers, students, hindsight, teacher)
            remaining = a.rounds-index-1
            metrics['remaining_seconds_at_last_round_rate'] = remaining*metrics['round_seconds']
            if a.method in ('ren_stable', 'ren_shared', 'ren_balanced'):
                from lulu.stability import record_round
                metrics['stability'] = record_round(root, index, metrics, a)
            tr.atomic_json(root/'metrics'/f'round_{index:04d}.json', [metrics])
            if a.method in ('ren_stable', 'ren_shared', 'ren_balanced'):
                state = manager.latest_metadata()
                tr.atomic_json(root/'latest.json', dict(state, metrics=[metrics]))
            if getattr(a,'validation_every',0) and ((index+1)%a.validation_every==0 or index+1==a.rounds or metrics.get('stability',{}).get('early_stop')):
                validation_round(a,workers,students,index+1)
            print(json.dumps(metrics), flush=True)
            if metrics.get('stability', {}).get('early_stop'):
                tr.atomic_json(root/'early_stop.json', {'completed_rounds': index+1,
                    'completed_updates': metrics['completed_updates'], 'reason': metrics['stability']['stop_reason'],
                    'checkpoint': str(manager.latest_path())})
                break
        print(f'Completed: {manager.latest_path()}', flush=True)
    finally:
        workers.close()
