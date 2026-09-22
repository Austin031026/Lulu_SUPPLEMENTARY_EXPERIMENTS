#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pyarrow.parquet as pq
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class PromptExample:
    prompt_index: int
    prompt_ids: list[int]
    data_source: str
    ground_truth: str


@dataclass
class Rollout:
    prompt_ids: list[int]
    response_ids: list[int]
    old_logprobs: list[float]
    support_ids: list[list[int]] | None
    reward: float
    raw_reward: float
    advantage: float = 0.0


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Lulu GRPO baseline. Multi-GPU phases collect "
            "on-policy rollouts without NCCL collectives; a single GPU then "
            "performs one globally normalized LoRA policy update."
        )
    )
    p.add_argument("--phase", choices=["plan", "init", "collect", "update"], required=True)
    p.add_argument("--algorithm", choices=["grpo", "dapo", "rlpt"], required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--train-data", required=True)
    p.add_argument("--parser-path", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--experiment-name", required=True)
    p.add_argument("--step", type=int, default=0)
    p.add_argument("--rollout-dir", default="")

    p.add_argument("--global-batch-prompts", type=int, default=48)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--global-epochs", type=float, default=1.0)
    p.add_argument("--max-response-tokens", type=int, default=8192)
    p.add_argument("--max-prompt-tokens", type=int, default=4096)

    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--rlpt-k", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--clip-ratio", type=float, default=0.2)
    p.add_argument("--dapo-clip-high", type=float, default=0.28)
    p.add_argument("--adv-eps", type=float, default=1e-6)
    p.add_argument("--ppo-epochs", type=int, default=1)

    p.add_argument("--dapo-overlong-buffer", type=int, default=1024)
    p.add_argument("--dapo-overlong-penalty", type=float, default=1.0)

    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-every", type=int, default=4)
    p.add_argument("--log-every", type=int, default=1)
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    return p.parse_args()


def worker_env():
    """Read torchrun rank assignment without creating a process group.

    Rollout collection is embarrassingly parallel.  Avoiding init_process_group
    removes the NCCL straggler failure mode entirely.
    """
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local)
    return rank, world, local, torch.device("cuda", local)


def single_device():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(0)
    return torch.device("cuda", 0)


def load_parser(path):
    spec = importlib.util.spec_from_file_location("rl_baseline_parser", str(Path(path).resolve()))
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def extract_answer(parser, text, data_source):
    try:
        return parser.extract_answer(text, data_source)
    except TypeError:
        return parser.extract_answer(text)


def load_prompts(path: str) -> list[PromptExample]:
    rows = pq.read_table(path).to_pylist()
    by_prompt: dict[int, PromptExample] = {}
    for row in rows:
        pidx = int(row["prompt_index"])
        plen = int(row["prompt_length"])
        prefix = [int(x) for x in row["critical_prefix_ids"]]
        if plen <= 0 or plen > len(prefix):
            raise ValueError(f"prompt={pidx}: invalid prompt_length={plen}")
        ex = PromptExample(
            prompt_index=pidx,
            prompt_ids=prefix[:plen],
            data_source=str(row.get("data_source", "math")),
            ground_truth=str(row.get("ground_truth", "")),
        )
        if not ex.ground_truth:
            raise ValueError(f"prompt={pidx}: empty ground_truth")
        old = by_prompt.get(pidx)
        if old is not None and old != ex:
            raise ValueError(f"prompt={pidx}: inconsistent duplicated prompt")
        by_prompt[pidx] = ex
    out = [by_prompt[k] for k in sorted(by_prompt)]
    if not out:
        raise RuntimeError("no prompts")
    return out


def epoch_order(n: int, seed: int, epoch: int) -> list[int]:
    order = list(range(n))
    random.Random(seed + 1000003 * epoch).shuffle(order)
    return order


def prompt_schedule(n: int, exposures: int, seed: int) -> list[int]:
    out = []
    epoch = 0
    while len(out) < exposures:
        out.extend(epoch_order(n, seed, epoch))
        epoch += 1
    return out[:exposures]


def plan(args, prompts: list[PromptExample]):
    n = len(prompts)
    requested = int(math.ceil(n * float(args.global_epochs)))
    steps = int(math.ceil(requested / int(args.global_batch_prompts)))
    scheduled = steps * int(args.global_batch_prompts)
    return {
        "unique_prompts": n,
        "requested_prompt_exposures": requested,
        "scheduled_prompt_exposures": scheduled,
        "steps": steps,
        "scheduled_prompt_rollouts": scheduled * int(args.group_size),
    }


def global_ids_for_step(args, n: int, step: int) -> list[int]:
    info = plan(args, [None] * n)  # only len() is used
    if not (1 <= step <= int(info["steps"])):
        raise ValueError(f"step={step} outside 1..{info['steps']}")
    schedule = prompt_schedule(n, int(info["scheduled_prompt_exposures"]), int(args.seed))
    start = (step - 1) * int(args.global_batch_prompts)
    return schedule[start : start + int(args.global_batch_prompts)]


def dtype_from_args(args):
    return torch.bfloat16 if args.dtype == "bfloat16" else torch.float16


def load_tokenizer(args):
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=False)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def make_base(args):
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype_from_args(args),
        trust_remote_code=False,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    return base


def lora_config(args):
    return LoraConfig(
        r=int(args.lora_rank),
        lora_alpha=int(args.lora_alpha),
        lora_dropout=float(args.lora_dropout),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )


def load_policy(args, device, *, adapter_path: Path | None, trainable: bool):
    base = make_base(args)
    if adapter_path is None:
        model = get_peft_model(base, lora_config(args))
    else:
        if not (adapter_path / "adapter_config.json").is_file():
            raise FileNotFoundError(adapter_path / "adapter_config.json")
        model = PeftModel.from_pretrained(
            base,
            str(adapter_path),
            is_trainable=bool(trainable),
        )
    model = model.to(device)
    if trainable:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        model.config.use_cache = False
        model.train()
    else:
        model.config.use_cache = True
        model.eval()
    return model


@torch.no_grad()
def rollout_group(
    model,
    tokenizer,
    ex: PromptExample,
    args,
    device,
    *,
    worker_rank: int,
    rollout_step: int,
) -> list[Rollout]:
    """Generate G responses from one prompt and cache behavior log-probs."""
    model.eval()
    model.config.use_cache = True

    g = int(args.group_size)
    prompt = torch.tensor([ex.prompt_ids] * g, dtype=torch.long, device=device)
    if prompt.shape[1] > int(args.max_prompt_tokens):
        raise ValueError(
            f"prompt={ex.prompt_index}: length={prompt.shape[1]} > "
            f"max_prompt_tokens={args.max_prompt_tokens}"
        )
    attn = torch.ones_like(prompt)
    out = model(input_ids=prompt, attention_mask=attn, use_cache=True, return_dict=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :].float()

    eos = tokenizer.eos_token_id
    if isinstance(eos, (list, tuple)):
        eos_ids = {int(x) for x in eos}
        eos_for_pad = int(eos[0])
    elif eos is None:
        eos_ids = set()
        eos_for_pad = int(tokenizer.pad_token_id)
    else:
        eos_ids = {int(eos)}
        eos_for_pad = int(eos)
    pad_id = int(tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_for_pad)

    responses = [[] for _ in range(g)]
    old_lps = [[] for _ in range(g)]
    supports = ([[] for _ in range(g)] if args.algorithm == "rlpt" else None)
    finished = torch.zeros(g, dtype=torch.bool, device=device)

    gen = torch.Generator(device=device)
    gen.manual_seed( int(args.seed) + 7919 * int(worker_rank) + 104729 * int(ex.prompt_index) + 1000003 * int(rollout_step) )

    for _step in range(int(args.max_response_tokens)):
        step_logits = logits / float(args.temperature)

        if args.algorithm == "rlpt":
            k = min(int(args.rlpt_k), int(step_logits.shape[-1]))
            top_vals, top_ids = torch.topk(step_logits, k=k, dim=-1)
            probs = torch.softmax(top_vals, dim=-1)
            rel = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1)
            tok = top_ids.gather(1, rel[:, None]).squeeze(1)
            lp = torch.log_softmax(top_vals, dim=-1).gather(1, rel[:, None]).squeeze(1)
        else:
            probs = torch.softmax(step_logits, dim=-1)
            tok = torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1)
            lp = torch.log_softmax(step_logits, dim=-1).gather(1, tok[:, None]).squeeze(1)
            top_ids = None

        tok = torch.where(finished, torch.full_like(tok, pad_id), tok)

        for i in range(g):
            if bool(finished[i]):
                continue
            tid = int(tok[i].item())
            responses[i].append(tid)
            old_lps[i].append(float(lp[i].item()))
            if supports is not None:
                supports[i].append([int(x) for x in top_ids[i].tolist()])
            if tid in eos_ids:
                finished[i] = True

        if bool(finished.all()):
            break

        step_ids = tok[:, None]
        attn = torch.cat([attn, (~finished).to(dtype=attn.dtype)[:, None]], dim=1)
        out = model(
            input_ids=step_ids,
            attention_mask=attn,
            past_key_values=past,
            use_cache=True,
            return_dict=True,
        )
        past = out.past_key_values
        logits = out.logits[:, -1, :].float()

    parser = args._parser
    rollouts = []
    for i in range(g):
        text = tokenizer.decode(responses[i], skip_special_tokens=True)
        pred = extract_answer(parser, text, ex.data_source)
        raw = float(pred is not None and bool(parser.math_equal(pred, ex.ground_truth)))
        reward = raw
        if args.algorithm == "dapo" and args.dapo_overlong_buffer > 0:
            start = max(0, int(args.max_response_tokens) - int(args.dapo_overlong_buffer))
            over = max(0, len(responses[i]) - start)
            penalty = float(args.dapo_overlong_penalty) * min(
                1.0, over / float(args.dapo_overlong_buffer)
            )
            reward -= penalty
        rollouts.append(
            Rollout(
                prompt_ids=ex.prompt_ids,
                response_ids=responses[i],
                old_logprobs=old_lps[i],
                support_ids=(supports[i] if supports is not None else None),
                reward=float(reward),
                raw_reward=raw,
            )
        )
    return rollouts


def assign_advantages(group: list[Rollout], eps: float):
    rewards = np.asarray([x.reward for x in group], dtype=np.float64)
    mean = float(rewards.mean())
    std = float(rewards.std(ddof=0))
    if std <= eps:
        for x in group:
            x.advantage = 0.0
        return False
    vals = (rewards - mean) / (std + eps)
    for x, a in zip(group, vals):
        x.advantage = float(a)
    return True


def rollout_to_payload(r: Rollout):
    support = None
    if r.support_ids is not None:
        support = torch.tensor(r.support_ids, dtype=torch.int32)
    return {
        "response_ids": torch.tensor(r.response_ids, dtype=torch.int32),
        "old_logprobs": torch.tensor(r.old_logprobs, dtype=torch.float32),
        "support_ids": support,
        "reward": float(r.reward),
        "raw_reward": float(r.raw_reward),
        "advantage": float(r.advantage),
    }


def payload_to_rollout(prompt_ids: list[int], x: dict[str, Any]):
    support = x.get("support_ids")
    return Rollout(
        prompt_ids=list(map(int, prompt_ids)),
        response_ids=[int(v) for v in x["response_ids"].tolist()],
        old_logprobs=[float(v) for v in x["old_logprobs"].tolist()],
        support_ids=(
            [[int(y) for y in row] for row in support.tolist()]
            if support is not None else None
        ),
        reward=float(x["reward"]),
        raw_reward=float(x["raw_reward"]),
        advantage=float(x["advantage"]),
    )


def write_rollout_shard(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def validate_existing_rollout_shard(path: Path, *, step: int, rank: int, world: int, indices: list[int]):
    if not path.is_file():
        return False
    try:
        x = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return False
    return (
        int(x.get("schema_version", -1)) == 2
        and int(x.get("step", -1)) == int(step)
        and int(x.get("rank", -1)) == int(rank)
        and int(x.get("world_size", -1)) == int(world)
        and [int(v) for v in x.get("schedule_indices", [])] == [int(v) for v in indices]
    )


def collect_phase(args, prompts: list[PromptExample]):
    if args.step <= 0:
        raise ValueError("collect requires --step >= 1")
    if not args.rollout_dir:
        raise ValueError("collect requires --rollout-dir")

    rank, world, local, device = worker_env()
    all_ids = global_ids_for_step(args, len(prompts), int(args.step))
    local_ids = all_ids[rank::world]
    out = Path(args.rollout_dir).expanduser().resolve() / f"rank-{rank:03d}.pt"

    if validate_existing_rollout_shard(
        out, step=args.step, rank=rank, world=world, indices=local_ids
    ):
        print(f"[rollout][SKIP] step={args.step} rank={rank}/{world} output={out}", flush=True)
        return

    prev_adapter = Path(args.output_dir).expanduser().resolve() / f"global_step_{args.step - 1}" / "adapter"
    tokenizer = load_tokenizer(args)
    model = load_policy(args, device, adapter_path=prev_adapter, trainable=False)
    args._parser = load_parser(args.parser_path)

    groups = []
    total_responses = 0
    total_tokens = 0
    active_groups = 0

    for j, idx in enumerate(local_ids, 1):
        ex = prompts[idx]
        group = rollout_group(
            model, tokenizer, ex, args, device,
            worker_rank=rank, rollout_step=int(args.step),
        )
        active = assign_advantages(group, float(args.adv_eps))
        active_groups += int(active)
        total_responses += len(group)
        total_tokens += sum(len(r.response_ids) for r in group)
        groups.append({
            "schedule_index": int(idx),
            "prompt_index": int(ex.prompt_index),
            "prompt_ids": torch.tensor(ex.prompt_ids, dtype=torch.int32),
            "rollouts": [rollout_to_payload(r) for r in group],
            "active": bool(active),
        })
        print(
            f"[rollout] step={args.step} rank={rank}/{world} "
            f"group={j}/{len(local_ids)} prompt={ex.prompt_index} "
            f"active={int(active)}",
            flush=True,
        )

    payload = {
        "schema_version": 2,
        "algorithm": args.algorithm,
        "step": int(args.step),
        "rank": rank,
        "world_size": world,
        "schedule_indices": [int(x) for x in local_ids],
        "groups": groups,
        "metrics": {
            "groups": len(groups),
            "active_groups": active_groups,
            "responses": total_responses,
            "response_tokens": total_tokens,
        },
    }
    write_rollout_shard(out, payload)
    print(
        f"[rollout][DONE] step={args.step} rank={rank}/{world} "
        f"groups={len(groups)} responses={total_responses} tokens={total_tokens} output={out}",
        flush=True,
    )


def token_logprobs(model, rollout: Rollout, algorithm: str, device):
    """Teacher-force one response and return current-policy log p(a_t|s_t)."""
    prompt = rollout.prompt_ids
    response = rollout.response_ids
    if not response:
        return torch.empty(0, dtype=torch.float32, device=device)

    seq = prompt + response
    ids = torch.tensor([seq], dtype=torch.long, device=device)
    attn = torch.ones_like(ids)
    out = model(input_ids=ids, attention_mask=attn, use_cache=False, return_dict=True)
    logits = out.logits[0].float()

    start = len(prompt) - 1
    pred_logits = logits[start : start + len(response), :]
    action_ids = torch.tensor(response, dtype=torch.long, device=device)

    if algorithm != "rlpt":
        return torch.log_softmax(pred_logits, dim=-1).gather(1, action_ids[:, None]).squeeze(1)

    support = rollout.support_ids
    if support is None or len(support) != len(response):
        raise RuntimeError("RLPT rollout is missing stored support")
    support_ids = torch.tensor(support, dtype=torch.long, device=device)
    support_logits = pred_logits.gather(1, support_ids)
    chosen = pred_logits.gather(1, action_ids[:, None]).squeeze(1)
    denom = torch.logsumexp(support_logits, dim=-1)

    if not bool((support_ids == action_ids[:, None]).any(dim=1).all()):
        raise RuntimeError("sampled RLPT token absent from stored support")
    return chosen - denom


def sequence_pg_loss(curr_lp, old_lp, advantage, low, high):
    ratio = torch.exp(curr_lp - old_lp)
    adv = torch.full_like(ratio, float(advantage))
    loss1 = -adv * ratio
    loss2 = -adv * torch.clamp(ratio, 1.0 - low, 1.0 + high)
    return torch.maximum(loss1, loss2), ratio


def iter_rollout_groups(shards: Iterable[Path]):
    for shard in shards:
        x = torch.load(shard, map_location="cpu", weights_only=False)
        for g in x["groups"]:
            prompt_ids = [int(v) for v in g["prompt_ids"].tolist()]
            yield {
                "schedule_index": int(g["schedule_index"]),
                "prompt_index": int(g["prompt_index"]),
                "active": bool(g["active"]),
                "rollouts": [payload_to_rollout(prompt_ids, r) for r in g["rollouts"]],
            }


def rollout_shards_and_stats(args, prompts: list[PromptExample]):
    if not args.rollout_dir:
        raise ValueError("update requires --rollout-dir")
    root = Path(args.rollout_dir).expanduser().resolve()
    shards = sorted(root.glob("rank-*.pt"))
    if not shards:
        raise RuntimeError(f"no rollout shards under {root}")

    expected_ids = global_ids_for_step(args, len(prompts), int(args.step))
    actual_ids = []
    total_groups = 0
    total_tokens = 0
    active_groups = 0
    raw_reward_sum = 0.0
    raw_reward_n = 0
    response_len_sum = 0
    response_len_n = 0

    for g in iter_rollout_groups(shards):
        actual_ids.append(int(g["schedule_index"]))
        total_groups += 1
        active_groups += int(g["active"])
        for r in g["rollouts"]:
            total_tokens += len(r.response_ids)
            raw_reward_sum += float(r.raw_reward)
            raw_reward_n += 1
            response_len_sum += len(r.response_ids)
            response_len_n += 1

    if Counter(actual_ids) != Counter(expected_ids):
        raise RuntimeError(
            "rollout schedule mismatch: "
            f"expected={Counter(expected_ids)} actual={Counter(actual_ids)}"
        )
    if total_groups != int(args.global_batch_prompts):
        raise RuntimeError(
            f"rollout group count mismatch: {total_groups}/{args.global_batch_prompts}"
        )

    return shards, {
        "groups": total_groups,
        "active_groups": active_groups,
        "tokens": total_tokens,
        "raw_reward_sum": raw_reward_sum,
        "raw_reward_n": raw_reward_n,
        "response_len_sum": response_len_sum,
        "response_len_n": response_len_n,
    }


def backward_group(model, group, args, device, *, global_groups: int, global_tokens: int):
    algorithm = args.algorithm
    low = float(args.clip_ratio)
    high = float(args.dapo_clip_high if algorithm == "dapo" else args.clip_ratio)

    valid = [r for r in group if r.response_ids]
    if not valid:
        return {"loss": 0.0, "tokens": 0, "clipfrac": 0.0}
    
    loss_value = 0.0
    clipped = 0
    ratio_count = 0

    for r in valid:
        curr = token_logprobs(model, r, algorithm, device)
        old = torch.tensor(r.old_logprobs, dtype=torch.float32, device=device)
        pg, ratio = sequence_pg_loss(curr, old, r.advantage, low, high)

        if algorithm == "dapo":
            # True token-level aggregation over the entire global rollout batch.
            loss = pg.sum() / max(1, int(global_tokens))
        else:
            # Equal weight per prompt-group, mean over responses and tokens.
            loss = pg.mean() / max(1, len(valid)) / max(1, int(global_groups))

        loss.backward()
        loss_value += float(loss.detach().item())
        clipped += int(((ratio < 1.0 - low) | (ratio > 1.0 + high)).sum().item())
        ratio_count += int(ratio.numel())

    return {
        "loss": loss_value,
        "tokens": sum(len(r.response_ids) for r in valid),
        "clipfrac": clipped / max(1, ratio_count),
    }

def checkpoint_dir(args, step: int):
    return Path(args.output_dir).expanduser().resolve() / f"global_step_{int(step)}"


def write_config(path: Path, args, metadata):
    path.write_text(json.dumps({
        "schema_version": 2,
        "architecture": "multi_gpu_rollout_single_gpu_update",
        "algorithm": args.algorithm,
        "model": args.model,
        "global_batch_prompts": args.global_batch_prompts,
        "group_size": args.group_size,
        "global_epochs": args.global_epochs,
        "max_response_tokens": args.max_response_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "rlpt_k": args.rlpt_k if args.algorithm == "rlpt" else None,
        "lr": args.lr,
        "clip_ratio_low": args.clip_ratio,
        "clip_ratio_high": args.dapo_clip_high if args.algorithm == "dapo" else args.clip_ratio,
        "ppo_epochs": args.ppo_epochs,
        "loss_aggregation": (
            "global-token-mean" if args.algorithm == "dapo" else "global-group-mean"
        ),
        "dapo_dynamic_sampling": (
            "zero-advantage-homogeneous-groups-within-fixed-rollout-budget"
            if args.algorithm == "dapo" else None
        ),
        **metadata,
    }, indent=2) + "\n")


def save_policy_and_optimizer(model, tokenizer, optimizer, args, step: int, metadata, metrics):
    parent = checkpoint_dir(args, step)
    adapter = parent / "adapter"
    adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    torch.save(optimizer.state_dict(), parent / "optimizer.pt")
    write_config(parent / "rl_baseline_config.json", args, metadata)
    (parent / "update_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")


def init_phase(args, prompts: list[PromptExample]):
    device = single_device()
    parent = checkpoint_dir(args, 0)
    adapter = parent / "adapter"
    if (adapter / "adapter_config.json").is_file():
        print(f"[policy-rl][INIT-SKIP] {adapter}", flush=True)
        return
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32 - 1))
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = load_tokenizer(args)
    model = load_policy(args, device, adapter_path=None, trainable=True)
    adapter.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter)
    tokenizer.save_pretrained(adapter)
    info = plan(args, prompts)
    write_config(parent / "rl_baseline_config.json", args, {
        **info,
        "step": 0,
        "note": "Fresh LoRA initialization used as the behavior policy for rollout step 1.",
    })
    print(f"[policy-rl][INIT-DONE] {adapter}", flush=True)


def update_phase(args, prompts: list[PromptExample]):
    if args.step <= 0:
        raise ValueError("update requires --step >= 1")
    device = single_device()
    out_parent = checkpoint_dir(args, args.step)
    if (
        (out_parent / "adapter" / "adapter_config.json").is_file()
        and (out_parent / "optimizer.pt").is_file()
        and (out_parent / "update_metrics.json").is_file()
    ):
        print(f"[policy-rl][UPDATE-SKIP] step={args.step} {out_parent}", flush=True)
        return

    shards, stats = rollout_shards_and_stats(args, prompts)
    prev_parent = checkpoint_dir(args, args.step - 1)
    prev_adapter = prev_parent / "adapter"

    random.seed(args.seed + 1009 * args.step)
    np.random.seed((args.seed + 1009 * args.step) % (2**32 - 1))
    torch.manual_seed(args.seed + 1009 * args.step)
    torch.cuda.manual_seed_all(args.seed + 1009 * args.step)

    tokenizer = load_tokenizer(args)
    model = load_policy(args, device, adapter_path=prev_adapter, trainable=True)
    params = [p for p in model.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        params, lr=float(args.lr), weight_decay=float(args.weight_decay)
    )

    prev_opt = prev_parent / "optimizer.pt"
    if prev_opt.is_file():
        optimizer.load_state_dict(torch.load(prev_opt, map_location="cpu", weights_only=False))

    epoch_metrics = []
    for ppo_epoch in range(int(args.ppo_epochs)):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        clipfracs = []
        for g in iter_rollout_groups(shards):
            m = backward_group(
                model,
                g["rollouts"],
                args,
                device,
                global_groups=int(stats["groups"]),
                global_tokens=int(stats["tokens"]),
            )
            losses.append(float(m["loss"]))
            clipfracs.append(float(m["clipfrac"]))

        grad = torch.nn.utils.clip_grad_norm_(params, float(args.clip_grad))
        optimizer.step()
        epoch_metrics.append({
            "ppo_epoch": ppo_epoch + 1,
            "loss": float(sum(losses)),
            "clipfrac_group_mean": float(sum(clipfracs) / max(1, len(clipfracs))),
            "grad_norm": float(grad),
        })

    reward_mean = stats["raw_reward_sum"] / max(1, stats["raw_reward_n"])
    active_frac = stats["active_groups"] / max(1, stats["groups"])
    mean_len = stats["response_len_sum"] / max(1, stats["response_len_n"])
    metrics = {
        "step": int(args.step),
        "algorithm": args.algorithm,
        "rollout_groups": int(stats["groups"]),
        "rollout_tokens": int(stats["tokens"]),
        "reward_mean": float(reward_mean),
        "active_group_fraction": float(active_frac),
        "mean_response_tokens": float(mean_len),
        "optimizer_updates_this_step": int(args.ppo_epochs),
        "epoch_metrics": epoch_metrics,
    }

    metadata = {
        **plan(args, prompts),
        "step": int(args.step),
        "optimizer_semantics": (
            "one globally normalized single-GPU optimizer update per PPO epoch "
            "over the complete global rollout batch"
        ),
        "rollout_world_size": len(shards),
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
    }
    save_policy_and_optimizer(model, tokenizer, optimizer, args, args.step, metadata, metrics)
    print(
        f"[policy-rl] step={args.step} alg={args.algorithm} "
        f"reward={reward_mean:.4f} active_groups={active_frac:.4f} "
        f"mean_response_tokens={mean_len:.1f} "
        f"loss={epoch_metrics[-1]['loss']:.6f} "
        f"clipfrac={epoch_metrics[-1]['clipfrac_group_mean']:.4f}",
        flush=True,
    )
    print(f"[policy-rl][UPDATE-DONE] step={args.step} {out_parent}", flush=True)


def main():
    args = parse_args()
    if args.global_batch_prompts <= 0:
        raise ValueError("global-batch-prompts must be positive")
    if args.group_size < 2:
        raise ValueError("group-size must be >=2")
    if args.algorithm == "rlpt" and args.rlpt_k < 2:
        raise ValueError("rlpt-k must be >=2")
    if args.temperature <= 0:
        raise ValueError("temperature must be >0")
    if abs(args.top_p - 1.0) > 1e-12:
        raise ValueError(
            "This compute-matched implementation intentionally fixes top_p=1.0; "
            "RLPT support is controlled by stored Top-K."
        )

    prompts = load_prompts(args.train_data)
    if args.phase == "plan":
        print(json.dumps({
            "algorithm": args.algorithm,
            **plan(args, prompts),
            "group_size": args.group_size,
            "rlpt_k": args.rlpt_k if args.algorithm == "rlpt" else None,
        }, indent=2))
        return
    if args.phase == "init":
        init_phase(args, prompts)
        return
    if args.phase == "collect":
        collect_phase(args, prompts)
        return
    if args.phase == "update":
        update_phase(args, prompts)
        return
    raise AssertionError(args.phase)

if __name__ == "__main__":
    main()
