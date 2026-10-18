#!/usr/bin/env python3
"""Generate and freeze ArcQ Experiment 1 diagnostic trajectories once."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lulu import training as tr
from lulu.exp1_predictiveness import (
    SCHEMA_VERSION, atomic_json, deterministic_subset, load_jsonl, sha256_file,
    trajectory_identity, validate_trajectory, write_jsonl,
)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-questions", type=int, default=1024,
                        help="Number of frozen questions (for example 256, 512, or 1024)")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-sequence-tokens", type=int, default=16384)
    parser.add_argument("--dtype", default="bfloat16", choices=("bfloat16", "float16"))
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--worker-rank", type=int)
    return parser.parse_args(argv)


def model_hashes(model: Path) -> dict[str, str]:
    result = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "model.safetensors.index.json"):
        path = model / name
        if path.is_file():
            result[name] = sha256_file(path)
    if "config.json" not in result or "tokenizer_config.json" not in result:
        raise ValueError(f"Incomplete local model snapshot: {model}")
    return result


def build_plan(args, output: Path) -> dict:
    source = Path(args.questions).resolve(strict=True)
    model = Path(args.model).resolve(strict=True)
    rows = load_jsonl(source)
    selected = deterministic_subset(rows, args.num_questions, args.seed)
    selected_path = output / "selected_questions.jsonl"
    write_jsonl(selected_path, selected)
    devices = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not devices or len(devices) != len(set(devices)):
        raise ValueError("--gpus must contain distinct GPU identifiers")
    return {
        "schema_version": SCHEMA_VERSION,
        "stage": "freeze_base_trajectories",
        "questions": str(source), "questions_sha256": sha256_file(source),
        "selected_questions": str(selected_path),
        "selected_questions_sha256": sha256_file(selected_path),
        "num_questions": args.num_questions, "selection_seed": args.seed,
        "model": str(model), "model_hashes": model_hashes(model), "gpus": devices,
        "generation": {"temperature": args.temperature, "top_p": args.top_p,
                       "top_k": args.top_k, "max_new_tokens": args.max_new_tokens,
                       "max_prompt_tokens": args.max_prompt_tokens,
                       "max_sequence_tokens": args.max_sequence_tokens,
                       "dtype": args.dtype, "per_question_seed": "seed + stable selected ordinal"},
    }


def run_worker(args, plan: dict) -> None:
    rank = int(args.worker_rank)
    devices = plan["gpus"]
    if not 0 <= rank < len(devices):
        raise ValueError("Invalid worker rank")
    os.environ.update(CUDA_VISIBLE_DEVICES=devices[rank], TOKENIZERS_PARALLELISM="false",
                      VLLM_WORKER_MULTIPROC_METHOD="spawn")
    output = Path(args.output_dir).resolve()
    for variable, name in (("VLLM_CACHE_ROOT", "vllm"),
                           ("TRITON_CACHE_DIR", "triton"),
                           ("TORCHINDUCTOR_CACHE_DIR", "inductor")):
        path = output / ".engine_cache" / f"worker-{rank:03d}" / name
        path.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(path)
    from vllm import LLM, SamplingParams

    tokenizer = tr.load_tokenizer(plan["model"])
    selected = load_jsonl(plan["selected_questions"])
    jobs = [(index, row) for index, row in enumerate(selected) if index % len(devices) == rank]
    records, prompts, parameters = [], [], []
    generation = plan["generation"]
    for index, row in jobs:
        views = tr.build_prompt_views(tokenizer, row["messages"], row["gold_answer"],
                                      enable_thinking=True, model_family="qwen")
        if max(len(views["causal_prompt_ids"]), len(views["hindsight_prompt_ids"])) > generation["max_prompt_tokens"]:
            raise ValueError(f"Question {row['id']} exceeds the prompt budget")
        records.append((index, row, views))
        prompts.append({"prompt_token_ids": views["causal_prompt_ids"]})
        parameters.append(SamplingParams(
            temperature=generation["temperature"], top_p=generation["top_p"],
            top_k=generation["top_k"], max_tokens=generation["max_new_tokens"],
            seed=args.seed + index, repetition_penalty=1.0,
        ))
    engine = LLM(
        model=plan["model"], dtype=generation["dtype"], tensor_parallel_size=1,
        max_model_len=generation["max_sequence_tokens"],
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs, max_num_batched_tokens=4096,
        enable_chunked_prefill=True, generation_config="vllm",
        compilation_config={"cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32]},
        seed=args.seed + rank,
    )
    started = time.monotonic()
    generated = engine.generate(prompts, parameters, use_tqdm=True)
    rows = []
    for (index, source, views), result in zip(records, generated):
        response_ids = list(result.outputs[0].token_ids)
        if not response_ids:
            raise RuntimeError(f"Empty response for question {source['id']}")
        mask = tr.reasoning_token_mask(tokenizer, response_ids,
                                       prompt_ids=views["causal_prompt_ids"])
        positions = [position for position, keep in enumerate(mask) if keep]
        row = {
            "selected_index": index, "question_id": str(source["id"]),
            "gold_answer": source["gold_answer"],
            "causal_prompt_ids": views["causal_prompt_ids"],
            "hindsight_prompt_ids": views["hindsight_prompt_ids"],
            "response_ids": response_ids, "positions": positions,
            "response": tokenizer.decode(response_ids, skip_special_tokens=True),
            "finish_reason": result.outputs[0].finish_reason,
            "truncated": result.outputs[0].finish_reason == "length",
            "generation_seed": args.seed + index,
        }
        row["trajectory_id"] = trajectory_identity(row["question_id"],
                                                    row["causal_prompt_ids"], response_ids)
        validate_trajectory(row, max_sequence_tokens=generation["max_sequence_tokens"])
        rows.append(row)
    shard = output / "shards" / f"shard-{rank:03d}.jsonl"
    write_jsonl(shard, rows)
    atomic_json(shard.with_suffix(".summary.json"), {
        "rank": rank, "gpu": devices[rank], "questions": len(rows),
        "response_tokens": sum(len(row["response_ids"]) for row in rows),
        "reasoning_positions": sum(len(row["positions"]) for row in rows),
        "seconds": time.monotonic() - started, "sha256": sha256_file(shard),
    })


def controller(args) -> None:
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output / "trajectory_plan.json"
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        requested = build_plan(args, output)
        # Ignore rewritten selected_questions bytes only after their hash agrees.
        if plan != requested:
            raise ValueError("Existing trajectory plan differs; use a new output directory")
    else:
        plan = build_plan(args, output)
        atomic_json(plan_path, plan)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    processes = []
    for rank in range(len(plan["gpus"])):
        shard = output / "shards" / f"shard-{rank:03d}.jsonl"
        if shard.exists():
            continue
        command = [sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:],
                   "--worker-rank", str(rank)]
        stream = (logs / f"trajectory-worker-{rank:03d}.log").open("w")
        processes.append((rank, subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT), stream))
    failures = []
    for rank, process, stream in processes:
        status = process.wait()
        stream.close()
        if status:
            failures.append((rank, status))
    if failures:
        raise RuntimeError(f"Trajectory workers failed: {failures}; inspect {logs}")
    all_rows = []
    for rank in range(len(plan["gpus"])):
        all_rows.extend(load_jsonl(output / "shards" / f"shard-{rank:03d}.jsonl"))
    all_rows.sort(key=lambda row: int(row["selected_index"]))
    if len(all_rows) != plan["num_questions"] or len({row["question_id"] for row in all_rows}) != len(all_rows):
        raise RuntimeError("Frozen trajectory coverage is incomplete or duplicated")
    for row in all_rows:
        validate_trajectory(row, max_sequence_tokens=plan["generation"]["max_sequence_tokens"])
    combined = output / "frozen_trajectories.jsonl"
    write_jsonl(combined, all_rows)
    atomic_json(output / "trajectory_manifest.json", {
        "schema_version": SCHEMA_VERSION, "status": "complete", "plan": plan,
        "trajectories": len(all_rows),
        "response_tokens": sum(len(row["response_ids"]) for row in all_rows),
        "reasoning_positions": sum(len(row["positions"]) for row in all_rows),
        "empty_reasoning_trajectories": sum(not row["positions"] for row in all_rows),
        "truncated_trajectories": sum(bool(row["truncated"]) for row in all_rows),
        "frozen_trajectories_sha256": sha256_file(combined),
    })
    print(f"ARCQ_EXP1_TRAJECTORIES=PASS\nTRAJECTORIES={combined}", flush=True)


def main(argv=None):
    args = arguments(argv)
    if args.worker_rank is None:
        controller(args)
    else:
        plan = json.loads((Path(args.output_dir).resolve() / "trajectory_plan.json").read_text())
        run_worker(args, plan)


if __name__ == "__main__":
    main()
