#!/usr/bin/env python3
"""Score D0, D0H, D1, S, and G on frozen Experiment 1 prefixes."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import time
import traceback
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lulu import training as tr
from lulu.exp1_predictiveness import (
    SCHEMA_VERSION, atomic_json, exact_pre_post_kl, load_jsonl, sha256_file,
    validate_score_rows, validate_trajectory,
)
from lulu.persistent import Workers, head_from_state, teacher_payload
from lulu.teacher_service import teacher_worker


def free_port():
    with socket.socket() as stream:
        stream.bind(("127.0.0.1", 0))
        return stream.getsockname()[1]


def model_evidence(path: str | Path) -> dict:
    path = Path(path).resolve(strict=True)
    hashes = {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json",
                 "model.safetensors.index.json", "lulu_state.json"):
        candidate = path / name
        if candidate.is_file():
            hashes[name] = sha256_file(candidate)
    if "config.json" not in hashes or "tokenizer_config.json" not in hashes:
        raise ValueError(f"Incomplete model/checkpoint: {path}")
    return {"path": str(path), "sha256": hashes}


def scoring_worker(settings, gpu, conn):
    try:
        os.environ.update(CUDA_VISIBLE_DEVICES=str(gpu), TOKENIZERS_PARALLELISM="false",
                          LOCAL_RANK="0", RANK="0", WORLD_SIZE="1")
        torch.cuda.set_device(0)
        torch.set_num_threads(1)
        device = torch.device("cuda", 0)
        from transformers import AutoModelForCausalLM, AutoTokenizer
        kwargs = dict(torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                      trust_remote_code=False, low_cpu_mem_usage=True)
        tokenizer = AutoTokenizer.from_pretrained(settings["pre_model"], trust_remote_code=False)
        post_tokenizer = AutoTokenizer.from_pretrained(settings["post_model"], trust_remote_code=False)
        if tokenizer.get_vocab() != post_tokenizer.get_vocab():
            raise ValueError("Pre/post Student tokenizer vocabularies differ")
        del post_tokenizer
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        pre = AutoModelForCausalLM.from_pretrained(settings["pre_model"], **kwargs).to(device).eval()
        post = AutoModelForCausalLM.from_pretrained(settings["post_model"], **kwargs).to(device).eval()
        pre.requires_grad_(False); post.requires_grad_(False)
        pre_head, post_head = pre.get_output_embeddings(), post.get_output_embeddings()
        if pre_head.weight.shape[0] != post_head.weight.shape[0]:
            raise ValueError("Pre/post Student vocabulary sizes differ")
        helper = SimpleNamespace(cpu=False, dtype="bfloat16",
                                 max_sequence_tokens=settings["max_sequence_tokens"])
        conn.send({"op": "ready", "role": "exp1_scorer", "gpu": gpu})
        teacher_head = None
        while True:
            message = conn.recv()
            if message["op"] == "stop":
                conn.send({"op": "stopped"})
                return
            if message["op"] == "teacher_head":
                teacher_head = head_from_state(message["state"], device)
                if teacher_head.weight.shape[0] != pre_head.weight.shape[0]:
                    raise ValueError("Teacher and Student vocabulary sizes differ")
                conn.send({"op": "initialized", "gpu": gpu})
                continue
            if message["op"] != "score" or teacher_head is None:
                raise ValueError("Scorer received an invalid command")
            started = time.monotonic()
            records = message["records"]
            with torch.inference_mode():
                pre_causal = tr.selected_hidden(pre, tokenizer, records, "causal_prompt_ids", helper)
                pre_hindsight = tr.selected_hidden(pre, tokenizer, records, "hindsight_prompt_ids", helper)
                post_causal = tr.selected_hidden(post, tokenizer, records, "causal_prompt_ids", helper)
                results = []
                for record, teacher_hidden, hc0, hh0, hc1 in zip(
                        records, message["teacher_hidden"], pre_causal, pre_hindsight, post_causal):
                    pieces = {key: [] for key in (
                        "initial_causal_kl", "initial_hindsight_kl", "post_causal_kl",
                        "hindsight_score", "actual_improvement", "teacher_entropy")}
                    for start in range(0, len(record["positions"]), settings["logit_chunk_size"]):
                        stop = start + settings["logit_chunk_size"]
                        teacher_logits = teacher_head(teacher_hidden[start:stop].to(device))
                        values = exact_pre_post_kl(
                            teacher_logits, pre_head(hc0[start:stop]), pre_head(hh0[start:stop]),
                            post_head(hc1[start:stop]),
                        )
                        for key, value in values.items():
                            pieces[key].append(value.float().cpu().numpy())
                    results.append({key: (np.concatenate(value) if value else np.empty(0, np.float32))
                                    for key, value in pieces.items()})
            conn.send({"op": "scored", "request_id": message["request_id"],
                       "gpu": gpu, "scores": results,
                       "seconds": time.monotonic() - started})
    except BaseException as error:
        conn.send({"op": "error", "role": "exp1_scorer",
                   "error": f"{type(error).__name__}: {error}",
                   "traceback": traceback.format_exc()})
        raise
    finally:
        conn.close()


def score_table(records, scores):
    columns = {key: [] for key in (
        "question_id", "trajectory_id", "selected_index", "position", "token_id",
        "truncated", "initial_causal_kl", "initial_hindsight_kl", "post_causal_kl",
        "hindsight_score", "actual_improvement", "teacher_entropy")}
    for record, values in zip(records, scores):
        positions = list(record["positions"])
        if any(len(value) != len(positions) for value in values.values()):
            raise ValueError("Scorer returned the wrong number of positions")
        n = len(positions)
        columns["question_id"].extend([str(record["question_id"])] * n)
        columns["trajectory_id"].extend([str(record["trajectory_id"])] * n)
        columns["selected_index"].extend([int(record["selected_index"])] * n)
        columns["position"].extend(positions)
        columns["token_id"].extend(int(record["response_ids"][position]) for position in positions)
        columns["truncated"].extend([bool(record["truncated"])] * n)
        for key, value in values.items():
            columns[key].extend(map(float, value))
    return pa.table(columns)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--pre-model", required=True)
    parser.add_argument("--post-model", required=True)
    parser.add_argument("--teacher-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--scorer-gpus", default="0,1,2,3,4,5")
    parser.add_argument("--teacher-gpus", default="6,7")
    parser.add_argument("--score-batch-size", type=int, default=2)
    parser.add_argument("--logit-chunk-size", type=int, default=64)
    parser.add_argument("--max-sequence-tokens", type=int, default=16384)
    parser.add_argument("--worker-timeout", type=int, default=7200)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    trajectories_path = Path(args.trajectories).resolve(strict=True)
    records = load_jsonl(trajectories_path)
    if not records:
        raise ValueError("No frozen trajectories")
    for record in records:
        validate_trajectory(record, max_sequence_tokens=args.max_sequence_tokens)
    scored_records = [record for record in records if record["positions"]]
    if not scored_records:
        raise ValueError("No frozen trajectory contains an eligible reasoning position")
    scorer_gpus = [value.strip() for value in args.scorer_gpus.split(",") if value.strip()]
    teacher_gpus = [value.strip() for value in args.teacher_gpus.split(",") if value.strip()]
    if not scorer_gpus or not teacher_gpus or set(scorer_gpus) & set(teacher_gpus):
        raise ValueError("Scorer and Teacher GPU sets must be nonempty and disjoint")
    if args.score_batch_size < 1 or args.logit_chunk_size < 1:
        raise ValueError("Batch/chunk sizes must be positive")
    if Path(args.pre_model).resolve() == Path(args.post_model).resolve():
        raise ValueError("Pre and post Student checkpoints must differ")
    post_state_path = Path(args.post_model).resolve() / "lulu_state.json"
    if post_state_path.is_file():
        post_state = json.loads(post_state_path.read_text())
        if int(post_state.get("completed_updates", post_state.get("completed_rounds", 0))) < 1:
            raise ValueError("Post checkpoint does not record a completed training update")
    plan = {
        "schema_version": SCHEMA_VERSION, "stage": "fixed_prefix_pre_post_scoring",
        "trajectories": str(trajectories_path),
        "trajectories_sha256": sha256_file(trajectories_path),
        "trajectories_count": len(records),
        "scored_trajectories_count": len(scored_records),
        "empty_reasoning_trajectories": len(records) - len(scored_records),
        "pre_model": str(Path(args.pre_model).resolve(strict=True)),
        "post_model": str(Path(args.post_model).resolve(strict=True)),
        "teacher_model": str(Path(args.teacher_model).resolve(strict=True)),
        "model_evidence": {"pre": model_evidence(args.pre_model),
                           "post": model_evidence(args.post_model),
                           "teacher": model_evidence(args.teacher_model)},
        "scorer_gpus": scorer_gpus, "teacher_gpus": teacher_gpus,
        "score_batch_size": args.score_batch_size, "logit_chunk_size": args.logit_chunk_size,
        "max_sequence_tokens": args.max_sequence_tokens,
        "teacher_policy": "answer-blind causal prefix only",
        "post_policy": "causal fixed-prefix scoring only; no answer and no generation",
        "kl": "exact full-vocabulary forward KL(q_teacher || p_student)",
    }
    plan_path = output / "scoring_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Existing scoring plan differs; use a new output directory")
    atomic_json(plan_path, plan)
    final_path = output / "position_scores.parquet"
    if final_path.exists():
        raise FileExistsError(f"Scores already exist: {final_path}")
    temporary = output / "position_scores.incomplete.parquet"
    if temporary.exists():
        raise FileExistsError(f"Incomplete attempt retained: {temporary}; use a new output directory")

    worker_args = SimpleNamespace(
        cpu=False, model=plan["pre_model"], teacher_model=plan["teacher_model"],
        model_family="qwen", teacher_tp_mode="eager-local", dtype="bfloat16",
        method="ren_resolved", score_batch_size=args.score_batch_size,
        logit_chunk_size=args.logit_chunk_size, max_sequence_tokens=args.max_sequence_tokens,
        worker_timeout=args.worker_timeout,
    )
    settings = dict(pre_model=plan["pre_model"], post_model=plan["post_model"],
                    max_sequence_tokens=args.max_sequence_tokens,
                    logit_chunk_size=args.logit_chunk_size)
    workers = Workers(args.worker_timeout)
    writer = None
    started = time.monotonic()
    completed_trajectories = completed_positions = 0

    def write_result(message, batch):
        nonlocal writer, completed_trajectories, completed_positions
        table = score_table(batch, message["scores"])
        validate_score_rows(table.to_pylist())
        writer = writer or pq.ParquetWriter(temporary, table.schema, compression="zstd")
        writer.write_table(table)
        completed_trajectories += len(batch)
        completed_positions += len(table)
        atomic_json(output / "live_progress.json", {
            "status": "scoring", "completed_trajectories": completed_trajectories,
            "total_trajectories": len(scored_records), "completed_positions": completed_positions,
            "elapsed_seconds": time.monotonic() - started,
        })
        print(json.dumps({"completed": completed_trajectories, "total": len(scored_records),
                          "positions": completed_positions}), flush=True)

    try:
        port = free_port()
        teacher = workers.launch(teacher_worker,
                                 (worker_args, 0, len(teacher_gpus), teacher_gpus, "127.0.0.1", port))
        for rank in range(1, len(teacher_gpus)):
            workers.launch(teacher_worker,
                           (worker_args, rank, len(teacher_gpus), teacher_gpus, "127.0.0.1", port),
                           connected=False)
        scorers = [workers.launch(scoring_worker, (settings, gpu)) for gpu in scorer_gpus]
        ready_teacher = workers.expect(teacher, "ready")
        head_state = ready_teacher.pop("teacher_head")
        for scorer in scorers:
            workers.expect(scorer, "ready")
            scorer.send({"op": "teacher_head", "state": head_state})
        for scorer in scorers:
            workers.expect(scorer, "initialized")
        del head_state
        idle = list(scorers)
        busy = {}
        request_id = 0
        for start in range(0, len(scored_records), args.score_batch_size):
            batch = scored_records[start:start + args.score_batch_size]
            teacher.send({"op": "score", "round": 0, "request_id": request_id,
                          "records": teacher_payload(batch)})
            teacher_result = workers.expect(teacher, "scored")
            if not idle:
                scorer, result = workers.receive(list(busy))
                write_result(result, busy.pop(scorer))
                idle.append(scorer)
            scorer = idle.pop()
            scorer.send({"op": "score", "request_id": request_id, "records": batch,
                         "teacher_hidden": [value["teacher_hidden"] for value in teacher_result["records"]]})
            busy[scorer] = batch
            request_id += 1
        while busy:
            scorer, result = workers.receive(list(busy))
            write_result(result, busy.pop(scorer))
        if writer is None:
            raise RuntimeError("No reasoning positions were scored")
        writer.close(); writer = None
        temporary.replace(final_path)
        manifest = {
            "schema_version": SCHEMA_VERSION, "status": "complete", "plan": plan,
            "trajectories": len(records), "scored_trajectories": completed_trajectories,
            "empty_reasoning_trajectories": len(records) - completed_trajectories,
            "positions": completed_positions,
            "seconds": time.monotonic() - started,
            "position_scores": str(final_path), "position_scores_sha256": sha256_file(final_path),
        }
        atomic_json(output / "scoring_manifest.json", manifest)
        atomic_json(output / "live_progress.json", {"status": "complete", **manifest})
        print(f"ARCQ_EXP1_SCORING=PASS\nSCORES={final_path}", flush=True)
    except BaseException as error:
        atomic_json(output / "live_progress.json", {"status": "failed", "error": repr(error),
                                                     "preserved_output": str(output)})
        raise
    finally:
        if writer is not None:
            writer.close()
        workers.close()


if __name__ == "__main__":
    main()
