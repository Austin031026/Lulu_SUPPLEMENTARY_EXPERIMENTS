#!/usr/bin/env python3
"""Run the existing OPD and OPSD objectives with two A100 80GB GPUs.

The default performs asset and hardware checks; --run starts training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

CODE = Path(__file__).resolve().parents[1]
POOL_SHA = "7d8e0ef07b341b90a71958e4513bacdc3e086e152f55d47245c7dd9871bc5cc4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_info(path: Path) -> dict:
    config = path / "config.json"
    index = path / "model.safetensors.index.json"
    if json.loads(config.read_text()).get("model_type") != "qwen3":
        raise ValueError(f"Expected Qwen3 checkpoint: {path}")
    shards = set(json.loads(index.read_text())["weight_map"].values()) if index.is_file() else {"model.safetensors"}
    for name in (*shards, "tokenizer.json", "tokenizer_config.json"):
        if not (path / name).is_file():
            raise FileNotFoundError(path / name)
    return {"path": str(path.resolve()), "shards": len(shards),
            "weight_gib_on_disk": sum((path / name).stat().st_size for name in shards) / 2**30,
            "config_sha256": sha256(config)}


def gpu_info(ids: list[str]) -> dict[str, dict]:
    result = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.free",
                             "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True)
    rows = {}
    for line in result.stdout.splitlines():
        index, name, total, free = (value.strip() for value in line.split(",", 3))
        rows[index] = {"name": name, "total_mib": int(total), "free_mib": int(free)}
    if any(index not in rows for index in ids):
        raise ValueError(f"GPU indices {ids} unavailable; found {list(rows)}")
    return {index: rows[index] for index in ids}


def command(assets: dict, output: Path, method: str, ids: list[str], rounds: int,
            batch: int, max_new_tokens: int) -> list[str]:
    student, service = ids
    cmd = [sys.executable, "-u", str(CODE / "scripts/train_lulu.py"),
           "--model", assets["student"]["path"], "--teacher-model", assets["teacher"]["path"],
           "--train-data", assets["pool"], "--output-dir", str(output),
           "--method", "vanilla_opd" if method == "opd" else "opsd",
           "--backend", "persistent", "--gpus", ",".join(ids), "--student-gpus", student,
           "--rounds", str(rounds), "--global-batch-prompts", str(batch),
           "--rollouts-per-prompt", "1", "--update-passes", "1",
           "--rollout-backend", "vllm", "--rollout-batch-size", "4",
           "--rollout-top-k", "20", "--rollout-vllm-memory", "0.35",
           "--rollout-vllm-max-seqs", "8", "--score-batch-size", "1",
           "--train-micro-batch-size", "1", "--logit-chunk-size", "64",
           "--max-new-tokens", str(max_new_tokens), "--max-prompt-tokens", "4096",
           "--max-sequence-tokens", str(max(16384, 4096 + max_new_tokens)),
           "--temperature", "0.6", "--top-p", "0.95", "--top-k", "32",
           "--learning-rate", "1e-6", "--weight-decay", "0", "--max-grad-norm", "1",
           "--lora-rank", "0", "--master-weights-fp32", "--dtype", "bfloat16",
           "--gradient-checkpointing", "--gradient-norm-every", "0",
           "--validation-every", "0", "--no-stability-early-stop",
           "--save-every", "50", "--retain-checkpoints", f"0,{rounds}",
           "--worker-timeout", "7200", "--seed", "42"]
    cmd += (["--teacher-gpus", service, "--teacher-gpus-per-worker", "1"]
            if method == "opd" else ["--hindsight-gpus", service])
    return cmd


def run_command(cmd: list[str], log_path: Path) -> int:
    environment = dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
                       PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("x") as log:
        with subprocess.Popen(cmd, cwd=CODE, env=environment, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as child:
            for line in child.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            return child.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arcq-root", default=os.environ.get("ARCQ_ROOT"))
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--method", choices=("opd", "opisd", "both"), default="both")
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--global-batch-prompts", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--output-dir")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.arcq_root:
        parser.error("Set ARCQ_ROOT or pass --arcq-root")
    if min(args.rounds, args.global_batch_prompts, args.max_new_tokens) < 1:
        parser.error("Rounds, batch size and max-new-tokens must be positive")
    ids = [part.strip() for part in args.gpus.split(",")]
    if len(ids) != 2 or len(set(ids)) != 2 or any(not part.isdigit() for part in ids):
        parser.error("--gpus requires two distinct indices, e.g. 0,1")
    root = Path(args.arcq_root).resolve(strict=True)
    output = (Path(args.output_dir).resolve() if args.output_dir else
              root / "experiments/two_gpu_opd_opisd" / time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime()))
    if not output.is_relative_to(root / "experiments"):
        parser.error("Output must stay under ARCQ_ROOT/experiments")
    pool = root / "data/prepared/dapo_pool2048_s42/train.jsonl"
    if sha256(pool) != POOL_SHA:
        raise ValueError("Historical 2048-question pool SHA256 mismatch")
    assets = {"student": model_info(root / "models/Qwen3-1.7B"),
              "teacher": model_info(root / "models/Qwen3-32B"),
              "pool": str(pool), "pool_sha256": POOL_SHA}
    hardware = gpu_info(ids)
    if any(item["total_mib"] < 75 * 1024 for item in hardware.values()):
        raise RuntimeError("Two A100 80GB-class GPUs required")
    selected = ("opd", "opisd") if args.method == "both" else (args.method,)
    if "opd" in selected and hardware[ids[1]]["free_mib"] < int((assets["teacher"]["weight_gib_on_disk"] + 10) * 1024):
        raise RuntimeError("Teacher GPU lacks memory for single-GPU 32B attempt")
    if args.run and any(hardware[gpu]["free_mib"] < 73 * 1024 for gpu in ids):
        raise RuntimeError("Both GPUs need at least 73 GiB free before training")
    if args.run and shutil.disk_usage(root).free < 150 * 2**30:
        raise RuntimeError("Less than 150 GiB free for checkpoints")
    commands = {name: command(assets, output / name / "train", name, ids,
                              args.rounds, args.global_batch_prompts, args.max_new_tokens)
                for name in selected}
    commit = subprocess.run(["git", "-C", str(CODE), "rev-parse", "HEAD"], capture_output=True, text=True)
    plan = {"schema_version": 1, "mode": "two_gpu_opd_opisd", "git_commit": commit.stdout.strip(),
            "root": str(root), "output_dir": str(output), "assets": assets, "hardware": hardware,
            "rounds": args.rounds, "global_batch_prompts": args.global_batch_prompts,
            "rollouts_per_prompt": 1, "scheduled_prompt_uses": args.rounds * args.global_batch_prompts,
            "teacher_single_gpu_feasibility": "estimated_only; actual load/smoke required", "commands": commands}
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if not args.run:
        print("PREFLIGHT=PASS; add --run to start", flush=True)
        return 0
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.mkdir(parents=True)
    (output / "experiment_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for name in selected:
        log = output / name / "train.log"
        started = time.time()
        returncode = run_command(commands[name], log)
        checkpoint = output / name / "train/checkpoints" / f"round_{args.rounds:06d}"
        if returncode == 0 and not (checkpoint / "lulu_state.json").is_file():
            returncode = 1
        status = {"method": name, "returncode": returncode, "wall_seconds": time.time() - started,
                  "log": str(log), "checkpoint": str(checkpoint)}
        (output / name / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        print(json.dumps(status, indent=2), flush=True)
        if returncode:
            return returncode
    print(f"TRAINING_COMPLETE={','.join(selected)}; RUN_DIR={output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
