#!/usr/bin/env python3
"""Two-GPU HinD: causal and privileged Student on GPU 0, Teacher on GPU 1.

The privileged forward waits until the Student's rollout and causal collection
finish. The existing ren_balanced objective and optimizer remain in use.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import run_two_gpu_opd_opisd as common


def hind_command(assets: dict, output: Path, ids: list[str], rounds: int,
                 batch: int, max_new_tokens: int) -> list[str]:
    cmd = common.command(assets, output, "opd", ids, rounds, batch, max_new_tokens)
    cmd[cmd.index("--method") + 1] = "ren_balanced"
    cmd[cmd.index("--rollout-backend") + 1] = "hf"
    cmd[cmd.index("--rollout-batch-size") + 1] = "1"
    cmd += ["--reasoning-ablation", "ren", "--cost-mode", "--match-causal-update",
            "--control-loss-coef", "0.5", "--reference-kl-coef", "0.1",
            "--hindsight-gpus", ids[0], "--collocate-hindsight-student"]
    return cmd


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arcq-root", default=os.environ.get("ARCQ_ROOT"))
    parser.add_argument("--gpus", default="0,1", help="Student/HinD GPU, Teacher GPU")
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--global-batch-prompts", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--output-dir")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.arcq_root:
        parser.error("Source ARCQ_ROOT/env/arcq_env.sh or pass --arcq-root")
    root = Path(args.arcq_root).resolve(strict=True)
    ids = [part.strip() for part in args.gpus.split(",")]
    if len(ids) != 2 or len(set(ids)) != 2 or any(not x.isdigit() for x in ids):
        parser.error("--gpus must name exactly two distinct GPU indices")
    if min(args.rounds, args.global_batch_prompts, args.max_new_tokens) < 1:
        parser.error("Rounds, batch size and max new tokens must be positive")
    output = (Path(args.output_dir).resolve() if args.output_dir else
              root / "experiments/two_gpu_hind" / time.strftime("run_%Y%m%dT%H%M%SZ", time.gmtime()))
    if not output.is_relative_to(root / "experiments"):
        parser.error("Output must remain under ARCQ_ROOT/experiments")
    pool = root / "data/prepared/dapo_pool2048_s42/train.jsonl"
    if common.sha256(pool) != common.POOL_SHA:
        raise ValueError("Historical 2048-question pool SHA256 mismatch")
    assets = {"student": common.model_info(root / "models/Qwen3-1.7B"),
              "teacher": common.model_info(root / "models/Qwen3-32B"),
              "pool": str(pool), "pool_sha256": common.POOL_SHA}
    hardware = common.gpu_info(ids)
    if any(row["total_mib"] < 75 * 1024 for row in hardware.values()):
        raise RuntimeError("Two 80GB-class GPUs are required")
    teacher_needed = int((assets["teacher"]["weight_gib_on_disk"] + 10) * 1024)
    if hardware[ids[1]]["free_mib"] < teacher_needed:
        raise RuntimeError(f"Teacher GPU needs at least {teacher_needed/1024:.1f} GiB free")
    if args.run and hardware[ids[0]]["free_mib"] < 73 * 1024:
        raise RuntimeError("Student/HinD GPU needs at least 73 GiB free at launch")
    if args.run and shutil.disk_usage(root).free < 150 * 2**30:
        raise RuntimeError("Less than 150 GiB free for full Student and optimizer checkpoints")
    cmd = hind_command(assets, output / "train", ids, args.rounds,
                       args.global_batch_prompts, args.max_new_tokens)
    commit = subprocess.run(["git", "-C", str(common.CODE), "rev-parse", "HEAD"],
                            text=True, capture_output=True)
    plan = {"schema_version": 1, "method": "HinD/ren_balanced",
            "git_commit": commit.stdout.strip() if commit.returncode == 0 else None,
            "root": str(root), "output_dir": str(output), "assets": assets,
            "gpu_roles": {"student_and_privileged_student": ids[0], "teacher_single_gpu": ids[1]},
            "hardware": hardware, "rounds": args.rounds, "global_batch_prompts": args.global_batch_prompts,
            "rollouts_per_prompt": 1, "scheduled_prompt_uses": args.rounds * args.global_batch_prompts,
            "rollout_backend": "hf", "rollout_batch_size": 1,
            "collocated_order": "Student rollout/causal, then privileged Student; Teacher scores on the other GPU",
            "teacher_single_gpu_feasibility": "estimated_only; actual load/long-sequence smoke required",
            "command": cmd}
    print(json.dumps(plan, indent=2, ensure_ascii=False), flush=True)
    if not args.run:
        print("PREFLIGHT=PASS; add --run to start", flush=True)
        return 0
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Run directory already contains files: {output}")
    output.mkdir(parents=True)
    (output / "experiment_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    log_path = output / "train.log"
    started = time.time()
    returncode = common.run_command(cmd, log_path)
    checkpoint = output / "train/checkpoints" / f"round_{args.rounds:06d}"
    if returncode == 0 and not (checkpoint / "lulu_state.json").is_file():
        returncode = 1
        print(f"MISSING_FINAL_CHECKPOINT={checkpoint}", flush=True)
    status = {"returncode": returncode, "wall_seconds": time.time() - started,
              "log": str(log_path), "checkpoint": str(checkpoint)}
    (output / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps(status, indent=2), flush=True)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
