#!/usr/bin/env python3
"""Prepare or run ArcQ Experiment 1 end to end on a configurable question count."""
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
from lulu.exp1_predictiveness import atomic_json, sha256_file


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arcq-root", default=os.environ.get("ARCQ_ROOT"))
    parser.add_argument("--post-model", default=os.environ.get("ARCQ_EXP1_POST_MODEL"),
                        help="Vanilla OPD post-training checkpoint")
    parser.add_argument("--num-questions", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--output-dir")
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--run", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.arcq_root:
        raise ValueError("Source env/arcq_env.sh or pass --arcq-root")
    if not args.post_model:
        raise ValueError("Pass --post-model pointing to the completed Vanilla OPD checkpoint")
    workspace = Path(args.arcq_root).resolve(strict=True)
    post = Path(args.post_model).resolve(strict=True)
    questions = workspace / "data/diagnostic/exp1_dapo1024_s42/questions.jsonl"
    pre = workspace / "models/Qwen3-1.7B"
    teacher = workspace / "models/Qwen3-32B"
    for path in (questions, pre, teacher, post):
        if not path.exists():
            raise FileNotFoundError(path)
    devices = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if len(devices) < 3 or len(devices) != len(set(devices)):
        raise ValueError("Experiment 1 needs at least three distinct GPUs")
    teacher_gpus = devices[-2:]
    scorer_gpus = devices[:-2]
    output = (Path(args.output_dir).resolve() if args.output_dir else
              workspace / "experiments/exp1_predictiveness" /
              time.strftime(f"run_n{args.num_questions}_s{args.seed}_%Y%m%dT%H%M%SZ", time.gmtime()))
    if not output.is_relative_to(workspace / "experiments"):
        raise ValueError("Experiment output must be under ARCQ_ROOT/experiments")
    output.mkdir(parents=True, exist_ok=True)
    trajectory_dir = workspace / "data/diagnostic" / f"exp1_dapo{args.num_questions}_s{args.seed}" / "frozen"
    trajectories = trajectory_dir / "frozen_trajectories.jsonl"
    scoring = output / "scoring"
    analysis = output / "analysis"
    python = sys.executable
    commands = {
        "trajectories": [python, "-u", str(ROOT / "scripts/prepare_arcq_exp1_trajectories.py"),
            "--questions", str(questions), "--model", str(pre), "--output-dir", str(trajectory_dir),
            "--num-questions", str(args.num_questions), "--seed", str(args.seed),
            "--gpus", ",".join(devices)],
        "scoring": [python, "-u", str(ROOT / "scripts/score_arcq_exp1_fixed_prefixes.py"),
            "--trajectories", str(trajectories), "--pre-model", str(pre),
            "--post-model", str(post), "--teacher-model", str(teacher),
            "--output-dir", str(scoring), "--scorer-gpus", ",".join(scorer_gpus),
            "--teacher-gpus", ",".join(teacher_gpus)],
        "analysis": [python, "-u", str(ROOT / "scripts/analyze_arcq_exp1_predictiveness.py"),
            "--scores", str(scoring / "position_scores.parquet"),
            "--trajectories", str(trajectories), "--output-dir", str(analysis),
            "--bootstrap-replicates", str(args.bootstrap_replicates), "--seed", str(args.seed)],
    }
    plan = {
        "schema_version": 1, "experiment": "ArcQ Experiment 1 hindsight-score predictiveness",
        "num_questions": args.num_questions, "seed": args.seed, "gpus": devices,
        "inputs": {"questions": str(questions), "questions_sha256": sha256_file(questions),
                   "pre_model": str(pre), "post_model": str(post), "teacher_model": str(teacher)},
        "outputs": {"trajectories": str(trajectories), "scoring": str(scoring),
                    "analysis": str(analysis)},
        "commands": commands,
        "invariants": ["diagnostic questions are disjoint from OPD training",
                       "Base trajectories are generated once",
                       "pre and post scores use identical trajectory_id and token position",
                       "Teacher and post Student are answer-blind",
                       "post Student performs no generation",
                       "initial causal KL is controlled before hindsight grouping",
                       "uncertainty resamples questions, never tokens"],
    }
    plan_path = output / "experiment_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError("Existing experiment plan differs; use a new output directory")
    atomic_json(plan_path, plan)
    print(f"PLAN={plan_path}")
    if not args.run:
        print("ARCQ_EXP1_PLAN=READY")
        return
    logs = output / "logs"; logs.mkdir(exist_ok=True)
    stages = [
        ("trajectories", trajectory_dir / "trajectory_manifest.json"),
        ("scoring", scoring / "scoring_manifest.json"),
        ("analysis", analysis / "validation.json"),
    ]
    env = dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               PYTHONUNBUFFERED="1", OMP_NUM_THREADS="1")
    for name, completion in stages:
        if completion.exists() and json.loads(completion.read_text()).get("status") in ("complete", "PASS"):
            print(f"SKIP completed {name}", flush=True)
            continue
        log = logs / f"{name}.log"
        print(f"RUN {name}; log={log}", flush=True)
        with log.open("w") as stream:
            result = subprocess.run(commands[name], cwd=ROOT, env=env, stdout=stream,
                                    stderr=subprocess.STDOUT)
        if result.returncode:
            atomic_json(output / "status.json", {"state": "failed", "stage": name,
                                                   "log": str(log), "returncode": result.returncode})
            raise subprocess.CalledProcessError(result.returncode, commands[name])
    atomic_json(output / "status.json", {"state": "complete", "report": str(analysis / "REPORT.md")})
    print(f"ARCQ_EXP1=PASS\nREPORT={analysis/'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
