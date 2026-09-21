#!/usr/bin/env python3
"""Three matched Nemotron RQ1 controls; no ReN training is launched.

Default mode writes a reviewable plan. --check checks local inputs and model
interfaces without loading weights. --smoke runs one small update for one
selected arm in an isolated subdirectory. --run executes the three full
training arms in order, then evaluates Base and all three Round-4 checkpoints.
The caller supplies every cluster path; no model or dataset path is embedded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARMS = (("control_ref", "none"), ("vanilla_opd", "vanilla"), ("opsd", "opsd"))
BENCHMARKS = ("math500", "aime25", "olympiadbench", "mmlu_pro", "gpqa_diamond")
BENCHMARK_ROWS = {"math500": 500, "aime25": 30, "olympiadbench": 512,
                  "mmlu_pro": 512, "gpqa_diamond": 198}


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ("model", "teacher-model", "train-data", "data-manifest", "output-dir"):
        p.add_argument("--" + flag, required=True)
    p.add_argument("--gpus", default="0,1,2,3,4,5,6,7",
                   help="Eight ordered GPU IDs: first 3 Student, fourth Hindsight, last 4 Teacher TP")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rollout-vllm-memory", type=float, default=0.24)
    p.add_argument("--rollout-batch-size", type=int, default=8)
    p.add_argument("--score-batch-size", type=int, default=1)
    p.add_argument("--worker-timeout", type=int, default=3600)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="check inputs and interfaces; do not use GPUs")
    mode.add_argument("--smoke", action="store_true", help="one 8-prompt update to test model loading and backward")
    mode.add_argument("--run", action="store_true", help="run training and evaluation sequentially")
    p.add_argument("--smoke-arm", choices=[name for name, _ in ARMS], default="control_ref")
    return p.parse_args(argv)


def gpu_roles(value):
    ids = [item.strip() for item in value.split(",")]
    if len(ids) != 8 or len(set(ids)) != 8 or not all(item.isdecimal() for item in ids):
        raise ValueError("--gpus must list eight distinct numeric GPU IDs in role order")
    return ids, ids[:3], ids[3], ids[4:]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_inputs(args, *, load_interfaces):
    student, teacher = Path(args.model), Path(args.teacher_model)
    for label, root in (("Student", student), ("Teacher", teacher)):
        if not (root / "config.json").is_file():
            raise FileNotFoundError(f"{label} config.json missing: {root}")
        shards = list(root.glob("*.safetensors"))
        if not shards or not all(shard.is_file() for shard in shards):
            raise FileNotFoundError(f"{label} safetensors shards missing or broken: {root}")
        if not (root / "tokenizer_config.json").is_file():
            raise FileNotFoundError(f"{label} tokenizer_config.json missing: {root}")
    data = Path(args.train_data)
    manifest = Path(args.data_manifest)
    if not data.is_file() or not manifest.is_file():
        raise FileNotFoundError("Training JSONL and evaluation manifest must both exist")
    with data.open(encoding="utf-8") as handle:
        rows = sum(bool(line.strip()) for line in handle)
    if rows != 2048:
        raise ValueError(f"Expected the fixed 2048-question pool, got {rows} rows")
    pool_manifest = data.parent / "manifest.json"
    if not pool_manifest.is_file():
        raise FileNotFoundError(f"Training pool provenance manifest missing: {pool_manifest}")
    metadata = json.loads(pool_manifest.read_text())
    if metadata.get("pool_size") != 2048 or metadata.get("seed") != args.seed:
        raise ValueError("Training pool manifest disagrees with size=2048 or seed")
    expected_hash = metadata.get("outputs", {}).get("train", {}).get("sha256")
    if not expected_hash or sha256(data) != expected_hash:
        raise ValueError("Training data SHA256 differs from pool manifest")
    benchmarks = json.loads(manifest.read_text()).get("benchmarks", {})
    missing = sorted(set(BENCHMARKS) - set(benchmarks))
    if missing:
        raise ValueError(f"Evaluation manifest missing benchmarks: {missing}")
    for name in BENCHMARKS:
        path = Path(benchmarks[name]["full"])
        if not path.is_absolute():
            path = manifest.parent / path
        if not path.is_file():
            raise FileNotFoundError(f"{name} evaluation parquet missing: {path}")
    if not load_interfaces:
        return
    import pyarrow.parquet as pq
    from transformers import AutoConfig, AutoTokenizer
    from transformers.integrations.tensor_parallel import ALL_PARALLEL_STYLES
    from lulu.nemotron_family import apply_reasoning_template, nemotron_teacher_tp_plan
    from lulu.data import build_prompt_views, load_prepared_jsonl

    student_config = AutoConfig.from_pretrained(student, local_files_only=True)
    teacher_config = AutoConfig.from_pretrained(teacher, local_files_only=True, trust_remote_code=True)
    if student_config.model_type != "llama":
        raise ValueError(f"Expected Llama-based Student; got {student_config.model_type}")
    plan = nemotron_teacher_tp_plan(teacher_config, 4)
    unsupported = set(plan.values()) - set(ALL_PARALLEL_STYLES)
    if unsupported:
        raise RuntimeError(f"Installed Transformers lacks Teacher TP styles: {sorted(unsupported)}")
    weight_index = teacher / "model.safetensors.index.json"
    if not weight_index.is_file():
        raise FileNotFoundError(f"Teacher weight index missing: {weight_index}")
    weights = json.loads(weight_index.read_text()).get("weight_map", {})
    for projection in ("q_proj", "o_proj", "gate_proj", "down_proj"):
        pattern = re.compile(rf"^model\.layers\.\d+\.(?:self_attn|mlp)\.{projection}\.weight$")
        if not any(pattern.match(name) for name in weights):
            raise ValueError(f"Teacher TP projection is absent from checkpoint: {projection}")
    student_tok = AutoTokenizer.from_pretrained(student, local_files_only=True)
    teacher_tok = AutoTokenizer.from_pretrained(teacher, local_files_only=True)
    if student_config.vocab_size != teacher_config.vocab_size or student_tok.get_vocab() != teacher_tok.get_vocab():
        raise ValueError("Student and Teacher need identical token IDs and output vocabulary sizes")
    for marker in ("<think>", "</think>"):
        if not student_tok.encode(marker, add_special_tokens=False):
            raise ValueError(f"Student tokenizer cannot encode the reasoning marker {marker}")
    rendered = apply_reasoning_template(student_tok, [{"role": "user", "content": "Solve 1+1."}],
                                        family="nemotron", tokenize=False, add_generation_prompt=True)
    if "detailed thinking on" not in rendered:
        raise RuntimeError("Nemotron thinking-on system prompt was not rendered")
    prepared = load_prepared_jsonl(data)
    if len(prepared) != 2048:
        raise ValueError("Training JSONL is not a unique 2048-question pool")
    for row in prepared:
        views = build_prompt_views(student_tok, row["messages"], row["gold_answer"],
                                   model_family="nemotron")
        teacher_views = build_prompt_views(teacher_tok, row["messages"], row["gold_answer"],
                                           model_family="nemotron")
        if teacher_views["causal_prompt_ids"] != views["causal_prompt_ids"]:
            raise ValueError(f"Student/Teacher causal prompt token IDs differ for question {row['id']}")
        for name in ("causal", "hindsight"):
            student_ids = views[f"{name}_prompt_ids"]
            if len(student_ids) > 4096 or len(student_ids) + 8192 > 16384:
                raise ValueError(f"Nemotron prompt exceeds frozen context budget: {row['id']} ({name})")
    for name in BENCHMARKS:
        path = Path(benchmarks[name]["full"])
        if not path.is_absolute():
            path = manifest.parent / path
        count = pq.read_metadata(path).num_rows
        if count != BENCHMARK_ROWS[name] or benchmarks[name].get("full_examples") != count:
            raise ValueError(f"{name} expected {BENCHMARK_ROWS[name]} evaluation rows, got {count}")
        for index, example in enumerate(pq.read_table(path, columns=["prompt"]).to_pylist()):
            text = apply_reasoning_template(student_tok, example["prompt"],
                                             family="nemotron", tokenize=False,
                                             add_generation_prompt=True)
            if len(student_tok.encode(text, add_special_tokens=False)) > 4096:
                raise ValueError(f"Nemotron evaluation prompt exceeds 4096 tokens: {name}[{index}]")


def commands(args):
    ids, student, hindsight, teacher = gpu_roles(args.gpus)
    if args.seed != 42:
        raise ValueError("The frozen matched baseline recipe uses seed=42")
    if not 0 < args.rollout_vllm_memory < 1:
        raise ValueError("rollout vLLM memory fraction must be in (0,1)")
    root = Path(args.output_dir).expanduser().resolve()
    common = [sys.executable, "-u", str(ROOT / "scripts/train_lulu.py"),
        "--model", str(Path(args.model).resolve()), "--teacher-model", str(Path(args.teacher_model).resolve()),
        "--model-family", "nemotron", "--train-data", str(Path(args.train_data).resolve()),
        "--backend", "persistent", "--gpus", ",".join(ids),
        "--student-gpus", ",".join(student), "--hindsight-gpus", hindsight,
        "--teacher-gpus-per-worker", "4", "--teacher-tp-mode", "native",
        "--method", "ren_balanced", "--rounds", "4", "--global-batch-prompts", "256",
        "--rollouts-per-prompt", "1", "--update-passes", "1",
        "--rollout-backend", "vllm", "--rollout-batch-size", str(args.rollout_batch_size),
        "--rollout-top-k", "20", "--rollout-vllm-memory", str(args.rollout_vllm_memory),
        "--rollout-vllm-max-seqs", "32", "--rollout-vllm-sleep",
        "--score-batch-size", str(args.score_batch_size), "--train-micro-batch-size", "1",
        "--logit-chunk-size", "128", "--max-new-tokens", "8192",
        "--max-prompt-tokens", "4096", "--max-sequence-tokens", "16384",
        "--temperature", "0.6", "--top-p", "0.95", "--top-k", "32",
        "--learning-rate", "1e-6", "--weight-decay", "0", "--max-grad-norm", "1",
        "--control-loss-coef", "0.5", "--reference-kl-coef", "0.1",
        "--lora-rank", "0", "--master-weights-fp32", "--gradient-checkpointing",
        "--match-causal-update", "--validation-every", "0", "--save-every", "20",
        "--retain-checkpoints", "4", "--worker-timeout", str(args.worker_timeout),
        "--seed", str(args.seed)]
    train = {}
    for name, ablation in ARMS:
        cmd = common + ["--reasoning-ablation", ablation,
                        "--output-dir", str(root / "arms" / name / "train")]
        if name != "opsd":
            cmd += ["--teacher-gpus", ",".join(teacher)]
        train[name] = cmd
    evaluate = [sys.executable, "-u", str(ROOT / "scripts/evaluate_lulu.py"),
        "--model", str(Path(args.model).resolve()), "--model-family", "nemotron",
        "--include-base", "--data-manifest", str(Path(args.data_manifest).resolve()),
        "--benchmarks", ",".join(BENCHMARKS), "--split", "full", "--max-examples", "0",
        "--backend", "vllm", "--gpus", ",".join(ids), "--decoding", "nemotron-thinking",
        "--max-response-tokens", "32768", "--max-prompt-tokens", "4096",
        "--max-model-len", "36864", "--store-token-ids", "--store-text",
        "--seed", str(args.seed), "--output-dir", str(root / "evaluation")]
    for name, _ in ARMS:
        checkpoint = root / "arms" / name / "train" / "checkpoints" / "round_000004"
        evaluate += ["--checkpoint", f"{name}={checkpoint}"]
    return {"schema_version": 1, "task": "Nemotron RQ1 three matched baselines",
            "roles": {"student": student, "hindsight": [hindsight], "teacher_tp4": teacher},
            "model": str(Path(args.model).resolve()), "teacher_model": str(Path(args.teacher_model).resolve()),
            "train_data": str(Path(args.train_data).resolve()),
            "data_manifest": str(Path(args.data_manifest).resolve()),
            "train_commands": train, "eval_command": evaluate}


def completed_checkpoint(path):
    state_file = path / "lulu_state.json"
    if not state_file.is_file():
        return False
    state = json.loads(state_file.read_text())
    return state.get("completed_rounds") == 4 and state.get("checkpoint_manager", {}).get("final") is True


def completed_evaluation(path):
    if not path.is_file():
        return False
    result = json.loads(path.read_text())
    if set(result.get("models", {})) != ({"base"} | {name for name, _ in ARMS}):
        return False
    for model in result["models"].values():
        benchmarks = model.get("benchmarks", {})
        if set(benchmarks) != set(BENCHMARK_ROWS):
            return False
        for name, expected in BENCHMARK_ROWS.items():
            if benchmarks[name].get("examples") != expected or benchmarks[name].get("scored_examples") != expected:
                return False
    return True


def execute_smoke(args, plan):
    import torch
    ids, _, _, _ = gpu_roles(args.gpus)
    if any(int(item) >= torch.cuda.device_count() for item in ids):
        raise RuntimeError("Eight requested GPU IDs are not all visible to PyTorch")
    root = Path(args.output_dir).expanduser().resolve() / "smoke" / args.smoke_arm
    output = root / "train"
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Existing smoke output; inspect before rerunning: {output}")
    root.mkdir(parents=True, exist_ok=True)
    cmd = list(plan["train_commands"][args.smoke_arm])
    for flag, value in (("--rounds", "1"), ("--global-batch-prompts", "8"),
                        ("--output-dir", str(output))):
        cmd[cmd.index(flag) + 1] = value
    log = root / "training.log"
    env = dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    print(f"SMOKE {args.smoke_arm}: {log}", flush=True)
    with log.open("w") as handle:
        subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)
    state_file = output / "checkpoints" / "round_000001" / "lulu_state.json"
    if not state_file.is_file():
        raise RuntimeError(f"Smoke exited without Round-1 checkpoint: {state_file}")
    state = json.loads(state_file.read_text())
    if state.get("completed_rounds") != 1 or state.get("completed_updates") != 1 or not state.get(
            "checkpoint_manager", {}).get("final"):
        raise RuntimeError(f"Smoke did not finish its backward/optimizer update: {state_file}")
    print(f"SMOKE PASS {args.smoke_arm}: {state_file}", flush=True)


def execute(args, plan):
    import torch
    ids, _, _, _ = gpu_roles(args.gpus)
    if any(int(item) >= torch.cuda.device_count() for item in ids):
        raise RuntimeError("Eight requested GPU IDs are not all visible to PyTorch")
    root = Path(args.output_dir).expanduser().resolve()
    logs = root / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
    for name, _ in ARMS:
        final = root / "arms" / name / "train" / "checkpoints" / "round_000004"
        if completed_checkpoint(final):
            print(f"TRAIN already complete: {name}", flush=True)
            continue
        output = root / "arms" / name / "train"
        if output.exists() and any(output.iterdir()):
            raise RuntimeError(f"Incomplete existing {name} output; inspect before rerunning: {output}")
        log = logs / f"training_{name}.log"
        print(f"TRAIN {name}: {log}", flush=True)
        with log.open("w") as handle:
            subprocess.run(plan["train_commands"][name], cwd=ROOT, env=env,
                           stdout=handle, stderr=subprocess.STDOUT, check=True)
        if not completed_checkpoint(final):
            raise RuntimeError(f"Training exited without a final Round-4 checkpoint: {final}")
    summary = root / "evaluation" / "summary.json"
    if summary.exists():
        if not completed_evaluation(summary):
            raise RuntimeError(f"Existing evaluation summary is incomplete: {summary}")
        print(f"EVAL already complete: {summary}", flush=True)
        return
    log = logs / "evaluation.log"
    print(f"EVAL Base + 3 baselines: {log}", flush=True)
    with log.open("w") as handle:
        subprocess.run(plan["eval_command"], cwd=ROOT, env=env,
                       stdout=handle, stderr=subprocess.STDOUT, check=True)
    if not completed_evaluation(summary):
        raise RuntimeError(f"Evaluation exited without a complete four-model summary: {summary}")


def main(argv=None):
    args = arguments(argv)
    plan = commands(args)
    check_inputs(args, load_interfaces=args.check or args.smoke or args.run)
    root = Path(args.output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    plan_file = root / "experiment_plan.json"
    if plan_file.exists() and json.loads(plan_file.read_text()) != plan:
        raise RuntimeError(f"Existing experiment plan differs: {plan_file}")
    plan_file.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + "\n")
    print(f"PLAN {plan_file}", flush=True)
    if args.smoke:
        execute_smoke(args, plan)
    elif args.run:
        execute(args, plan)


if __name__ == "__main__":
    main()
