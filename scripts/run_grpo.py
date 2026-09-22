#!/usr/bin/env python3
"""Run Lulu's zero-reference-KL GRPO baseline locally or inside one Slurm node."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT / "scripts" / "grpo" / "train_grpo.py"
DEFAULT_DATA = ROOT / "data" / "grpo" / "train_prompts.parquet"
DEFAULT_PARSER = ROOT / "lulu" / "benchmark_parser.py"


def visible_devices(value: str) -> list[str]:
    if value == "auto":
        inherited = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if inherited:
            value = inherited
        else:
            try:
                lines = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                    text=True,
                ).splitlines()
            except (OSError, subprocess.CalledProcessError):
                lines = []
            value = ",".join(x.strip() for x in lines if x.strip())
    devices = [x.strip() for x in value.split(",") if x.strip()]
    if not devices:
        raise ValueError("No GPUs found; set --gpus or CUDA_VISIBLE_DEVICES")
    if len(set(devices)) != len(devices):
        raise ValueError("GPU identifiers must be unique")
    return devices


def load_reward_parser(path: Path):
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("grpo_reward_parser", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=os.environ.get("GRPO_MODEL", "Qwen/Qwen3-1.7B"))
    p.add_argument("--train-data", default=os.environ.get("GRPO_TRAIN_DATA", str(DEFAULT_DATA)))
    p.add_argument("--expected-train-data-sha256", default=os.environ.get("GRPO_TRAIN_DATA_SHA256"))
    p.add_argument("--parser-path", default=os.environ.get("GRPO_PARSER", str(DEFAULT_PARSER)))
    p.add_argument("--output-dir", required=True)
    p.add_argument("--gpus", default=os.environ.get("GRPO_GPUS", "auto"), help="auto, ordinals, or GPU UUIDs")
    p.add_argument("--tuning-mode", choices=["full", "lora"], default="full")
    p.add_argument("--prompt-mode", choices=["qwen3-thinking", "pretokenized"], default="qwen3-thinking")
    p.add_argument("--global-batch-prompts", type=int, default=48)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--global-epochs", type=float, default=1.0)
    p.add_argument("--max-response-tokens", type=int, default=8192)
    p.add_argument("--max-prompt-tokens", type=int, default=4096)
    p.add_argument("--lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--clip-ratio", type=float, default=0.2)
    p.add_argument("--ppo-epochs", type=int, default=1)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--max-steps", type=int, help="stop after this global step; reuse the output directory to resume")
    p.add_argument("--plan-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    try:
        devices = visible_devices(args.gpus)
    except ValueError as exc:
        if args.plan_only and args.gpus == "auto":
            devices = ["0"]
        else:
            raise SystemExit(str(exc)) from exc
    for name in ("global_epochs", "ppo_epochs", "max_response_tokens", "max_prompt_tokens"):
        if getattr(args, name) <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.max_steps is not None and args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive")

    data = Path(args.train_data).expanduser().resolve()
    parser_path = Path(args.parser_path).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    for path in (data, parser_path, TRAINER):
        if not path.is_file():
            raise FileNotFoundError(path)
    data_sha256 = hashlib.sha256(data.read_bytes()).hexdigest()
    if args.expected_train_data_sha256 and data_sha256 != args.expected_train_data_sha256.lower():
        raise ValueError(
            f"Training-data SHA256 mismatch: expected={args.expected_train_data_sha256.lower()} "
            f"actual={data_sha256} path={data}"
        )

    common = [
        "--algorithm", "grpo", "--model", args.model,
        "--train-data", str(data), "--parser-path", str(parser_path),
        "--output-dir", str(output), "--experiment-name", "lulu-grpo-baseline",
        "--tuning-mode", args.tuning_mode,
        "--prompt-mode", args.prompt_mode,
        # The exported objective is only rollout/update-consistent at T=1, top-p=1.
        "--temperature", "1.0", "--top-p", "1.0",
    ]
    for name in (
        "global_batch_prompts", "group_size", "global_epochs", "max_response_tokens",
        "max_prompt_tokens", "lr", "weight_decay", "clip_grad", "clip_ratio",
        "ppo_epochs", "lora_rank", "lora_alpha", "lora_dropout", "seed", "dtype",
    ):
        common.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])

    pythonpath = os.environ.get("PYTHONPATH", "")
    env = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        OMP_NUM_THREADS=os.environ.get("OMP_NUM_THREADS", "1"),
        PYTHONPATH=str(ROOT) + (os.pathsep + pythonpath if pythonpath else ""),
    )
    command = [sys.executable, str(TRAINER)]
    info = json.loads(subprocess.check_output(command + ["--phase", "plan"] + common, env=env, text=True))

    sources = set(info["data_sources"])
    if "livecodebench" in {source.lower() for source in sources}:
        raise ValueError("LiveCodeBench execution rewards are not implemented for GRPO training")
    if int(info["max_prompt_tokens_observed"]) > args.max_prompt_tokens:
        raise ValueError("Training prompt exceeds --max-prompt-tokens")
    reward = load_reward_parser(parser_path)
    for source in sorted(sources):
        if source.lower() in {"mmlu_pro", "gpqa_diamond"}:
            sample, expected, wrong = "The final answer is B.", "__CHOICE__B", "__CHOICE__C"
        else:
            sample, expected, wrong = r"The final answer is \boxed{2}.", "2", "3"
        prediction = reward.extract_answer(sample, source)
        if not reward.math_equal(prediction, expected) or reward.math_equal(prediction, wrong):
            raise RuntimeError(f"Reward parser preflight failed for {source}: {prediction!r}")

    info.update({
        "model": args.model,
        "train_data": str(data),
        "train_data_sha256": data_sha256,
        "parser_path": str(parser_path),
        "output_dir": str(output),
        "rollout_devices": devices,
        "reward_sources": sorted(sources),
        "tuning_mode": args.tuning_mode,
        "prompt_mode": args.prompt_mode,
    })
    print(json.dumps(info, indent=2), flush=True)
    if args.plan_only:
        return

    output.mkdir(parents=True, exist_ok=True)
    signature = {
        "schema_version": 1,
        "common_arguments": common,
        "prompt_mode": args.prompt_mode,
        "rollout_devices": devices,
        "train_data_sha256": data_sha256,
    }
    metadata = output / "portable_run_config.json"
    if metadata.exists():
        if json.loads(metadata.read_text()) != signature:
            raise RuntimeError("Output contains a different GRPO configuration; use a new output directory")
    elif any(output.glob("global_step_*")) or (output / "rollouts").exists():
        raise RuntimeError("Output contains an untracked run; use a new output directory")
    else:
        metadata.write_text(json.dumps(signature, indent=2) + "\n")

    logs = output / "logs"
    logs.mkdir(exist_ok=True)

    def run(cmd: list[str], log_name: str, gpu_list: str):
        print(f"[grpo] {log_name}", flush=True)
        with (logs / log_name).open("a") as handle:
            subprocess.run(
                cmd,
                env=dict(env, CUDA_VISIBLE_DEVICES=gpu_list),
                stdout=handle,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def complete(step: int) -> bool:
        parent = output / f"global_step_{step}"
        policy = parent / ("model" if args.tuning_mode == "full" else "adapter")
        config_name = "config.json" if args.tuning_mode == "full" else "adapter_config.json"
        if not (policy / config_name).is_file():
            return False
        weight_files = [
            policy / "model.safetensors", policy / "model.safetensors.index.json",
            policy / "pytorch_model.bin", policy / "adapter_model.safetensors",
        ]
        if not any(path.is_file() for path in weight_files):
            raise RuntimeError(f"Incomplete checkpoint: {parent}")
        if step and not all((parent / name).is_file() for name in ("optimizer.pt", "update_metrics.json")):
            raise RuntimeError(f"Incomplete checkpoint: {parent}")
        return True

    update_device = devices[0]
    all_devices = ",".join(devices)
    if not complete(0):
        run(command + ["--phase", "init"] + common, "init.log", update_device)
    stop = min(int(info["steps"]), args.max_steps or int(info["steps"]))
    for step in range(1, stop + 1):
        if complete(step):
            print(f"[grpo][resume] completed step {step}", flush=True)
            continue
        rollout_dir = output / "rollouts" / f"step_{step:06d}"
        phase = ["--step", str(step), "--rollout-dir", str(rollout_dir)]
        if len(devices) > 1:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            collect = [
                sys.executable, "-m", "torch.distributed.run", "--nnodes", "1",
                "--node_rank", "0", "--master_addr", "127.0.0.1",
                "--master_port", str(port), "--nproc_per_node", str(len(devices)),
                str(TRAINER),
            ]
        else:
            collect = command
        run(collect + ["--phase", "collect"] + phase + common, f"collect_{step:06d}.log", all_devices)
        run(command + ["--phase", "update"] + phase + common, f"update_{step:06d}.log", update_device)
        if not complete(step):
            raise RuntimeError(f"GRPO step {step} did not produce a complete checkpoint")
    policy_name = "model" if args.tuning_mode == "full" else "adapter"
    print(f"[grpo][done] {output / f'global_step_{stop}' / policy_name}", flush=True)


if __name__ == "__main__":
    main()
