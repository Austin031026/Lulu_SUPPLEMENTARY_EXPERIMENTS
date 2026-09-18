#!/usr/bin/env python3
"""Batched ordinary-policy evaluation for Lulu / ReN-OPD checkpoints.

All evaluation components are bundled inside the Lulu project.
"""
from __future__ import annotations

import argparse
from collections import deque
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from lulu.paths import (
    DEFAULT_OUTPUT_ROOT,
    PROJECT_SCRIPTS,
    benchmark_parser_path,
)
from lulu.evaluation_runner import PlainModelRunner, load_parser

DEFAULT_BENCHMARKS = ("math500", "aime25", "olympiadbench", "mmlu_pro", "gpqa_diamond")


def named_value(value):
    name, sep, location = value.partition("=")
    if not sep or not location or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError(f"expected NAME=PATH with a simple unique NAME, got {value!r}")
    return name, location


def resolve_devices(value):
    if value == "auto":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible is not None:
            value = visible
        else:
            try:
                output = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
                )
                value = ",".join(output.split())
            except (OSError, subprocess.CalledProcessError):
                value = ""
    devices = [x.strip() for x in value.split(",") if x.strip()]
    if not devices or "-1" in devices:
        raise ValueError("No visible GPUs; set --gpus 0,1,... (or --gpus cpu for a tiny CPU test)")
    if len(set(devices)) != len(devices) or ("cpu" in devices and len(devices) != 1):
        raise ValueError("GPU identifiers must be unique; cpu must be used alone")
    return devices


def resolve_model_reference(value):
    """Freeze local paths before recording a plan; leave Hub identifiers intact."""
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.exists() or value.startswith((".", "/", "~")):
        return str(path.resolve())
    return value


def build_plan(args):
    if args.batch_size <= 0 or args.max_response_tokens <= 0 or args.max_prompt_tokens <= 0:
        raise ValueError("batch-size and token limits must be positive")
    if args.max_examples < 0:
        raise ValueError("max-examples must be nonnegative")
    if args.lcb_processes <= 0:
        raise ValueError("lcb-processes must be positive")
    devices = resolve_devices(args.gpus)
    backend = ('hf' if devices == ['cpu'] else 'vllm') if args.backend == 'auto' else args.backend
    if backend == 'vllm' and devices == ['cpu']:
        raise ValueError('vLLM evaluation requires CUDA GPUs; use --backend hf on CPU')
    if args.context_safety_margin < 0:
        raise ValueError('context-safety-margin must be nonnegative')
    if args.max_model_len is not None:
        if backend != 'vllm' or args.max_model_len <= args.context_safety_margin:
            raise ValueError('max-model-len requires vLLM and must exceed the safety margin')
    elif args.context_safety_margin:
        raise ValueError('context-safety-margin requires an explicit max-model-len')
    if args.store_token_ids and backend != 'vllm':
        raise ValueError('store-token-ids currently requires vLLM')
    if not 0 < args.vllm_gpu_memory_utilization < 1 or min(args.vllm_max_num_seqs, args.vllm_max_num_batched_tokens) < 1:
        raise ValueError('Invalid vLLM memory utilization or scheduler limits')
    decoding = ("qwen-thinking" if args.thinking else "greedy") if args.decoding == "auto" else args.decoding
    sampling = ({"temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": args.seed}
                if decoding == "qwen-thinking" else {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "seed": args.seed})
    models = []
    if not args.checkpoint or args.include_base:
        models.append({"name": "base", "model": resolve_model_reference(args.model)})
    models.extend({"name": name, "model": resolve_model_reference(path)}
                  for name, path in map(named_value, args.checkpoint))
    if len({x["name"] for x in models}) != len(models):
        raise ValueError("checkpoint names must be unique; base is reserved when evaluating the base")
    source = {}
    manifest_root = Path.cwd()
    if args.data_manifest:
        manifest_path = Path(args.data_manifest).expanduser().resolve()
        source = json.loads(manifest_path.read_text())["benchmarks"]
        manifest_root = manifest_path.parent
    direct = dict(map(named_value, args.benchmark))
    if len(direct) != len(args.benchmark):
        raise ValueError("benchmark names must be unique")
    if not source and not direct:
        raise ValueError("provide --data-manifest or --benchmark NAME=PARQUET")
    if args.benchmarks == "all":
        names = list(dict.fromkeys([*source, *direct]))
    elif args.benchmarks:
        names = [x.strip() for x in args.benchmarks.split(",") if x.strip()]
    elif direct:
        names = list(direct)
    else:
        names = list(DEFAULT_BENCHMARKS)
    if not names or len(set(names)) != len(names):
        raise ValueError("select at least one benchmark; benchmark names must be unique")
    benchmarks = []
    for name in names:
        named_value(f"{name}=unused")  # Names become output path components.
        metadata = dict(source.get(name, {}))
        if name in direct:
            path = Path(direct[name]).expanduser().resolve()
            metadata.pop(f"{args.split}_examples", None)
        else:
            if name not in source:
                raise ValueError(f"benchmark {name!r} missing from manifest; set --benchmarks explicitly")
            path = Path(metadata[args.split]).expanduser()
            if not path.is_absolute():
                path = manifest_root / path
            path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"benchmark {name}: {path}")
        default_scorer = ("livecodebench" if name == "livecodebench" else
                          "choice" if name in {"mmlu_pro", "gpqa_diamond"} else "math")
        scorer = metadata.get("scorer", default_scorer)
        if scorer == "livecodebench":
            if not args.lcb_repo or not Path(args.lcb_repo).expanduser().is_dir():
                raise ValueError("LiveCodeBench requires --lcb-repo pointing to the official evaluator checkout")
            if args.max_examples:
                raise ValueError("LiveCodeBench official scorer requires the complete split; omit --max-examples")
            if not metadata.get("livecodebench", {}).get("release_version"):
                raise ValueError("LiveCodeBench requires release_version metadata in --data-manifest")
        expected = metadata.get(f"{args.split}_examples")
        if expected is not None and args.max_examples:
            expected = min(int(expected), args.max_examples)
        benchmarks.append({"name": name, "path": str(path), "scorer": scorer,
                           "expected_examples": expected, "livecodebench": metadata.get("livecodebench")})

    parser_path = benchmark_parser_path(args.parser_path)
    return {
        "schema_version": 1, "policy": "ordinary_student_" + decoding, "models": models,
        "decoding": decoding, "sampling": sampling,
        "benchmarks": benchmarks, "split": args.split, "devices": devices, "backend": backend,
        "vllm": {"gpu_memory_utilization": args.vllm_gpu_memory_utilization,
                 "max_num_seqs": args.vllm_max_num_seqs,
                 "max_num_batched_tokens": args.vllm_max_num_batched_tokens,
                 "enforce_eager": args.vllm_enforce_eager},
        "batch_size": args.batch_size, "max_response_tokens": args.max_response_tokens,
        "max_prompt_tokens": args.max_prompt_tokens, "max_examples": args.max_examples,
        "max_model_len": args.max_model_len, "context_safety_margin": args.context_safety_margin,
        "store_token_ids": args.store_token_ids,
        "thinking": args.thinking, "dtype": args.dtype, "store_text": args.store_text,
        "trust_remote_code": args.trust_remote_code,
        "adapter_base_model": resolve_model_reference(args.adapter_base_model),
        "parser_path": str(parser_path), "output_dir": str(Path(args.output_dir).expanduser().resolve()),
        "lcb_repo": str(Path(args.lcb_repo).expanduser().resolve()) if args.lcb_repo else None,
        "lcb_python": args.lcb_python, "lcb_processes": args.lcb_processes,
    }


def load_model_assets(model_id, *, dtype, device, thinking, trust_remote_code=False, adapter_base_model=None):
    """Load a full HF student or an ordinary PEFT student adapter exactly once."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    path = Path(model_id).expanduser()
    adapter = (path / "adapter_config.json").is_file()
    kwargs = dict(torch_dtype=dtype, trust_remote_code=trust_remote_code, low_cpu_mem_usage=True)
    if adapter:
        from peft import PeftConfig, PeftModel
        config = PeftConfig.from_pretrained(str(path))
        base_id = adapter_base_model or config.base_model_name_or_path
        tokenizer_id = str(path) if (path / "tokenizer_config.json").is_file() else base_id
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, trust_remote_code=trust_remote_code)
        base = AutoModelForCausalLM.from_pretrained(base_id, **kwargs)
        model = PeftModel.from_pretrained(base, str(path), is_trainable=False)
        # This is inference: merge once on CPU instead of executing separate
        # low-rank projections at every decoding token on the GPU.
        model = model.merge_and_unload(safe_merge=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must define a pad or EOS token")
    tokenizer.padding_side = "left"
    model = model.to(device).eval()
    model.config.use_cache = True
    # Reset saved generation defaults; the evaluation runner explicitly selects
    # greedy or Qwen thinking sampling from the recorded evaluation plan.
    model.generation_config = GenerationConfig(
        do_sample=False, num_beams=1, repetition_penalty=1.0,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

    class ChatTokenizer:
        def __getattr__(self, key):
            return getattr(tokenizer, key)

        def __call__(self, *args, **kwargs):
            return tokenizer(*args, **kwargs)

        def apply_chat_template(self, *args, **kwargs):
            kwargs["enable_thinking"] = thinking
            return tokenizer.apply_chat_template(*args, **kwargs)

    return model, ChatTokenizer()

def make_runner(plan, device):
    class LuluRunner(PlainModelRunner):
        def load_model(self, model_id, **kwargs):
            if self.model_id == model_id and self.model is not None:
                return
            self.close()
            self.model, self.tokenizer = load_model_assets(
                model_id, dtype=self.dtype, device=device, thinking=plan["thinking"],
                trust_remote_code=plan["trust_remote_code"],
                adapter_base_model=plan["adapter_base_model"],
            )
            self.model_id = model_id
            print(f"[lulu-eval][READY] model={model_id} device={device}", flush=True)

    return LuluRunner(parser_path=plan["parser_path"], batch_size=plan["batch_size"],
                      max_response_tokens=plan["max_response_tokens"],
                      max_prompt_tokens=plan["max_prompt_tokens"], dtype=plan["dtype"],
                      sampling=plan.get("sampling"))


def run_worker(plan, shard_id):
    if plan.get('backend', 'hf') == 'vllm':
        from lulu.vllm_evaluation import run_worker as run_vllm_worker
        return run_vllm_worker(plan, shard_id)
    device = "cpu" if plan["devices"] == ["cpu"] else "cuda:0"
    runner = make_runner(plan, device)
    try:
        for model in plan["models"]:
            for benchmark in plan["benchmarks"]:
                output = Path(plan["output_dir"]) / model["name"] / benchmark["name"]
                runner.evaluate_shard(
                    model_id=model["model"], input_parquet=benchmark["path"],
                    output=str(output / f"shard-{shard_id:03d}.jsonl"),
                    shard_id=shard_id, num_shards=len(plan["devices"]),
                    max_examples=plan["max_examples"], store_text=plan["store_text"], progress_every=8,
                )
    finally:
        runner.close()


def load_complete_rows(directory, num_shards, expected_examples):
    """Only accept exactly the current shard set, with unique complete row coverage."""
    directory = Path(directory)
    expected_files = {f"shard-{i:03d}.jsonl" for i in range(num_shards)}
    actual_files = {p.name for p in directory.glob("shard-*.jsonl")}
    if actual_files != expected_files:
        raise ValueError(f"incomplete or stale shards in {directory}: {actual_files ^ expected_files}")
    indexed = {}
    for sid in range(num_shards):
        for line in (directory / f"shard-{sid:03d}.jsonl").read_text().split('\n'):
            if not line.strip():
                continue
            row = json.loads(line)
            index = int(row["prompt_index"])
            if index in indexed or index % num_shards != sid:
                raise ValueError(f"duplicate or incorrectly assigned prompt_index={index} in {directory}")
            indexed[index] = row
    if set(indexed) != set(range(expected_examples)):
        raise ValueError(f"generation coverage {len(indexed)} does not match {expected_examples} in {directory}")
    return [indexed[i] for i in range(expected_examples)]


def summarize_rows(rows):
    rewards = [float(x["reward"]) for x in rows if x.get("reward") is not None]
    n = len(rows)
    return {
        "schema_version": 1, "examples": n, "scored_examples": len(rewards),
        "accuracy": sum(rewards) / len(rewards) if rewards else None,
        "mean_response_tokens": sum(x["response_tokens"] for x in rows) / n if n else None,
        "mean_prompt_tokens": sum(x["prompt_tokens"] for x in rows) / n if n else None,
        "hit_cap_fraction": sum(bool(x["hit_cap"]) for x in rows) / n if n else None,
    }


def paired_comparison(base, candidate):
    if [x["prompt_index"] for x in base] != [x["prompt_index"] for x in candidate]:
        raise ValueError("base and checkpoint row indices must match")
    pairs = [(a["reward"], b["reward"]) for a, b in zip(base, candidate)
             if a.get("reward") is not None and b.get("reward") is not None]
    return {"paired_examples": len(pairs),
            "accuracy_delta": sum(float(b) - float(a) for a, b in pairs) / len(pairs) if pairs else None,
            "rescues": sum(a == 0 and b == 1 for a, b in pairs),
            "degradations": sum(a == 1 and b == 0 for a, b in pairs)}


def score_livecodebench(plan, benchmark, raw_dir):
    scripts = PROJECT_SCRIPTS
    export_script = scripts / "export_livecodebench_custom.py"
    inject_script = scripts / "inject_livecodebench_rewards.py"

    if not export_script.is_file():
        raise FileNotFoundError(export_script)
    if not inject_script.is_file():
        raise FileNotFoundError(inject_script)
    custom = raw_dir / "livecodebench_custom.json"
    subprocess.run([sys.executable, str(export_script),
                    "--data", benchmark["path"], "--generation-dir", str(raw_dir),
                    "--output", str(custom)], check=True)
    
    subprocess.run([plan["lcb_python"], "-m", "lcb_runner.runner.custom_evaluator",
                    "--custom_output_file", str(custom), "--scenario", "codegeneration",
                    "--release_version", benchmark["livecodebench"]["release_version"],
                    "--n", "1", "--temperature", str((plan.get("sampling") or {}).get("temperature", 0.0)),
                    "--num_process_evaluate", str(plan["lcb_processes"])],
                   cwd=plan["lcb_repo"], check=True)
    scored = raw_dir / "scored"
    subprocess.run([sys.executable, str(inject_script),
                    "--data", benchmark["path"], "--generation-dir", str(raw_dir),
                    "--lcb-eval-all", str(custom.with_name(custom.stem + "_codegeneration_output_eval_all.json")),
                    "--output-dir", str(scored)], check=True)
    return scored


def finish_suite(plan):
    root = Path(plan["output_dir"])
    result = {"schema_version": 1, "decoding": plan.get("decoding", "greedy"), "thinking": plan["thinking"],
              "sampling": plan.get("sampling"),
              "split": plan["split"], "backend": plan.get("backend", "hf"),
              "max_examples_per_benchmark": plan["max_examples"],
              "selection": "first_rows_before_sharding", "models": {}}
    base_rows = {}
    for model in plan["models"]:
        summaries = {}
        for benchmark in plan["benchmarks"]:
            directory = root / model["name"] / benchmark["name"]
            rows = load_complete_rows(directory, len(plan["devices"]), benchmark["expected_examples"])
            if benchmark["scorer"] == "livecodebench":
                scored = score_livecodebench(plan, benchmark, directory)
                rows = load_complete_rows(scored, len(plan["devices"]), benchmark["expected_examples"])
            if any(x.get("reward") is None for x in rows):
                raise ValueError(f"unscored rows remain in {directory}")
            summary = summarize_rows(rows)
            if model["name"] == "base":
                base_rows[benchmark["name"]] = rows
            elif benchmark["name"] in base_rows:
                summary["vs_base"] = paired_comparison(base_rows[benchmark["name"]], rows)
            summaries[benchmark["name"]] = summary
            (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        accuracies = [s["accuracy"] for s in summaries.values() if s["accuracy"] is not None]
        result["models"][model["name"]] = {"model": model["model"], "benchmarks": summaries,
            "macro_accuracy": sum(accuracies) / len(accuracies) if accuracies else None}
    (root / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return result


def worker_environment(plan, shard_id):
    """Independent TP=1 engines must not race on the same rank-0 compile cache."""
    env = dict(os.environ)
    gpu = plan['devices'][shard_id]
    env['CUDA_VISIBLE_DEVICES'] = '' if gpu == 'cpu' else gpu
    env.setdefault('TOKENIZERS_PARALLELISM', 'false')
    env.setdefault('OMP_NUM_THREADS', '1')
    if plan.get('backend', 'hf') == 'vllm':
        cache = Path(plan['output_dir'])/'.engine_cache'/f'shard-{shard_id:03d}'
        for key, folder in [('VLLM_CACHE_ROOT', 'vllm'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor'),
                            ('TRITON_CACHE_DIR', 'triton')]:
            path = cache/folder
            path.mkdir(parents=True, exist_ok=True)
            env[key] = str(path)
    return env


def run_suite(plan):
    import pyarrow.parquet as pq

    parser = load_parser(plan["parser_path"])
    if any(b["scorer"] == "math" for b in plan["benchmarks"]) and hasattr(parser, "_math_parser"):
        parser._math_parser()
    for benchmark in plan["benchmarks"]:
        total = pq.ParquetFile(benchmark["path"]).metadata.num_rows
        actual = min(total, plan["max_examples"]) if plan["max_examples"] else total
        if actual <= 0:
            raise ValueError(f"empty benchmark: {benchmark['name']}")
        if benchmark["expected_examples"] is not None and benchmark["expected_examples"] != actual:
            raise ValueError(f"manifest example count disagrees with parquet for {benchmark['name']}")
        benchmark["expected_examples"] = actual
    root = Path(plan["output_dir"])
    if root.exists() and any(root.iterdir()):
        raise ValueError(f"output directory must be empty to prevent mixing runs: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if plan.get('backend', 'hf') == 'vllm':
        import importlib.util
        if importlib.util.find_spec('vllm') is None:
            raise RuntimeError('vLLM is unavailable in this Python environment; select --backend hf explicitly')
        from lulu.vllm_evaluation import prepare_models
        prepare_models(plan, load_model_assets)
    plan_path = root / "eval_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    workers = []
    logs = []
    try:
        phases = [[model] for model in plan['models']] if plan.get('backend', 'hf') == 'vllm' else [plan['models']]
        pending = deque()
        for phase in phases:
            current_plan = plan_path
            suffix = ''
            if plan.get('backend', 'hf') == 'vllm':
                suffix = '-'+phase[0]['name']
                current_plan = root/f'worker-plan{suffix}.json'
                current_plan.write_text(json.dumps(dict(plan, models=phase), indent=2)+'\n')
            for shard_id in range(len(plan['devices'])):
                pending.append((shard_id, current_plan, suffix))
        active = {}
        with (root/'dispatch.jsonl').open('w') as dispatch:
            while pending or active:
                for slot, worker in list(active.items()):
                    code = worker.poll()
                    if code not in (None, 0):
                        raise RuntimeError(f'evaluation worker on GPU {plan["devices"][slot]} failed: {code}; see {root}/worker*.log')
                    if code == 0:
                        del active[slot]
                for slot, gpu in enumerate(plan['devices']):
                    if slot in active or not pending:
                        continue
                    shard_id, current_plan, suffix = pending.popleft()
                    # Cache and CUDA device follow the physical slot; data and
                    # request seeds follow the logical shard, independently.
                    env = worker_environment(plan, slot)
                    log_path = root/f'worker{suffix}-{shard_id:03d}.log'
                    log = log_path.open('w')
                    logs.append(log)
                    worker = subprocess.Popen(
                        [sys.executable, '-u', str(Path(__file__).resolve()), '--worker-plan', str(current_plan),
                         '--shard-id', str(shard_id)], env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                    workers.append(worker)
                    active[slot] = worker
                    dispatch.write(json.dumps({'gpu':gpu,'shard_id':shard_id,'worker_plan':str(current_plan),
                                               'log':str(log_path),'started_unix':time.time()})+'\n')
                    dispatch.flush()
                    print(f'[lulu-eval] gpu={gpu} shard={shard_id} backend={plan.get("backend", "hf")} log={log_path}', flush=True)
                if active:
                    time.sleep(1)
    finally:
        for worker in workers:
            if worker.poll() is None:
                try:
                    os.killpg(worker.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for worker in workers:
            try:
                worker.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(worker.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                worker.wait()
        for log in logs:
            log.close()
    finish_suite(plan)


def argument_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="Qwen/Qwen3-1.7B", help="base student identifier")
    p.add_argument("--checkpoint", action="append", default=[], metavar="NAME=PATH", help="repeat for HF/PEFT checkpoints")
    p.add_argument("--include-base", action="store_true", help="also evaluate base and compute paired deltas")
    p.add_argument("--adapter-base-model", help="override the base path stored in local PEFT adapters")
    p.add_argument("--data-manifest", help="existing prepare_crossbench_data.py manifest.json")
    p.add_argument("--benchmark", action="append", default=[], metavar="NAME=PARQUET")
    p.add_argument("--benchmarks", help="comma-separated manifest names; default five math/general benchmarks; 'all' includes coding")
    p.add_argument("--split", choices=["probe", "full"], default="full")
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "evaluation"))
    p.add_argument("--gpus", default="auto", help="physical GPU IDs/UUIDs; auto respects CUDA_VISIBLE_DEVICES")
    p.add_argument("--batch-size", type=int, default=8, help="HF static batch size; vLLM uses its scheduler limits")
    p.add_argument("--backend", choices=["auto", "hf", "vllm"], default="auto",
                   help="auto selects vLLM on GPU and HF on CPU")
    p.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.8)
    p.add_argument("--vllm-max-num-seqs", type=int, default=64)
    p.add_argument("--vllm-max-num-batched-tokens", type=int, default=4096)
    p.add_argument("--vllm-enforce-eager", action="store_true", help="disable CUDA graphs for debugging")
    p.add_argument("--max-response-tokens", type=int, default=8192)
    p.add_argument("--max-prompt-tokens", type=int, default=4096)
    p.add_argument("--max-model-len", type=int, help="vLLM total context; cap responses to fit complete prompts")
    p.add_argument("--context-safety-margin", type=int, default=0, help="reserved tokens within max-model-len")
    p.add_argument("--store-token-ids", action="store_true", help="save response token IDs for positional audits")
    p.add_argument("--max-examples", type=int, default=199,
                   help="per-benchmark cap before GPU sharding; fixed first rows shared by all checkpoints; 0 means all")
    p.add_argument("--decoding", choices=["auto", "greedy", "qwen-thinking"], default="auto",
                   help="auto: Qwen thinking sampling when thinking is enabled, otherwise greedy")
    p.add_argument("--seed", type=int, default=42, help="same request seeds across checkpoints")
    p.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    p.add_argument("--store-text", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument(
        "--parser-path",
        help="benchmark parser override; default: Lulu's bundled lulu/benchmark_parser.py",
    )
    p.add_argument("--lcb-repo")
    p.add_argument("--lcb-python", default=sys.executable)
    p.add_argument("--lcb-processes", type=int, default=16)
    p.add_argument("--dry-run", action="store_true", help="print resolved plan without importing model dependencies or writing outputs")
    p.add_argument("--worker-plan", help=argparse.SUPPRESS)
    p.add_argument("--shard-id", type=int, default=0, help=argparse.SUPPRESS)
    return p


def main():
    args = argument_parser().parse_args()
    if args.worker_plan:
        run_worker(json.loads(Path(args.worker_plan).read_text()), args.shard_id)
        return
    plan = build_plan(args)
    if args.dry_run:
        print(json.dumps(plan, indent=2))
    else:
        run_suite(plan)


if __name__ == "__main__":
    main()
