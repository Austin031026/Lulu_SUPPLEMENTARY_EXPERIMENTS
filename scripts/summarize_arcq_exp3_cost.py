#!/usr/bin/env python3
"""Combine ArcQ Experiment 3 training, GPU telemetry, and evaluation metrics."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED_LABELS = ("vanilla_opd", "hind", "opsd")
REQUIRED_BENCHMARKS = (
    "math500", "aime25", "olympiadbench", "mmlu_pro", "gpqa_diamond",
)


def load_json(path: Path):
    return json.loads(path.read_text())


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("--run must use LABEL=RUN_ROOT")
    label, raw = value.split("=", 1)
    if label not in EXPECTED_LABELS:
        raise ValueError(f"unknown label {label!r}; expected one of {EXPECTED_LABELS}")
    return label, Path(raw).expanduser().resolve()


def metric_rows(train: Path) -> list[dict]:
    rows = []
    for path in sorted((train / "metrics").glob("round_*.json")):
        value = load_json(path)
        rows.extend(value if isinstance(value, list) else [value])
    return rows


def select_evaluation_model(label: str, summary: dict, requested: str | None = None) -> tuple[str, dict]:
    """Select exactly one trained model from an evaluate_lulu summary.

    Evaluation summaries may contain both ``base`` and the trained checkpoint.
    Silently walking the whole JSON can therefore mix two models that use the
    same benchmark names.  Selection is explicit when requested and otherwise
    only succeeds when it is unambiguous.
    """
    models = summary.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("evaluation summary has no non-empty 'models' mapping")
    if requested is not None:
        if requested not in models:
            raise ValueError(f"evaluation model {requested!r} is absent; available={tuple(models)}")
        return requested, models[requested]

    preferred = {
        "vanilla_opd": ("vanilla_opd", "opd", "final"),
        "hind": ("hind", "ren", "ren_balanced", "final"),
        "opsd": ("opsd", "final"),
    }[label]
    for name in preferred:
        if name in models:
            return name, models[name]
    non_base = [(name, value) for name, value in models.items() if name != "base"]
    if len(non_base) == 1:
        return non_base[0]
    raise ValueError(
        f"cannot infer trained evaluation model for {label}; available={tuple(models)}; "
        "pass --evaluation-model LABEL=MODEL"
    )


def read_run(label: str, root: Path, evaluation_model: str | None = None) -> dict:
    train = root / "train"
    telemetry_path = root / "telemetry" / "summary.json"
    evaluation_path = root / "evaluation" / "summary.json"
    required = [train / "run_config.json", telemetry_path, evaluation_path]
    missing = [str(path) for path in required if not path.is_file()]
    metrics = metric_rows(train)
    if not metrics:
        missing.append(str(train / "metrics/round_*.json"))
    if missing:
        raise FileNotFoundError("missing required Experiment 3 outputs:\n" + "\n".join(missing))

    telemetry = load_json(telemetry_path)
    if telemetry.get("status") != "complete" or telemetry.get("exit_code") != 0:
        raise ValueError(f"telemetry marks {label} as incomplete")
    evaluation = load_json(evaluation_path)
    selected_name, selected = select_evaluation_model(label, evaluation, evaluation_model)
    benchmarks = selected.get("benchmarks")
    if not isinstance(benchmarks, dict):
        raise ValueError(f"evaluation model {selected_name!r} has no benchmark mapping")
    missing_benchmarks = [name for name in REQUIRED_BENCHMARKS if name not in benchmarks]
    if missing_benchmarks:
        raise ValueError(f"evaluation model {selected_name!r} is missing {missing_benchmarks}")
    accuracies = {}
    for name in REQUIRED_BENCHMARKS:
        accuracy = benchmarks[name].get("accuracy")
        if not isinstance(accuracy, (int, float)):
            raise ValueError(f"{selected_name}/{name} has no numerical accuracy")
        accuracies[name] = float(accuracy)
    macro = selected.get("macro_accuracy")
    recomputed_macro = sum(accuracies.values()) / len(accuracies)
    if macro is None:
        macro = recomputed_macro
    elif abs(float(macro) - recomputed_macro) > 1e-12:
        raise ValueError(
            f"{selected_name} macro accuracy mismatch: summary={macro}, recomputed={recomputed_macro}"
        )

    generated_tokens = sum(int(row.get("response_tokens", 0)) for row in metrics)
    rollout_scoring_seconds = sum(float(row.get("pipeline_seconds", 0)) for row in metrics)
    update_seconds = sum(float(row.get("update_seconds", row.get("seconds", 0))) for row in metrics)
    round_seconds = sum(float(row.get("round_seconds", 0)) for row in metrics)
    teacher_seconds = sum(float(row.get("teacher_seconds", 0)) for row in metrics)
    hindsight_seconds = sum(float(row.get("hindsight_seconds", 0)) for row in metrics)
    allocator_allocated = max(
        (float(row["student_peak_memory_allocated_mib"]) for row in metrics
         if row.get("student_peak_memory_allocated_mib") is not None),
        default=None,
    )
    allocator_reserved = max(
        (float(row["student_peak_memory_reserved_mib"]) for row in metrics
         if row.get("student_peak_memory_reserved_mib") is not None),
        default=None,
    )
    return {
        "label": label,
        "run_root": str(root),
        "completed_rounds": len(metrics),
        "generated_tokens": generated_tokens,
        "rollout_and_scoring_seconds": rollout_scoring_seconds,
        "update_seconds": update_seconds,
        "round_seconds": round_seconds,
        "teacher_service_seconds": teacher_seconds,
        "hindsight_service_seconds": hindsight_seconds,
        "wall_seconds": float(telemetry["wall_seconds"]),
        "gpu_count": int(telemetry["allocated_gpu_count"]),
        "gpu_hours": float(telemetry["gpu_hours"]),
        "peak_device_memory_used_mib": telemetry.get("peak_device_memory_used_mib"),
        "student_peak_memory_allocated_mib": allocator_allocated,
        "student_peak_memory_reserved_mib": allocator_reserved,
        "sampled_gpu_energy_wh": telemetry.get("sampled_gpu_energy_wh"),
        "evaluation_model": selected_name,
        "macro_accuracy": macro,
        "benchmark_accuracy": accuracies,
        "telemetry": str(telemetry_path),
        "evaluation_summary": str(evaluation_path),
    }


def write_tsv(path: Path, rows: list[dict]) -> None:
    fields = [
        "label", "completed_rounds", "generated_tokens",
        "rollout_and_scoring_seconds", "update_seconds", "round_seconds",
        "wall_seconds", "gpu_count", "gpu_hours",
        "peak_device_memory_used_mib", "sampled_gpu_energy_wh", "macro_accuracy",
        "student_peak_memory_allocated_mib", "student_peak_memory_reserved_mib",
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict]) -> None:
    lines = [
        "# ArcQ Experiment 3 cost summary",
        "",
        "`rollout_and_scoring_seconds` is the measured collection and target-scoring pipeline, not pure generation time.",
        "Peak memory is sampled total device memory, including PyTorch, vLLM, NCCL, and other processes on the allocated GPU.",
        "",
        "| Algorithm | GPU-hours | Wall hours | Generated tokens | Rollout + scoring hours | Update hours | Device peak MiB | Student allocated MiB | Student reserved MiB | Macro accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        macro = "" if row["macro_accuracy"] is None else f'{100*row["macro_accuracy"]:.2f}%'
        peak = "" if row["peak_device_memory_used_mib"] is None else f'{row["peak_device_memory_used_mib"]:.0f}'
        allocated = "" if row["student_peak_memory_allocated_mib"] is None else f'{row["student_peak_memory_allocated_mib"]:.0f}'
        reserved = "" if row["student_peak_memory_reserved_mib"] is None else f'{row["student_peak_memory_reserved_mib"]:.0f}'
        lines.append(
            f'| {row["label"]} | {row["gpu_hours"]:.3f} | {row["wall_seconds"]/3600:.3f} | '
            f'{row["generated_tokens"]} | {row["rollout_and_scoring_seconds"]/3600:.3f} | '
            f'{row["update_seconds"]/3600:.3f} | {peak} | {allocated} | {reserved} | {macro} |'
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="LABEL=RUN_ROOT")
    parser.add_argument(
        "--evaluation-model", action="append", default=[], help="optional LABEL=MODEL selection",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    parsed = dict(parse_run(value) for value in args.run)
    if set(parsed) != set(EXPECTED_LABELS):
        raise ValueError(f"require exactly {EXPECTED_LABELS}; received {tuple(parsed)}")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected_models = {}
    for value in args.evaluation_model:
        if "=" not in value:
            raise ValueError("--evaluation-model must use LABEL=MODEL")
        label, model = value.split("=", 1)
        if label not in EXPECTED_LABELS or not model:
            raise ValueError(f"invalid --evaluation-model {value!r}")
        selected_models[label] = model
    rows = [read_run(label, parsed[label], selected_models.get(label)) for label in EXPECTED_LABELS]
    payload = {
        "schema_version": 1,
        "scope": "same-hardware ArcQ Experiment 3 cost comparison",
        "algorithms": rows,
    }
    (output / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    write_tsv(output / "summary.tsv", rows)
    write_markdown(output / "REPORT.md", rows)
    print(json.dumps({"status": "complete", "output_dir": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
