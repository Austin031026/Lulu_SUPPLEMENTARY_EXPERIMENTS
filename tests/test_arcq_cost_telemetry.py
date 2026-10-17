import json
from pathlib import Path
import sys


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_with_cost_telemetry import git_provenance, summarize
from summarize_arcq_exp3_cost import read_run


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_gpu_telemetry_summary_uses_elapsed_gpu_hours_and_peak_memory():
    samples = [
        {"unix": 10.0, "gpus": [
            {"index": "0", "uuid": "a", "name": "A100", "memory_total_mib": 81920,
             "memory_used_mib": 100, "utilization_percent": 10, "power_watts": 100},
            {"index": "1", "uuid": "b", "name": "A100", "memory_total_mib": 81920,
             "memory_used_mib": 200, "utilization_percent": 20, "power_watts": 120},
        ]},
        {"unix": 3610.0, "gpus": [
            {"index": "0", "uuid": "a", "name": "A100", "memory_total_mib": 81920,
             "memory_used_mib": 300, "utilization_percent": 90, "power_watts": 200},
            {"index": "1", "uuid": "b", "name": "A100", "memory_total_mib": 81920,
             "memory_used_mib": 400, "utilization_percent": 80, "power_watts": 220},
        ]},
    ]
    result = summarize(samples, label="opd", command=["train"], gpus=["0", "1"],
                       started_unix=10.0, finished_unix=3610.0, exit_code=0)
    assert result["gpu_hours"] == 2.0
    assert result["peak_device_memory_used_mib"] == 400
    assert result["status"] == "complete"


def test_git_provenance_is_captured_for_repository():
    result = git_provenance(str(Path(__file__).resolve().parents[1]))
    assert result["commit"]
    assert result["root"]


def test_exp3_reader_combines_training_telemetry_and_accuracy(tmp_path):
    root = tmp_path / "opd"
    write_json(root / "train/run_config.json", {})
    write_json(root / "train/metrics/round_0000.json", [{
        "response_tokens": 100, "pipeline_seconds": 5, "update_seconds": 2, "round_seconds": 8,
        "teacher_seconds": 3, "hindsight_seconds": 1,
        "student_peak_memory_allocated_mib": 1000,
        "student_peak_memory_reserved_mib": 1200,
    }])
    write_json(root / "train/metrics/round_0001.json", [{
        "response_tokens": 200, "pipeline_seconds": 7, "update_seconds": 4, "round_seconds": 12,
        "teacher_seconds": 5, "hindsight_seconds": 2,
        "student_peak_memory_allocated_mib": 1400,
        "student_peak_memory_reserved_mib": 1800,
    }])
    write_json(root / "telemetry/summary.json", {
        "status": "complete", "exit_code": 0, "wall_seconds": 30,
        "allocated_gpu_count": 8, "gpu_hours": 30 * 8 / 3600,
        "peak_device_memory_used_mib": 70000, "sampled_gpu_energy_wh": 12,
    })
    benchmark_values = {
        "math500": 0.8,
        "aime25": 0.2,
        "olympiadbench": 0.5,
        "mmlu_pro": 0.6,
        "gpqa_diamond": 0.4,
    }
    write_json(root / "evaluation/summary.json", {"models": {
        "base": {
            "macro_accuracy": 0.1,
            "benchmarks": {name: {"accuracy": 0.1} for name in benchmark_values},
        },
        "final": {
            "macro_accuracy": 0.5,
            "benchmarks": {name: {"accuracy": value} for name, value in benchmark_values.items()},
        },
    }})
    result = read_run("vanilla_opd", root)
    assert result["generated_tokens"] == 300
    assert result["rollout_and_scoring_seconds"] == 12
    assert result["update_seconds"] == 6
    assert result["round_seconds"] == 20
    assert result["gpu_count"] == 8
    assert result["macro_accuracy"] == 0.5
    assert result["evaluation_model"] == "final"
    assert result["benchmark_accuracy"]["math500"] == 0.8
    assert result["student_peak_memory_allocated_mib"] == 1400
    assert result["student_peak_memory_reserved_mib"] == 1800
