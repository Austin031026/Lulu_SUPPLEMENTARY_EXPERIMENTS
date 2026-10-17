#!/usr/bin/env python3
"""Run one command while recording reproducible GPU cost telemetry.

This wrapper does not modify the wrapped training process or its CUDA
environment.  It samples device-level NVML data through ``nvidia-smi`` and
writes both the raw time series and an atomic summary.  The process exit code
is preserved, so launchers can treat it exactly like the original command.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import signal
import socket
import subprocess
import sys
import time


QUERY_FIELDS = (
    "index",
    "uuid",
    "name",
    "memory.total",
    "memory.used",
    "utilization.gpu",
    "power.draw",
)


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_provenance(cwd: str) -> dict:
    def run(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", cwd, *arguments], check=True, capture_output=True, text=True,
        ).stdout.strip()

    try:
        root = run("rev-parse", "--show-toplevel")
        return {
            "root": root,
            "commit": run("rev-parse", "HEAD"),
            "branch": run("branch", "--show-current"),
            "dirty": bool(run("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"root": None, "commit": None, "branch": None, "dirty": None}


def parse_devices(value: str) -> list[str]:
    devices = [item.strip() for item in value.split(",") if item.strip()]
    if not devices or len(devices) != len(set(devices)):
        raise ValueError("--gpus must contain distinct GPU indices")
    return devices


def query_gpus(devices: list[str]) -> list[dict]:
    command = [
        "nvidia-smi",
        "--query-gpu=" + ",".join(QUERY_FIELDS),
        "--format=csv,noheader,nounits",
        "-i",
        ",".join(devices),
    ]
    output = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    rows = []
    for line in output.splitlines():
        values = [item.strip() for item in line.split(",")]
        if len(values) != len(QUERY_FIELDS):
            raise RuntimeError(f"Unexpected nvidia-smi row: {line!r}")
        index, uuid, name, memory_total, memory_used, utilization, power = values
        rows.append(
            {
                "index": index,
                "uuid": uuid,
                "name": name,
                "memory_total_mib": float(memory_total),
                "memory_used_mib": float(memory_used),
                "utilization_percent": float(utilization),
                "power_watts": None if power in {"[N/A]", "N/A"} else float(power),
            }
        )
    if {row["index"] for row in rows} != set(devices):
        raise RuntimeError("nvidia-smi did not return every requested GPU")
    return rows


def summarize(samples: list[dict], *, label: str, command: list[str], gpus: list[str],
              started_unix: float, finished_unix: float, exit_code: int) -> dict:
    by_gpu = {gpu: [] for gpu in gpus}
    for sample in samples:
        for row in sample["gpus"]:
            by_gpu[row["index"]].append((sample["unix"], row))

    gpu_rows = {}
    total_energy_wh = 0.0
    for gpu, observations in by_gpu.items():
        if not observations:
            continue
        rows = [row for _, row in observations]
        energy_wh = 0.0
        for (left_time, left), (right_time, right) in zip(observations, observations[1:]):
            if left["power_watts"] is not None and right["power_watts"] is not None:
                energy_wh += (right_time - left_time) * (left["power_watts"] + right["power_watts"]) / 2 / 3600
        total_energy_wh += energy_wh
        gpu_rows[gpu] = {
            "uuid": rows[0]["uuid"],
            "name": rows[0]["name"],
            "memory_total_mib": rows[0]["memory_total_mib"],
            "peak_device_memory_used_mib": max(row["memory_used_mib"] for row in rows),
            "mean_device_memory_used_mib": sum(row["memory_used_mib"] for row in rows) / len(rows),
            "peak_utilization_percent": max(row["utilization_percent"] for row in rows),
            "mean_utilization_percent": sum(row["utilization_percent"] for row in rows) / len(rows),
            "peak_power_watts": max((row["power_watts"] for row in rows if row["power_watts"] is not None), default=None),
            "estimated_energy_wh": energy_wh if len(observations) > 1 else None,
            "samples": len(rows),
        }

    wall_seconds = finished_unix - started_unix
    return {
        "schema_version": 1,
        "label": label,
        "status": "complete" if exit_code == 0 else "failed",
        "command": command,
        "exit_code": exit_code,
        "started_unix": started_unix,
        "finished_unix": finished_unix,
        "wall_seconds": wall_seconds,
        "allocated_gpu_count": len(gpus),
        "gpu_hours": wall_seconds * len(gpus) / 3600,
        "sample_count": len(samples),
        "sampled_gpu_energy_wh": total_energy_wh if len(samples) > 1 else None,
        "gpus": gpu_rows,
        "peak_device_memory_used_mib": max(
            (row["peak_device_memory_used_mib"] for row in gpu_rows.values()), default=None
        ),
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
        },
        "note": (
            "Peak memory is total device memory used as sampled by nvidia-smi; "
            "it includes PyTorch, vLLM, NCCL, and other processes on the allocated device."
        ),
    }


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    return args


def main() -> int:
    args = arguments()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    samples_path = output / "gpu_samples.jsonl"
    summary_path = output / "summary.json"
    manifest_path = output / "manifest.json"
    if samples_path.exists() or summary_path.exists() or manifest_path.exists():
        raise FileExistsError(f"Telemetry output must be empty: {output}")

    devices = parse_devices(args.gpus)
    initial = query_gpus(devices)
    started_unix = time.time()
    manifest = {
        "schema_version": 1,
        "label": args.label,
        "command": args.command,
        "gpus": devices,
        "poll_seconds": args.poll_seconds,
        "started_utc": utc_now(),
        "cwd": os.getcwd(),
        "git": git_provenance(os.getcwd()),
        "initial_gpu_state": initial,
    }
    atomic_json(manifest_path, manifest)

    child = subprocess.Popen(args.command)

    def forward(signum, _frame):
        if child.poll() is None:
            child.send_signal(signum)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, forward)

    samples = []
    with samples_path.open("x", encoding="utf-8") as stream:
        while True:
            sample = {"unix": time.time(), "utc": utc_now(), "gpus": query_gpus(devices)}
            samples.append(sample)
            stream.write(json.dumps(sample, ensure_ascii=False) + "\n")
            stream.flush()
            exit_code = child.poll()
            if exit_code is not None:
                break
            time.sleep(args.poll_seconds)

    finished_unix = time.time()
    summary = summarize(
        samples,
        label=args.label,
        command=args.command,
        gpus=devices,
        started_unix=started_unix,
        finished_unix=finished_unix,
        exit_code=exit_code,
    )
    summary["started_utc"] = manifest["started_utc"]
    summary["finished_utc"] = utc_now()
    atomic_json(summary_path, summary)
    print(json.dumps({"telemetry": str(summary_path), "exit_code": exit_code}), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
