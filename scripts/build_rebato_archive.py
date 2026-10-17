#!/usr/bin/env python3
"""Copy completed Qwen3 evaluation directories into a simple Rebato archive.

This script performs directory-level archival only. It does not concatenate
JSONL files, recalculate metrics, rename algorithms, or modify source files.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    feng = Path("/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J")
    experiments = feng / "Lulu_outputs/experiments"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("scaffold", "non-coding", "coding", "all"), default="all"
    )
    parser.add_argument("--target-root", type=Path, default=feng / "Rebato")
    parser.add_argument(
        "--non-coding-root",
        type=Path,
        default=experiments / "rq1_qwen3_1p7b_pass16_32k_five_bench_s42",
    )
    parser.add_argument(
        "--coding-root",
        type=Path,
        default=(
            experiments
            / "rq1_livecodebench_v6new175_qwen3_1p7b_pass16_4b_pass1_s42"
            / "qwen3_1p7b_pass16"
        ),
    )
    return parser.parse_args()


def sample_name(seed: int) -> str:
    return f"sample_{seed - 42:02d}_seed_{seed}"


def non_coding_sample(root: Path, seed: int) -> Path:
    name = sample_name(seed)
    direct = root / name
    reverse = root / "reverse_6gpu" / name
    matches = [path for path in (direct, reverse) if path.is_dir()]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected exactly one source for {name}; found: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def copy_file(source: Path, destination: Path) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or destination.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"destination conflicts with source: {destination}")
        return "existing"

    temporary = destination.with_name(destination.name + f".partial.{os.getpid()}")
    print(
        f"[copy] bytes={source.stat().st_size} source={source} destination={destination}",
        flush=True,
    )
    try:
        with source.open("rb") as reader, temporary.open("wb") as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
        shutil.copystat(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return "copied"


def copy_tree(source: Path, destination: Path) -> tuple[int, int, int]:
    if not source.is_dir():
        raise FileNotFoundError(source)
    copied = existing = total_bytes = 0
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise RuntimeError(f"unsupported symlink in source: {path}")
        if not path.is_file():
            continue
        total_bytes += path.stat().st_size
        status = copy_file(path, destination / path.relative_to(source))
        copied += status == "copied"
        existing += status == "existing"
    return copied, existing, total_bytes


def write_tsv(path: Path, header: list[str], rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)
    os.replace(temporary, path)


def write_readmes(target: Path) -> None:
    qwen17 = target / "Qwen3-1.7B"
    qwen4 = target / "Qwen3-4B"
    qwen17.mkdir(parents=True, exist_ok=True)
    qwen4.mkdir(parents=True, exist_ok=True)
    (qwen17 / "README.md").write_text(
        """# Qwen3-1.7B rebuttal evaluation results

## What is stored here

These are the original **Pass@16** evaluation outputs for Qwen3-1.7B. The
archive is a directory-level copy: files are not concatenated, transformed or
rescored.

```text
Qwen3-1.7B/
  non_coding/pass16/
    sample_00_seed_42/
    ...
    sample_15_seed_57/
  coding/pass16/
    sample_00_seed_42/
    ...
    sample_15_seed_57/
```

The 16 non-coding samples originally stored in two source locations are placed
side by side in the same `non_coding/pass16` directory. The Coding directory is
copied directly from the completed Qwen3-1.7B LiveCodeBench result directory.

## Algorithms and benchmarks

The original internal algorithm directory names are preserved:

- `base`
- `control_ref_round4` (no-reasoning/control-reference baseline)
- `vanilla_opd_round4`
- `opsd_round4`

Non-coding benchmarks are `math500`, `aime25`, `olympiadbench`, `mmlu_pro`
and `gpqa_diamond`. Coding uses LiveCodeBench v6 new-175 and preserves its raw
generation files, scored rows and official evaluator JSON artifacts.

## How to use the files

Each `sample_XX_seed_YY` is one of the 16 independent generations. Read the
original `eval_plan.json` and `summary.json` in that sample before analysis.
Benchmark directories contain the original shard JSONL files and summaries.

Rows commonly preserve the prompt identity, prediction/response, ground truth,
reward, token counts, stop reason, generation seed and prompt hash. Exact keys
must be checked in the actual JSONL because benchmark evaluators can add their
own fields. The retained data supports Pass@1 per seed, Pass@16 by taking a
per-problem OR over all 16 samples, accuracy, response-length and truncation
analysis, and paired comparisons between algorithms. LiveCodeBench `scored`
rows should be used for official coding correctness.

Pass@16 is not the average of 16 Pass@1 values. Match the same problem across
samples using its prompt index/hash, verify ground-truth alignment, and count a
problem as solved if any of the 16 corresponding rows is correct.

`SOURCE_PATHS_*.tsv` records where every copied directory came from, and
`ARCHIVE_FILES.tsv` lists every archived file and its byte size.
""",
        encoding="utf-8",
    )
    placeholder = qwen4 / "README.md"
    if not placeholder.exists():
        placeholder.write_text(
            "# Qwen3-4B rebuttal evaluation results\n\n"
            "This directory is reserved for the separately audited Qwen3-4B results.\n",
            encoding="utf-8",
        )


def copy_non_coding(args: argparse.Namespace) -> None:
    destination_root = args.target_root / "Qwen3-1.7B/non_coding/pass16"
    rows: list[list[object]] = []
    for seed in range(42, 58):
        name = sample_name(seed)
        source = non_coding_sample(args.non_coding_root, seed)
        destination = destination_root / name
        print(f"[non-coding] sample={name} source={source}", flush=True)
        copied, existing, total_bytes = copy_tree(source, destination)
        rows.append([seed, name, str(source), str(destination), copied, existing, total_bytes])
        print(
            f"[non-coding] completed sample={name} copied={copied} "
            f"existing={existing} bytes={total_bytes}",
            flush=True,
        )
    write_tsv(
        args.target_root / "Qwen3-1.7B/SOURCE_PATHS_NON_CODING.tsv",
        ["seed", "sample", "source", "destination", "files_copied",
         "files_already_present", "bytes"],
        rows,
    )


def copy_coding(args: argparse.Namespace) -> None:
    destination = args.target_root / "Qwen3-1.7B/coding/pass16"
    print(f"[coding] copying original directory {args.coding_root}", flush=True)
    copied, existing, total_bytes = copy_tree(args.coding_root, destination)
    write_tsv(
        args.target_root / "Qwen3-1.7B/SOURCE_PATHS_CODING.tsv",
        ["source", "destination", "files_copied", "files_already_present", "bytes"],
        [[str(args.coding_root), str(destination), copied, existing, total_bytes]],
    )
    print(
        f"[coding] completed copied={copied} existing={existing} bytes={total_bytes}",
        flush=True,
    )


def inventory(target: Path) -> None:
    rows = []
    inventory_path = target / "ARCHIVE_FILES.tsv"
    for path in sorted(target.rglob("*")):
        if path.is_file() and path != inventory_path:
            rows.append([str(path.relative_to(target)), path.stat().st_size])
    write_tsv(inventory_path, ["relative_path", "bytes"], rows)


def main() -> None:
    args = parse_args()
    args.target_root = args.target_root.expanduser().resolve()
    args.non_coding_root = args.non_coding_root.expanduser().resolve()
    args.coding_root = args.coding_root.expanduser().resolve()
    print(f"target={args.target_root}", flush=True)
    write_readmes(args.target_root)
    if args.phase in ("non-coding", "all"):
        copy_non_coding(args)
    if args.phase in ("coding", "all"):
        copy_coding(args)
    inventory(args.target_root)
    print(f"REBATO_COPY_COMPLETE={args.target_root}", flush=True)


if __name__ == "__main__":
    main()
