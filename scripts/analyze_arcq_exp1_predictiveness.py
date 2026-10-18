#!/usr/bin/env python3
"""Analyze whether pre-training hindsight scores predict post-OPD KL gains."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lulu.exp1_predictiveness import (
    GROUPS, SCHEMA_VERSION, assign_groups, atomic_json, cluster_bootstrap,
    load_jsonl, sha256_file, validate_score_rows, validate_trajectory,
)


LABELS = {"nonpositive": "S ≤ 0", "low": "Low S", "medium": "Medium S", "high": "High S"}


def svg_plot(path: Path, estimates: dict) -> None:
    width, height = 760, 480
    margin = dict(left=90, right=30, top=45, bottom=85)
    values = []
    for group in GROUPS:
        row = estimates[group]
        if row["mean_improvement"] is not None:
            values.append(float(row["mean_improvement"]))
        if row["ci95"]:
            values.extend(map(float, row["ci95"]))
    low, high = (min(values), max(values)) if values else (-1.0, 1.0)
    padding = max((high - low) * 0.15, 1e-4)
    low, high = min(low - padding, 0), max(high + padding, 0)
    plot_w = width - margin["left"] - margin["right"]
    plot_h = height - margin["top"] - margin["bottom"]
    x = lambda index: margin["left"] + plot_w * (index + 0.5) / len(GROUPS)
    y = lambda value: margin["top"] + (high - value) / (high - low) * plot_h
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<style>text{font-family:Arial,sans-serif;fill:#222}.axis{stroke:#333;stroke-width:1.5}.ci{stroke:#276FBF;stroke-width:3}.point{fill:#276FBF}</style>',
             f'<text x="{width/2}" y="25" text-anchor="middle" font-size="18">Hindsight score predicts post-OPD causal KL improvement</text>',
             f'<line class="axis" x1="{margin["left"]}" y1="{y(0)}" x2="{width-margin["right"]}" y2="{y(0)}" stroke-dasharray="5 4"/>',
             f'<line class="axis" x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{height-margin["bottom"]}"/>']
    for index, group in enumerate(GROUPS):
        row = estimates[group]
        xpos = x(index)
        lines.append(f'<text x="{xpos}" y="{height-50}" text-anchor="middle" font-size="14">{LABELS[group]}</text>')
        if row["mean_improvement"] is None:
            continue
        mean = float(row["mean_improvement"]); ci = row["ci95"] or [mean, mean]
        lines.extend([
            f'<line class="ci" x1="{xpos}" y1="{y(float(ci[0]))}" x2="{xpos}" y2="{y(float(ci[1]))}"/>',
            f'<line class="ci" x1="{xpos-8}" y1="{y(float(ci[0]))}" x2="{xpos+8}" y2="{y(float(ci[0]))}"/>',
            f'<line class="ci" x1="{xpos-8}" y1="{y(float(ci[1]))}" x2="{xpos+8}" y2="{y(float(ci[1]))}"/>',
            f'<circle class="point" cx="{xpos}" cy="{y(mean)}" r="6"/>',
        ])
    lines.extend([
        f'<text x="{width/2}" y="{height-15}" text-anchor="middle" font-size="15">Hindsight score group within initial-causal-KL strata</text>',
        f'<text x="20" y="{height/2}" text-anchor="middle" font-size="15" transform="rotate(-90 20 {height/2})">Mean KL improvement G = D0 − D1</text>',
        '</svg>'])
    path.write_text("\n".join(lines) + "\n")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--trajectories", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--initial-kl-bins", type=int, default=10)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    scores_path = Path(args.scores).resolve(strict=True)
    trajectories_path = Path(args.trajectories).resolve(strict=True)
    rows = pq.read_table(scores_path).to_pylist()
    validate_score_rows(rows)
    trajectories = load_jsonl(trajectories_path)
    expected = set()
    for trajectory in trajectories:
        validate_trajectory(trajectory, max_sequence_tokens=10**9)
        expected.update((trajectory["trajectory_id"], int(position)) for position in trajectory["positions"])
    actual = {(row["trajectory_id"], int(row["position"])) for row in rows}
    if expected != actual:
        raise ValueError(f"Frozen/scored position mismatch: missing={len(expected-actual)}, extra={len(actual-expected)}")
    grouped = assign_groups(rows, initial_kl_bins=args.initial_kl_bins)
    grouped_path = output / "grouped_position_scores.parquet"
    pq.write_table(pa.Table.from_pylist(grouped), grouped_path, compression="zstd")
    estimates = cluster_bootstrap(grouped, replicates=args.bootstrap_replicates, seed=args.seed)
    summary = {
        "schema_version": SCHEMA_VERSION, "status": "complete",
        "scientific_question": "Does pre-training hindsight score predict post-OPD causal KL improvement on frozen prefixes?",
        "definitions": {"D0": "KL(qT||p0)", "D0H": "KL(qT||p0H)",
                        "D1": "KL(qT||p1)", "S": "D0-D0H", "G": "D0-D1"},
        "control": {"initial_causal_kl_bins": args.initial_kl_bins,
                    "positive_hindsight_groups_per_bin": 3,
                    "nonpositive_group_separate": True,
                    "aggregation": "token mean within question x D0 stratum x S group; question mean within stratum; equal mean across nonempty D0 strata"},
        "uncertainty": {"unit": "question", "replicates": args.bootstrap_replicates,
                        "seed": args.seed, "token_independence_assumed": False},
        "trajectories": len(trajectories), "scored_positions": len(rows),
        "inputs": {"scores": str(scores_path), "scores_sha256": sha256_file(scores_path),
                   "trajectories": str(trajectories_path),
                   "trajectories_sha256": sha256_file(trajectories_path)},
        "groups": estimates,
    }
    atomic_json(output / "summary.json", summary)
    with (output / "controlled_groups.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["group", "mean_improvement", "ci95_low", "ci95_high", "questions",
                         "tokens", "nonempty_initial_kl_bins"])
        for group in GROUPS:
            row = estimates[group]; ci = row["ci95"] or [None, None]
            writer.writerow([group, row["mean_improvement"], *ci, row["questions"], row["tokens"],
                             row["nonempty_initial_kl_bins"]])
    svg_plot(output / "predictiveness.svg", estimates)
    contrast = estimates["high_minus_low"]
    report = [
        "# ArcQ Experiment 1: Hindsight-score predictiveness", "",
        "All quantities are evaluated on the same frozen Base-generated prefixes. No post-training trajectory is regenerated.", "",
        "| Hindsight group | Mean causal-KL improvement | 95% question-bootstrap CI | Questions | Tokens |",
        "|---|---:|---:|---:|---:|",
    ]
    for group in GROUPS:
        row = estimates[group]; ci = row["ci95"]
        ci_text = "NA" if ci is None else f"[{ci[0]:.6f}, {ci[1]:.6f}]"
        mean_text = "NA" if row["mean_improvement"] is None else f"{row['mean_improvement']:.6f}"
        report.append(f"| {LABELS[group]} | {mean_text} | {ci_text} | {row['questions']} | {row['tokens']} |")
    report += ["", f"High-minus-low improvement: {contrast['estimate']}; 95% CI: {contrast['ci95']}.", "",
               "Initial causal KL is controlled by equal-frequency strata before hindsight grouping. Confidence intervals resample complete questions, not tokens.", "",
               "![Predictiveness](predictiveness.svg)", ""]
    (output / "REPORT.md").write_text("\n".join(report))
    atomic_json(output / "validation.json", {
        "status": "PASS", "score_identities": True, "frozen_position_alignment": True,
        "question_cluster_bootstrap": True, "initial_kl_control": True,
        "post_generation_performed": False,
    })
    print(f"ARCQ_EXP1_ANALYSIS=PASS\nREPORT={output/'REPORT.md'}", flush=True)


if __name__ == "__main__":
    main()
