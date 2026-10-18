"""Pure helpers for ArcQ Experiment 1 fixed-prefix predictiveness.

The GPU entry points live in ``scripts/``.  This module deliberately keeps the
selection, KL identities, joining, stratification, and question-cluster
bootstrap testable without a GPU.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch


SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected an object at {path}:{line_number}")
                rows.append(value)
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Mapping]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), ensure_ascii=False, separators=(",", ":")) + "\n")
    temporary.replace(path)


def deterministic_subset(rows: Sequence[Mapping], count: int, seed: int) -> list[dict]:
    """Select a batch-order-independent subset using stable row identities."""
    if count < 1 or count > len(rows):
        raise ValueError(f"--num-questions must be in [1, {len(rows)}], received {count}")
    identities = []
    for row in rows:
        identity = str(row.get("id", ""))
        if not identity:
            raise ValueError("Every diagnostic question must have a nonempty id")
        key = hashlib.sha256(f"arcq-exp1:{seed}:{identity}".encode()).digest()
        identities.append((key, identity, dict(row)))
    if len({identity for _, identity, _ in identities}) != len(identities):
        raise ValueError("Diagnostic question ids must be unique")
    return [row for _, _, row in sorted(identities)[:count]]


def trajectory_identity(question_id: str, causal_prompt_ids: Sequence[int], response_ids: Sequence[int]) -> str:
    payload = json.dumps(
        {"question_id": str(question_id), "causal_prompt_ids": list(causal_prompt_ids),
         "response_ids": list(response_ids)},
        sort_keys=True, separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def validate_trajectory(row: Mapping, *, max_sequence_tokens: int) -> None:
    required = {"question_id", "trajectory_id", "causal_prompt_ids", "hindsight_prompt_ids",
                "response_ids", "positions", "gold_answer"}
    missing = required - set(row)
    if missing:
        raise ValueError(f"Trajectory is missing fields: {sorted(missing)}")
    causal = list(row["causal_prompt_ids"])
    hindsight = list(row["hindsight_prompt_ids"])
    response = list(row["response_ids"])
    positions = list(row["positions"])
    if not causal or not hindsight or not response:
        raise ValueError("Trajectory prompt/response token sequences must be nonempty")
    if positions != sorted(set(positions)) or any(p < 0 or p >= len(response) for p in positions):
        raise ValueError("Trajectory positions must be unique increasing response indices")
    if max(len(causal), len(hindsight)) + len(response) > max_sequence_tokens:
        raise ValueError("Frozen trajectory exceeds the scoring context budget")
    expected = trajectory_identity(str(row["question_id"]), causal, response)
    if row["trajectory_id"] != expected:
        raise ValueError("Trajectory identity does not match its frozen token IDs")


@torch.no_grad()
def exact_pre_post_kl(
    teacher_logits: torch.Tensor,
    pre_causal_logits: torch.Tensor,
    pre_hindsight_logits: torch.Tensor,
    post_causal_logits: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute the exact full-vocabulary quantities required by Experiment 1."""
    tensors = (teacher_logits, pre_causal_logits, pre_hindsight_logits, post_causal_logits)
    if teacher_logits.ndim != 2 or any(value.shape != teacher_logits.shape for value in tensors):
        raise ValueError("Expected four matching [positions, vocabulary] logit tensors")
    if any(not value.is_floating_point() for value in tensors):
        raise ValueError("KL inputs must be floating tensors")
    dtype = torch.float64 if any(value.dtype == torch.float64 for value in tensors) else torch.float32
    teacher, causal0, hindsight0, causal1 = [value.detach().to(dtype).log_softmax(-1) for value in tensors]
    q = teacher.exp()
    d0 = (q * (teacher - causal0)).sum(-1).clamp_min(0)
    d0h = (q * (teacher - hindsight0)).sum(-1).clamp_min(0)
    d1 = (q * (teacher - causal1)).sum(-1).clamp_min(0)
    result = {
        "initial_causal_kl": d0,
        "initial_hindsight_kl": d0h,
        "post_causal_kl": d1,
        "hindsight_score": d0 - d0h,
        "actual_improvement": d0 - d1,
        "teacher_entropy": -(q * teacher).sum(-1),
    }
    if any(not bool(value.isfinite().all()) for value in result.values()):
        raise FloatingPointError("Non-finite Experiment 1 score")
    return result


SCORE_KEYS = ("initial_causal_kl", "initial_hindsight_kl", "post_causal_kl",
              "hindsight_score", "actual_improvement", "teacher_entropy")


def validate_score_rows(rows: Sequence[Mapping], *, tolerance: float = 2e-5) -> None:
    seen = set()
    for row in rows:
        identity = (str(row["trajectory_id"]), int(row["position"]))
        if identity in seen:
            raise ValueError(f"Duplicate scored position: {identity}")
        seen.add(identity)
        values = {key: float(row[key]) for key in SCORE_KEYS}
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError(f"Non-finite score at {identity}")
        if min(values[key] for key in ("initial_causal_kl", "initial_hindsight_kl", "post_causal_kl")) < -tolerance:
            raise ValueError(f"Negative KL beyond tolerance at {identity}")
        if abs(values["hindsight_score"] - (values["initial_causal_kl"] - values["initial_hindsight_kl"])) > tolerance:
            raise ValueError(f"Hindsight score identity failed at {identity}")
        if abs(values["actual_improvement"] - (values["initial_causal_kl"] - values["post_causal_kl"])) > tolerance:
            raise ValueError(f"Improvement identity failed at {identity}")


def _quantile_edges(values: np.ndarray, bins: int) -> np.ndarray:
    if bins < 2:
        raise ValueError("At least two initial-KL strata are required")
    edges = np.quantile(values, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def assign_groups(rows: Sequence[Mapping], *, initial_kl_bins: int = 10) -> list[dict]:
    """Control D0 first, then split positive S into tertiles within each stratum."""
    validate_score_rows(rows)
    if not rows:
        raise ValueError("No scored positions")
    # Analysis commonly receives millions of Parquet rows. Reuse mutable dicts
    # returned by ``to_pylist`` instead of doubling their memory footprint.
    output = [row if isinstance(row, dict) else dict(row) for row in rows]
    d0 = np.asarray([float(row["initial_causal_kl"]) for row in output])
    strata = np.digitize(d0, _quantile_edges(d0, initial_kl_bins)[1:-1], right=True)
    for row, stratum in zip(output, strata.tolist()):
        row["initial_kl_bin"] = int(stratum)
    for stratum in range(initial_kl_bins):
        indices = [i for i, row in enumerate(output) if row["initial_kl_bin"] == stratum]
        positive = np.asarray([float(output[i]["hindsight_score"]) for i in indices
                               if float(output[i]["hindsight_score"]) > 0])
        cuts = np.quantile(positive, [1 / 3, 2 / 3]) if len(positive) else np.asarray([0.0, 0.0])
        for index in indices:
            score = float(output[index]["hindsight_score"])
            output[index]["hindsight_group"] = (
                "nonpositive" if score <= 0 else "low" if score <= cuts[0]
                else "medium" if score <= cuts[1] else "high"
            )
    return output


GROUPS = ("nonpositive", "low", "medium", "high")


def question_balanced_means(rows: Sequence[Mapping]) -> dict[str, dict[str, float | int | None]]:
    """Average tokens within question, then give every question equal weight."""
    by_question: dict[tuple[str, str], list[float]] = defaultdict(list)
    token_counts = defaultdict(int)
    for row in rows:
        group = str(row["hindsight_group"])
        if group not in GROUPS:
            raise ValueError(f"Unknown hindsight group: {group}")
        by_question[(group, str(row["question_id"]))].append(float(row["actual_improvement"]))
        token_counts[group] += 1
    result = {}
    for group in GROUPS:
        values = [float(np.mean(value)) for (label, _), value in by_question.items() if label == group]
        result[group] = {
            "mean_improvement": float(np.mean(values)) if values else None,
            "questions": len(values), "tokens": int(token_counts[group]),
        }
    return result


def controlled_means(rows: Sequence[Mapping]) -> dict[str, dict[str, float | int | None]]:
    """Question-balance within D0 strata, then give every nonempty stratum equal weight."""
    cell_sums = defaultdict(float)
    cell_counts = defaultdict(int)
    token_counts = defaultdict(int)
    questions = defaultdict(set)
    for row in rows:
        group = str(row["hindsight_group"])
        stratum = int(row["initial_kl_bin"])
        question = str(row["question_id"])
        key = (group, stratum, question)
        cell_sums[key] += float(row["actual_improvement"])
        cell_counts[key] += 1
        token_counts[group] += 1
        questions[group].add(question)
    result = {}
    for group in GROUPS:
        strata = defaultdict(list)
        for (label, stratum, question), total in cell_sums.items():
            if label == group:
                strata[stratum].append(total / cell_counts[(label, stratum, question)])
        stratum_means = {stratum: float(np.mean(values)) for stratum, values in strata.items()}
        result[group] = {
            "mean_improvement": float(np.mean(list(stratum_means.values()))) if stratum_means else None,
            "questions": len(questions[group]), "tokens": int(token_counts[group]),
            "nonempty_initial_kl_bins": len(stratum_means),
            "stratum_means": {str(key): value for key, value in sorted(stratum_means.items())},
        }
    return result


def cluster_bootstrap(
    rows: Sequence[Mapping], *, replicates: int = 5000, seed: int = 42,
) -> dict[str, dict[str, float | int | list[float] | None]]:
    """Question-cluster bootstrap; token rows are never sampled independently."""
    if replicates < 1:
        raise ValueError("Bootstrap replicates must be positive")
    questions = sorted({str(row["question_id"]) for row in rows})
    if not questions:
        raise ValueError("No questions to bootstrap")
    point = controlled_means(rows)
    question_index = {question: index for index, question in enumerate(questions)}
    strata = sorted({int(row["initial_kl_bin"]) for row in rows})
    stratum_index = {stratum: index for index, stratum in enumerate(strata)}
    sums = {group: np.zeros((len(strata), len(questions)), dtype=np.float64) for group in GROUPS}
    counts = {group: np.zeros((len(strata), len(questions)), dtype=np.int64) for group in GROUPS}
    for row in rows:
        group = str(row["hindsight_group"])
        i = stratum_index[int(row["initial_kl_bin"])]
        j = question_index[str(row["question_id"])]
        sums[group][i, j] += float(row["actual_improvement"])
        counts[group][i, j] += 1
    cell_values = {group: np.divide(sums[group], counts[group],
                                    out=np.zeros_like(sums[group]), where=counts[group] > 0)
                   for group in GROUPS}
    cell_present = {group: counts[group] > 0 for group in GROUPS}
    # A multinomial row is exactly one question-cluster bootstrap replicate.
    rng = np.random.default_rng(seed)
    cluster_counts = rng.multinomial(len(questions), np.full(len(questions), 1 / len(questions)),
                                     size=replicates).astype(np.float64)
    draws = {}
    for group in GROUPS:
        numerator = cluster_counts @ cell_values[group].T
        denominator = cluster_counts @ cell_present[group].T
        stratum_draws = np.divide(numerator, denominator,
                                  out=np.full_like(numerator, np.nan), where=denominator > 0)
        with np.errstate(invalid="ignore"):
            draws[group] = np.nanmean(stratum_draws, axis=1)
    contrasts = draws["high"] - draws["low"]
    result = {}
    for group in GROUPS:
        samples = np.asarray(draws[group], dtype=np.float64)
        samples = samples[np.isfinite(samples)]
        result[group] = dict(point[group])
        result[group]["ci95"] = (np.quantile(samples, [0.025, 0.975]).tolist()
                                  if len(samples) else None)
        result[group]["bootstrap_valid_replicates"] = int(len(samples))
    contrast = np.asarray(contrasts, dtype=np.float64)
    contrast = contrast[np.isfinite(contrast)]
    low, high = point["low"]["mean_improvement"], point["high"]["mean_improvement"]
    result["high_minus_low"] = {
        "estimate": None if low is None or high is None else float(high) - float(low),
        "ci95": np.quantile(contrast, [0.025, 0.975]).tolist() if len(contrast) else None,
        "bootstrap_valid_replicates": int(len(contrast)),
    }
    return result
