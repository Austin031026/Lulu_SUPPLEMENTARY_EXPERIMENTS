"""Small, pure helpers for paper-RQ evaluation summaries.

The evaluation runner already writes per-model/per-benchmark accuracies and
example counts.  These helpers derive the grouped metrics used repeatedly in
RQ1/RQ2/RQ3 tables without changing the underlying benchmark scores.
"""
from __future__ import annotations

MATH = ("math500", "aime25", "olympiadbench")
OOD = ("mmlu_pro", "gpqa_diamond")
EXTERNAL = MATH + OOD


def mean(values):
    values = [float(x) for x in values if x is not None]
    return sum(values) / len(values) if values else None


def _benchmark(model_summary, name):
    return (model_summary.get("benchmarks") or {}).get(name)


def grouped_metrics(model_summary):
    """Return raw benchmark accuracies plus grouped paper metrics.

    ``external_micro`` weights each benchmark by its actual number of examples;
    ``math_avg``/``ood_avg``/``external_macro`` are equal-benchmark averages.
    Missing benchmarks stay missing rather than silently becoming zero.
    """
    benches = model_summary.get("benchmarks") or {}
    out = {}
    for name in (*EXTERNAL, "dapo_dev128"):
        entry = benches.get(name)
        out[name] = None if entry is None else entry.get("accuracy")
    out["math_avg"] = mean(out[b] for b in MATH)
    out["ood_avg"] = mean(out[b] for b in OOD)
    out["external_macro"] = mean(out[b] for b in EXTERNAL)
    numerator = denominator = 0.0
    for name in EXTERNAL:
        entry = benches.get(name)
        if entry is None or entry.get("accuracy") is None:
            continue
        n = entry.get("examples", entry.get("scored_examples"))
        if n is None:
            continue
        numerator += float(entry["accuracy"]) * int(n)
        denominator += int(n)
    out["external_micro"] = numerator / denominator if denominator else None
    out["external_examples"] = int(denominator)
    return out


def with_deltas(values, base):
    row = dict(values)
    for key, value in list(values.items()):
        if key == "external_examples" or value is None or base.get(key) is None:
            continue
        row[key + "_delta"] = float(value) - float(base[key])
    return row
