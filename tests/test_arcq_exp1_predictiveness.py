import numpy as np
import pytest
import torch

from lulu.exp1_predictiveness import (
    assign_groups, cluster_bootstrap, deterministic_subset, exact_pre_post_kl,
    trajectory_identity, validate_score_rows, validate_trajectory,
)


def test_configurable_deterministic_question_subset():
    rows = [{"id": f"q{i}"} for i in range(20)]
    small = deterministic_subset(rows, 5, 42)
    large = deterministic_subset(list(reversed(rows)), 10, 42)
    assert [row["id"] for row in small] == [row["id"] for row in large[:5]]
    assert len(deterministic_subset(rows, 20, 42)) == 20
    with pytest.raises(ValueError, match="num-questions"):
        deterministic_subset(rows, 21, 42)


def test_subset_accepts_question_id_and_normalizes_to_id():
    selected = deterministic_subset(
        [{"question_id": "q0"}, {"question_id": "q1"}], 2, seed=42
    )
    assert {row["id"] for row in selected} == {"q0", "q1"}


def test_frozen_trajectory_identity_and_context_validation():
    row = dict(question_id="q", causal_prompt_ids=[1, 2], hindsight_prompt_ids=[1, 3, 2],
               response_ids=[4, 5, 6], positions=[0, 2], gold_answer="6")
    row["trajectory_id"] = trajectory_identity("q", row["causal_prompt_ids"], row["response_ids"])
    validate_trajectory(row, max_sequence_tokens=8)
    changed = dict(row, response_ids=[4, 5, 7])
    with pytest.raises(ValueError, match="identity"):
        validate_trajectory(changed, max_sequence_tokens=8)


def test_exact_full_vocabulary_pre_post_identities():
    torch.manual_seed(7)
    logits = [torch.randn(6, 11, dtype=torch.float64) for _ in range(4)]
    values = exact_pre_post_kl(*logits)
    torch.testing.assert_close(values["hindsight_score"],
                               values["initial_causal_kl"] - values["initial_hindsight_kl"])
    torch.testing.assert_close(values["actual_improvement"],
                               values["initial_causal_kl"] - values["post_causal_kl"])
    assert all(value.shape == (6,) and torch.isfinite(value).all() for value in values.values())


def synthetic_rows():
    rows = []
    # Every question contributes all four S regimes at two distinct D0 levels.
    for question in range(12):
        for level, d0 in enumerate((1.0, 2.0)):
            for offset, score in enumerate((-0.1, 0.1, 0.3, 0.8)):
                improvement = 0.02 + 0.4 * score + question * 1e-4
                rows.append({
                    "question_id": f"q{question}", "trajectory_id": f"t{question}",
                    "position": level * 10 + offset, "initial_causal_kl": d0,
                    "initial_hindsight_kl": d0 - score, "post_causal_kl": d0 - improvement,
                    "hindsight_score": score, "actual_improvement": improvement,
                    "teacher_entropy": 1.0,
                })
    return rows


def test_initial_kl_control_and_question_cluster_bootstrap():
    grouped = assign_groups(synthetic_rows(), initial_kl_bins=2)
    assert {row["hindsight_group"] for row in grouped} == {"nonpositive", "low", "medium", "high"}
    result = cluster_bootstrap(grouped, replicates=100, seed=9)
    assert result["high"]["mean_improvement"] > result["low"]["mean_improvement"]
    assert result["high_minus_low"]["estimate"] > 0
    assert result["high_minus_low"]["bootstrap_valid_replicates"] == 100


def test_score_validation_rejects_token_duplication():
    rows = synthetic_rows()[:1]
    validate_score_rows(rows)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_score_rows(rows + rows)
