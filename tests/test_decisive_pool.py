import importlib.util
import json
from pathlib import Path

import pytest

from lulu.data import normalize_record

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/prepare_decisive_pool.py"
SPEC = importlib.util.spec_from_file_location("prepare_decisive_pool", SCRIPT)
pool = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pool)


class Tokenizer:
    chat_template = "fake thinking template"

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs["enable_thinking"] is True
        return " ".join(m["content"] for m in messages) + " <assistant>"

    def encode(self, text, **kwargs):
        return list(range(len(text.split())))

    def get_vocab(self):
        return {"fake": 0}


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def sources(tmp_path):
    train = [normalize_record({"question": f"Compute question {i}", "answer": i}) for i in range(16)]
    heldout = [normalize_record({"question": "A held out question", "answer": 42})]
    train_path, heldout_path = tmp_path / "train.jsonl", tmp_path / "dev.jsonl"
    write_rows(train_path, train)
    write_rows(heldout_path, heldout)
    return train_path, heldout_path, train, heldout


def prepare(sources, output, **kwargs):
    return pool.prepare_pool(sources[0], sources[1], output, tokenizer=Tokenizer(), size=8, **kwargs)


def test_fixed_pool_reproducible_and_source_order_independent(sources, tmp_path):
    first = prepare(sources, tmp_path / "first")
    second = prepare(sources, tmp_path / "first")
    assert first == second
    selected = json.loads((tmp_path / "first/question_ids.json").read_text())
    assert len(set(selected)) == first["pool_size"] == 8
    assert first["heldout"]["overlap_by_normalized_question"] == 0
    assert first["length_validation"]["excluded_questions"] == 0
    write_rows(sources[0], list(reversed(sources[2])))
    reordered = prepare(sources, tmp_path / "reordered")
    assert (tmp_path / "first/train.jsonl").read_bytes() == (tmp_path / "reordered/train.jsonl").read_bytes()
    assert reordered["source_train"]["sha256"] != first["source_train"]["sha256"]
    prepare(sources, tmp_path / "other_seed", seed=43)
    assert (tmp_path / "other_seed/question_ids.json").read_bytes() != (tmp_path / "first/question_ids.json").read_bytes()


def test_existing_pool_is_immutable_and_corruption_detected(sources, tmp_path):
    prepare(sources, tmp_path / "fixed")
    with pytest.raises(FileExistsError, match="Immutable pool"):
        prepare(sources, tmp_path / "fixed", seed=100)
    (tmp_path / "fixed/train.jsonl").write_text("corrupted")
    with pytest.raises(FileExistsError, match="Immutable pool"):
        prepare(sources, tmp_path / "fixed")


def test_overlap_detected_using_content_even_when_ids_differ(sources, tmp_path):
    duplicate = dict(sources[2][0], id="misleading-id")
    write_rows(sources[1], [duplicate])
    with pytest.raises(ValueError, match="overlaps held-out"):
        prepare(sources, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_duplicate_traceability_ids_rejected(sources, tmp_path):
    sources[2][1]["id"] = sources[2][0]["id"]
    write_rows(sources[0], sources[2])
    with pytest.raises(ValueError, match="duplicate question IDs"):
        prepare(sources, tmp_path / "out")


def test_token_limits_validate_hindsight_without_filtering(sources, tmp_path):
    with pytest.raises(ValueError, match="hindsight prompt.*pool was not filtered"):
        prepare(sources, tmp_path / "out", max_prompt_tokens=10)
    assert not (tmp_path / "out").exists()
    with pytest.raises(ValueError, match="response budget exceeds"):
        prepare(sources, tmp_path / "out", max_new_tokens=8192, max_sequence_tokens=8193)
    assert not (tmp_path / "out").exists()


def test_exclusions_applied_before_selection_and_preserved(sources, tmp_path):
    excluded = [{"id": row["id"], "reason": "exact benchmark question overlap"} for row in sources[2][:8]]
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"normalizer_policy": "test canonical whitespace", "excluded_questions": excluded}))
    result = prepare(sources, tmp_path / "out", exclusions_json=audit)
    selected = set(json.loads((tmp_path / "out/question_ids.json").read_text()))
    assert selected == {row["id"] for row in sources[2][8:]}
    assert result["eligible_questions"] == result["excluded_questions"] == 8
    assert result["exclusion_audit"]["audit"]["excluded_questions"] == excluded
    assert result["exclusion_audit"]["sha256"] == pool.digest(audit.read_bytes())


def test_unknown_exclusions_and_insufficient_pool_fail(sources, tmp_path):
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"normalizer_policy": "test", "excluded_questions": [{"id": "unknown", "reason": "overlap"}]}))
    with pytest.raises(ValueError, match="IDs absent"):
        prepare(sources, tmp_path / "out", exclusions_json=audit)
    with pytest.raises(ValueError, match="eligible source contains only"):
        pool.prepare_pool(sources[0], sources[1], tmp_path / "out", tokenizer=Tokenizer(), size=17)


def test_exclusion_audit_must_match_source(sources, tmp_path):
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"normalizer_policy": "test", "excluded_questions": [],
                                "source_train": {"sha256": "not-the-source"}}))
    with pytest.raises(ValueError, match="SHA256 does not match"):
        prepare(sources, tmp_path / "out", exclusions_json=audit)
