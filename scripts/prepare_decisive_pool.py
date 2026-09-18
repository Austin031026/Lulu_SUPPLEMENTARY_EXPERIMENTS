"""Freeze a reproducible DAPO training prompt pool without touching held-out data.

Selection uses seeded hashes of normalized question content. Length checks reject
an invalid configuration; they never silently change the sampled question pool.
Only the cached tokenizer is loaded, so this command needs no GPU or downloads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lulu.data import build_prompt_views, load_prepared_jsonl, question_key

WORKSPACE = Path(__file__).resolve().parents[2]
POLICY = "sha256(lulu-decisive-pool-v1:{seed}:{normalized_question_sha256}), ascending"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def validate_lengths(rows, tokenizer, *, max_prompt_tokens: int, max_new_tokens: int,
                     max_sequence_tokens: int) -> dict[str, Any]:
    lengths = {"causal": [], "hindsight": []}
    for row in rows:
        views = build_prompt_views(tokenizer, row["messages"], row["gold_answer"], enable_thinking=True)
        for view in lengths:
            size = len(views[f"{view}_prompt_ids"])
            if size > max_prompt_tokens:
                raise ValueError(f"{view} prompt {row['id']} has {size} tokens, exceeding --max-prompt-tokens={max_prompt_tokens}; pool was not filtered")
            if size + max_new_tokens > max_sequence_tokens:
                raise ValueError(f"{view} prompt {row['id']} plus response budget exceeds --max-sequence-tokens={max_sequence_tokens}; pool was not filtered")
            lengths[view].append(size)
    stats = {}
    for view, values in lengths.items():
        ordered = sorted(values)
        stats[view] = {"min": min(values), "median": ordered[len(ordered) // 2],
                       "p95": ordered[min(len(ordered) - 1, (95 * len(ordered) + 99) // 100 - 1)],
                       "max": max(values)}
    vocab = getattr(tokenizer, "get_vocab", lambda: {})()
    return {"enable_thinking": True, "checked_questions": len(rows), "excluded_questions": 0,
            "max_prompt_tokens": max_prompt_tokens, "max_new_tokens": max_new_tokens,
            "max_sequence_tokens": max_sequence_tokens, "prompt_tokens": stats,
            "tokenizer_vocab_sha256": digest(json_bytes(vocab)),
            "chat_template_sha256": digest(json_bytes(getattr(tokenizer, "chat_template", None)))}


def prepare_pool(train_data: str | Path, heldout_data: str | Path, output_dir: str | Path,
                 *, tokenizer, tokenizer_name: str = "Qwen/Qwen3-1.7B", size: int = 2048,
                 seed: int = 42, max_prompt_tokens: int = 4096, max_new_tokens: int = 8192,
                 max_sequence_tokens: int = 16384, exclusions_json: str | Path | None = None) -> dict[str, Any]:
    if min(size, max_prompt_tokens, max_new_tokens, max_sequence_tokens) <= 0:
        raise ValueError("Pool size and token budgets must be positive")
    train_path, heldout_path, output = (Path(p).expanduser().resolve() for p in (train_data, heldout_data, output_dir))
    train_raw, heldout_raw = train_path.read_bytes(), heldout_path.read_bytes()
    train, heldout = load_prepared_jsonl(train_path), load_prepared_jsonl(heldout_path)
    train_ids = [row["id"] for row in train]
    if len(set(train_ids)) != len(train_ids):
        raise ValueError("Prepared training data has duplicate question IDs")
    heldout_ids = {row["id"] for row in heldout}
    heldout_keys = {question_key(row["messages"]) for row in heldout}
    if set(train_ids) & heldout_ids or any(question_key(row["messages"]) in heldout_keys for row in train):
        raise ValueError("Training source overlaps held-out questions (IDs or normalized content)")
    exclusion_manifest = None
    excluded_ids = set()
    if exclusions_json is not None:
        exclusion_path = Path(exclusions_json).expanduser().resolve()
        exclusion_data = exclusion_path.read_bytes()
        exclusion_manifest = json.loads(exclusion_data)
        entries = exclusion_manifest.get("excluded_questions")
        if not exclusion_manifest.get("normalizer_policy") or not isinstance(entries, list):
            raise ValueError("Exclusion audit requires normalizer_policy and excluded_questions list")
        audited_source_sha = exclusion_manifest.get("source_train", {}).get("sha256")
        if audited_source_sha is not None and audited_source_sha != digest(train_raw):
            raise ValueError("Exclusion audit source_train SHA256 does not match training source")
        for entry in entries:
            if not isinstance(entry, dict) or not entry.get("id") or not entry.get("reason"):
                raise ValueError("Every exclusion requires id and reason")
            excluded_ids.add(entry["id"])
        unknown = excluded_ids - set(train_ids)
        if unknown:
            raise ValueError(f"Exclusion audit contains IDs absent from training source: {sorted(unknown)}")
        exclusion_manifest = {"path": str(exclusion_path), "sha256": digest(exclusion_data),
                              "audit": exclusion_manifest}
    eligible = [row for row in train if row["id"] not in excluded_ids]
    if len(eligible) < size:
        raise ValueError(f"Requested {size} distinct questions but eligible source contains only {len(eligible)}")
    selected = sorted(eligible, key=lambda row: hashlib.sha256(
        f"lulu-decisive-pool-v1:{seed}:{question_key(row['messages'])}".encode("utf-8")).digest())[:size]
    validation = validate_lengths(selected, tokenizer, max_prompt_tokens=max_prompt_tokens,
                                  max_new_tokens=max_new_tokens, max_sequence_tokens=max_sequence_tokens)
    pool_data = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in selected).encode("utf-8")
    ids = [row["id"] for row in selected]
    ids_data = json_bytes(ids)
    manifest = {
        "schema_version": 1, "dataset": "BytedTsinghua-SIA/DAPO-Math-17k",
        "seed": seed, "pool_size": size, "selection_policy": POLICY,
        "eligible_questions": len(eligible), "excluded_questions": len(excluded_ids),
        "exclusion_audit": exclusion_manifest,
        "source_train": {"path": str(train_path), "sha256": digest(train_raw), "questions": len(train)},
        "heldout": {"path": str(heldout_path), "sha256": digest(heldout_raw), "questions": len(heldout),
                    "overlap_by_id": 0, "overlap_by_normalized_question": 0},
        "tokenizer": tokenizer_name, "length_validation": validation,
        "outputs": {"train": {"path": str(output / "train.jsonl"), "sha256": digest(pool_data)},
                    "question_ids": {"path": str(output / "question_ids.json"), "sha256": digest(ids_data)}},
    }
    source_manifest = train_path.parent / "manifest.json"
    if source_manifest.is_file():
        manifest["source_manifest"] = {"path": str(source_manifest), "sha256": digest(source_manifest.read_bytes())}
    # Detect a concurrent change instead of claiming provenance for different bytes.
    if train_path.read_bytes() != train_raw or heldout_path.read_bytes() != heldout_raw:
        raise RuntimeError("Source data changed during pool preparation; no pool was written")
    files = {"train.jsonl": pool_data, "question_ids.json": ids_data, "manifest.json": json_bytes(manifest)}
    if output.exists():
        if not output.is_dir() or any(not (output / name).is_file() or (output / name).read_bytes() != data
                                      for name, data in files.items()):
            raise FileExistsError(f"Immutable pool already exists with different contents/configuration: {output}; choose another output directory")
        return manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for name, data in files.items():
            (temporary / name).write_bytes(data)
        temporary.rename(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", default=str(WORKSPACE / "Soraka_rlrl/data/lulu_dapo/train.jsonl"))
    parser.add_argument("--heldout-data", default=str(WORKSPACE / "Soraka_rlrl/data/lulu_dapo/dev.jsonl"))
    parser.add_argument("--output-dir", default=str(WORKSPACE / "LuLu_outputs/data/dapo_pool2048_s42"))
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--cache-dir", help="Local Hugging Face hub cache directory")
    parser.add_argument("--exclusions-json", help="Exact benchmark-overlap audit: normalizer_policy and excluded_questions with id/reason")
    parser.add_argument("--max-prompt-tokens", type=int, default=4096)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--max-sequence-tokens", type=int, default=16384)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, cache_dir=args.cache_dir, local_files_only=True)
    manifest = prepare_pool(args.train_data, args.heldout_data, args.output_dir, tokenizer=tokenizer,
                            tokenizer_name=args.tokenizer, size=args.size, seed=args.seed,
                            max_prompt_tokens=args.max_prompt_tokens, max_new_tokens=args.max_new_tokens,
                            max_sequence_tokens=args.max_sequence_tokens, exclusions_json=args.exclusions_json)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
