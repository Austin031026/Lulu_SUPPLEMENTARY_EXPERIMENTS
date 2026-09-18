#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
import pyarrow.parquet as pq


def load_generations(root: Path):
    out = {}
    for p in sorted(root.glob("shard-*.jsonl")):
        for line in p.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            x = json.loads(line)
            out[int(x["prompt_index"])] = x
    return out


def extract_code(text: str) -> str:
    # Match the standard LCB/Lighteval expectation: an extracted program, not the
    # full reasoning trace. Prefer the final fenced block.
    blocks = re.findall(r"```(?:python|py)?\s*\n?(.*?)```", str(text), flags=re.I | re.S)
    if blocks:
        return blocks[-1].strip()
    return str(text).strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--generation-dir", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()

    rows = pq.read_table(Path(a.data).expanduser().resolve()).to_pylist()
    generations = load_generations(Path(a.generation_dir).expanduser().resolve())
    if len(generations) != len(rows):
        raise RuntimeError(f"generation coverage {len(generations)} != data rows {len(rows)}")

    outputs = []
    for i, row in enumerate(rows):
        x = generations[i]
        text = x.get("text")
        if text is None:
            raise RuntimeError(
                "LiveCodeBench generation JSONL has no `text`. "
                "Run eval_streaming_selector.sh with STORE_TEXT=1."
            )
        info = row.get("extra_info") or {}
        qid = str(info.get("question_id") or info.get("benchmark_id") or "")
        if not qid:
            raise RuntimeError(f"LCB row {i} missing question_id")
        outputs.append({"question_id": qid, "code_list": [extract_code(text)]})

    path = Path(a.output).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(outputs, indent=2) + "\n")
    print(f"[lcb-export] {len(outputs)} -> {path}")


if __name__ == "__main__":
    main()
