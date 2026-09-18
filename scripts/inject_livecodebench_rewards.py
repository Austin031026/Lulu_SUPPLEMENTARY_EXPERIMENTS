#!/usr/bin/env python3
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pyarrow.parquet as pq


def graded_map(path: Path):
    items = json.loads(path.read_text())
    out = {}
    for x in items:
        qid = str(x.get("question_id", ""))
        graded = x.get("graded_list") or []
        if not qid or not graded:
            raise RuntimeError(f"Malformed LCB eval row: keys={list(x)}")
        out[qid] = float(bool(graded[0]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--generation-dir", required=True)
    p.add_argument("--lcb-eval-all", required=True)
    p.add_argument("--output-dir", required=True)
    a = p.parse_args()

    data = pq.read_table(Path(a.data).expanduser().resolve()).to_pylist()
    qid_by_idx = {}
    for i, row in enumerate(data):
        info = row.get("extra_info") or {}
        qid_by_idx[i] = str(info.get("question_id") or info.get("benchmark_id") or "")
    grades = graded_map(Path(a.lcb_eval_all).expanduser().resolve())

    src = Path(a.generation_dir).expanduser().resolve()
    dst = Path(a.output_dir).expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)
    total = 0
    for shard in sorted(src.glob("shard-*.jsonl")):
        target = dst / shard.name
        with shard.open(errors="replace") as fi, target.open("w") as fo:
            for line in fi:
                if not line.strip():
                    continue
                x = json.loads(line)
                idx = int(x["prompt_index"])
                qid = qid_by_idx[idx]
                if qid not in grades:
                    raise RuntimeError(f"LCB grade missing question_id={qid}")
                x["reward"] = float(grades[qid])
                x["official_livecodebench_scored"] = True
                fo.write(json.dumps(x, ensure_ascii=False) + "\n")
                total += 1
    if total != len(data):
        raise RuntimeError(f"scored rows {total} != data rows {len(data)}")
    print(f"[lcb-inject] {total} -> {dst}")


if __name__ == "__main__":
    main()
