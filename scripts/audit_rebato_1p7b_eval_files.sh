#!/usr/bin/env bash
set -euo pipefail

# Read-only audit of Qwen3-1.7B evaluation outputs. The only writes are reports
# under AUDIT_OUT. No source experiment file is copied, moved, or deleted.

FENG_J="${FENG_J:-/pfss/mlde/workspaces/mlde_wsp_Model_Distil/Feng_J}"
ECO_ROOT="${ECO_ROOT:-/pfss/mlde/workspaces/mlde_wsp_Eco_Inference/Rona_Soraka/LuLu_outputs/experiments}"
EXPERIMENTS="${EXPERIMENTS:-$FENG_J/Lulu_outputs/experiments}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
AUDIT_OUT="${AUDIT_OUT:-$FENG_J/rebato_audit_$STAMP}"

mkdir -p "$AUDIT_OUT"

PATHS_TSV="$AUDIT_OUT/candidate_paths.tsv"
FILES_TSV="$AUDIT_OUT/files.tsv"
OVERSIZE_TSV="$AUDIT_OUT/github_oversize.tsv"
CODING_TSV="$AUDIT_OUT/coding_candidates.tsv"
PASS16_TSV="$AUDIT_OUT/pass16_validation.tsv"
SUMMARY_TXT="$AUDIT_OUT/README.txt"

printf 'status\tbytes\thuman_size\tpath\n' > "$PATHS_TSV"
printf 'bytes\thuman_size\tpath\n' > "$FILES_TSV"
printf 'github_class\tbytes\thuman_size\tpath\n' > "$OVERSIZE_TSV"
printf 'kind\tbytes\thuman_size\tpath\n' > "$CODING_TSV"

human_size() {
  if command -v numfmt >/dev/null 2>&1; then
    numfmt --to=iec-i --suffix=B "$1"
  else
    awk -v bytes="$1" 'BEGIN {
      split("B KiB MiB GiB TiB", units, " "); i=1; value=bytes+0;
      while (value >= 1024 && i < 5) { value /= 1024; i++ }
      if (i == 1) printf "%.0f%s\n", value, units[i];
      else printf "%.2f%s\n", value, units[i];
    }'
  fi
}

record_path() {
  local path="$1"
  local status bytes human
  if [[ -e "$path" ]]; then
    status="EXISTS"
    bytes="$(du -sb --apparent-size "$path" | awk '{print $1}')"
    human="$(human_size "$bytes")"
  else
    status="MISSING"
    bytes=0
    human="0B"
  fi
  printf '%s\t%s\t%s\t%s\n' "$status" "$bytes" "$human" "$path" >> "$PATHS_TSV"
}

PASS16_ROOT="$EXPERIMENTS/rq1_qwen3_1p7b_pass16_32k_five_bench_s42"

CANDIDATES=(
  "$PASS16_ROOT"
  "$PASS16_ROOT/reverse_6gpu"
  "$PASS16_ROOT/PASS16_SPLIT_MANIFEST.json"
  "$EXPERIMENTS/no_reasoning_control_ref_qwen3_1p7b_pool2048_b256_r4_s42/evaluation_base_control_ref_round4_32k_max512"
  "$EXPERIMENTS/vanilla_opd_qwen3_1p7b_pool2048_b256_r4_s42/evaluation_vanilla_opd_round4_32k_max512"
  "$EXPERIMENTS/opsd_qwen3_1p7b_pool2048_b256_r4_s42/evaluation_opsd_round4_32k_max512"
  "$EXPERIMENTS/rq2_teacher_sweep_qwen3_1p7b_pool2048_b256_r4_s42_6gpu/evaluation_rq23_students_round4_32k_five_bench_s42"
  "$ECO_ROOT/ren_balanced_matched_b256_c0p5_8k_r4_s42_20260918"
  "$ECO_ROOT/ren_balanced_abs_full_qwen1p7_teacher32b_pool2048_dev256_r12_s42"
)

for path in "${CANDIDATES[@]}"; do
  record_path "$path"
done

# Inventory all regular files inside the known candidate roots. De-duplicate
# nested roots such as reverse_6gpu by sorting on the absolute path.
tmp_files="$AUDIT_OUT/.files.unsorted.tsv"
: > "$tmp_files"
for root in "${CANDIDATES[@]}"; do
  [[ -d "$root" ]] || continue
  find -L "$root" -type f -printf '%s\t%p\n' >> "$tmp_files"
done

sort -t $'\t' -k2,2 -u "$tmp_files" | while IFS=$'\t' read -r bytes path; do
  [[ -n "$path" ]] || continue
  human="$(human_size "$bytes")"
  printf '%s\t%s\t%s\n' "$bytes" "$human" "$path" >> "$FILES_TSV"
  if (( bytes > 100 * 1024 * 1024 )); then
    printf 'BLOCKED_BY_REGULAR_GIT\t%s\t%s\t%s\n' "$bytes" "$human" "$path" >> "$OVERSIZE_TSV"
  elif (( bytes > 50 * 1024 * 1024 )); then
    printf 'GITHUB_WARNING\t%s\t%s\t%s\n' "$bytes" "$human" "$path" >> "$OVERSIZE_TSV"
  fi
done
rm -f "$tmp_files"

# Locate coding evaluation evidence without assuming its experiment name.
# Checkpoint/model/cache trees are excluded so this remains an output scan.
if [[ -d "$EXPERIMENTS" ]]; then
  find "$EXPERIMENTS" \
    \( -type d \( -name checkpoints -o -name model -o -name .engine_cache -o -name wandb -o -name code \) -prune \) -o \
    \( -type d -iname '*livecodebench*' -print \) -o \
    \( -type f \( \
        -iname '*livecodebench*' -o \
        -iname '*codegeneration_output*' -o \
        -iname '*humaneval*' -o \
        -iname '*mbpp*' -o \
        -iname '*codeforces*' \
      \) -print \) \
    | sort -u \
    | while IFS= read -r path; do
        [[ -e "$path" ]] || continue
        if [[ -d "$path" ]]; then
          kind="directory"
          bytes="$(du -sb --apparent-size "$path" | awk '{print $1}')"
        else
          kind="file"
          bytes="$(stat -c '%s' "$path")"
        fi
        printf '%s\t%s\t%s\t%s\n' "$kind" "$bytes" "$(human_size "$bytes")" "$path" >> "$CODING_TSV"
      done
fi

# Validate the documented non-coding Pass@16 split. This checks the filesystem
# and row coverage only; it does not calculate Pass@16 accuracy.
python - "$PASS16_ROOT" "$PASS16_TSV" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
output = Path(sys.argv[2])

algorithms = {
    "base": "base",
    "no_reasoning": "control_ref_round4",
    "vanilla_opd": "vanilla_opd_round4",
    "opsd": "opsd_round4",
}
benchmarks = {
    "math500": 500,
    "aime25": 30,
    "olympiadbench": 512,
    "mmlu_pro": 512,
    "gpqa_diamond": 198,
}

def sample_path(index: int, seed: int) -> Path:
    name = f"sample_{index:02d}_seed_{seed}"
    return root / name if seed <= 52 else root / "reverse_6gpu" / name

def nonempty_jsonl_rows(paths: list[Path]) -> tuple[int, int]:
    rows = 0
    malformed = 0
    for path in paths:
        with path.open(errors="replace") as stream:
            for line in stream:
                if not line.strip():
                    continue
                rows += 1
                try:
                    json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
    return rows, malformed

columns = [
    "seed", "sample_index", "sample_dir", "sample_exists", "eval_plan",
    "root_summary", "algorithm", "internal_name", "benchmark", "expected_rows",
    "shard_count", "actual_rows", "malformed_json_rows", "benchmark_summary", "status",
]

with output.open("w", encoding="utf-8") as out:
    out.write("\t".join(columns) + "\n")
    for seed in range(42, 58):
        index = seed - 42
        sample = sample_path(index, seed)
        sample_exists = sample.is_dir()
        eval_plan = (sample / "eval_plan.json").is_file()
        root_summary = (sample / "summary.json").is_file()
        for algorithm, internal in algorithms.items():
            for benchmark, expected in benchmarks.items():
                directory = sample / internal / benchmark
                shards = sorted(directory.glob("shard-*.jsonl")) if directory.is_dir() else []
                rows, malformed = nonempty_jsonl_rows(shards)
                benchmark_summary = (directory / "summary.json").is_file()
                checks = [
                    sample_exists,
                    eval_plan,
                    root_summary,
                    bool(shards),
                    rows == expected,
                    malformed == 0,
                    benchmark_summary,
                ]
                status = "OK" if all(checks) else "INCOMPLETE"
                values = [
                    seed, index, sample, sample_exists, eval_plan, root_summary,
                    algorithm, internal, benchmark, expected, len(shards), rows,
                    malformed, benchmark_summary, status,
                ]
                out.write("\t".join(map(str, values)) + "\n")
PY

candidate_total="$(awk -F '\t' 'NR>1 && $1=="EXISTS" {s+=$2} END {print s+0}' "$PATHS_TSV")"
file_count="$(awk 'END {print NR-1}' "$FILES_TSV")"
blocked_count="$(awk -F '\t' 'NR>1 && $1=="BLOCKED_BY_REGULAR_GIT" {n++} END {print n+0}' "$OVERSIZE_TSV")"
warning_count="$(awk -F '\t' 'NR>1 && $1=="GITHUB_WARNING" {n++} END {print n+0}' "$OVERSIZE_TSV")"
coding_count="$(awk 'END {print NR-1}' "$CODING_TSV")"
pass16_ok="$(awk -F '\t' 'NR>1 && $15=="OK" {n++} END {print n+0}' "$PASS16_TSV")"
pass16_total="$(awk 'END {print NR-1}' "$PASS16_TSV")"

cat > "$SUMMARY_TXT" <<EOF
Rebato Qwen3-1.7B evaluation audit
generated_utc=$STAMP
experiments_root=$EXPERIMENTS
audit_out=$AUDIT_OUT

known_candidate_bytes=$candidate_total
known_candidate_human=$(human_size "$candidate_total")
inventoried_files=$file_count
files_over_100MiB=$blocked_count
files_between_50_and_100MiB=$warning_count
coding_candidates=$coding_count
pass16_cells_ok=$pass16_ok
pass16_cells_total=$pass16_total

Reports:
  $PATHS_TSV
  $FILES_TSV
  $OVERSIZE_TSV
  $CODING_TSV
  $PASS16_TSV
EOF

cat "$SUMMARY_TXT"
echo
echo "=== candidate paths ==="
column -t -s $'\t' "$PATHS_TSV" || cat "$PATHS_TSV"
echo
echo "=== coding candidates ==="
column -t -s $'\t' "$CODING_TSV" || cat "$CODING_TSV"
echo
echo "=== incomplete Pass@16 cells (maximum 80 shown) ==="
awk -F '\t' 'NR==1 || $15!="OK"' "$PASS16_TSV" | head -n 81 | column -t -s $'\t' || true
echo
echo "AUDIT_COMPLETE=$AUDIT_OUT"
