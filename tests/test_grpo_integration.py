from pathlib import Path
import os
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_grpo_plan_is_cwd_independent(tmp_path):
    for dependency in ("torch", "transformers", "peft", "pyarrow", "sympy", "regex", "word2number", "latex2sympy2_extended"):
        pytest.importorskip(dependency)
    output = tmp_path / "grpo-output"
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_grpo.py"),
            "--output-dir",
            str(output),
            "--plan-only",
        ],
        cwd=tmp_path,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        text=True,
        capture_output=True,
        check=True,
    )
    assert '"unique_prompts": 1536' in result.stdout
    assert '"steps": 32' in result.stdout
    assert '"reward_sources"' in result.stdout
    assert not output.exists()


def test_grpo_cluster_files_have_no_local_download_paths():
    paths = [
        ROOT / "scripts" / "run_grpo.py",
        ROOT / "runs" / "train_grpo.sh",
        ROOT / "runs" / "grpo_8gpu.slurm",
        ROOT / "runs" / "eval_grpo.sh",
    ]
    assert all(path.is_file() for path in paths)
    assert all("/Users/" not in path.read_text() for path in paths)
