from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace

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
            "--prompt-mode",
            "pretokenized",
            "--global-batch-prompts",
            "48",
            "--global-epochs",
            "1.0",
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


def test_grpo_scheduler_does_not_require_materialized_prompts():
    source = (ROOT / "scripts" / "grpo" / "train_grpo.py").read_text()
    assert "plan(args, [None] * n)" not in source
    assert "schedule = prompt_schedule(n, scheduled, int(args.seed))" in source


def test_grpo_nemotron_raw_jsonl_uses_family_prompt_protocol(tmp_path, monkeypatch):
    for dependency in ("torch", "transformers", "peft", "pyarrow"):
        pytest.importorskip(dependency)
    from scripts.grpo import train_grpo
    from transformers import AutoConfig

    data = tmp_path / "train.jsonl"
    data.write_text(
        '{"messages":[{"role":"user","content":"Solve 1+1"}],'
        '"gold_answer":"2","data_source":"math"}\n'
    )

    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 2

        def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
            assert messages[0] == {"role": "system", "content": "detailed thinking on"}
            assert add_generation_prompt is True
            return [11, 12, 13] if tokenize else "detailed thinking on"

        def decode(self, ids, skip_special_tokens=False):
            return "detailed thinking on assistant"

    monkeypatch.setattr(
        AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(model_type="llama"),
    )
    monkeypatch.setattr(train_grpo, "load_tokenizer", lambda args: Tokenizer())

    rows = train_grpo.load_prompts(SimpleNamespace(
        train_data=str(data),
        prompt_mode="nemotron-thinking",
        model_family="nemotron",
        model="local-nemotron",
    ))
    assert len(rows) == 1
    assert rows[0].prompt_ids == [11, 12, 13]
    assert rows[0].ground_truth == "2"


def test_grpo_launchers_expose_model_family_and_memory_safe_controls():
    run_source = (ROOT / "scripts" / "run_grpo.py").read_text()
    train_source = (ROOT / "runs" / "train_grpo.sh").read_text()
    for marker in (
        "--model-family", "--optimizer-device", "--activation-offload",
        "--logprob-chunk-size",
    ):
        assert marker in run_source
    assert "GRPO_MODEL_FAMILY" in train_source
    assert "GRPO_OPTIMIZER_DEVICE" in train_source
    assert "GRPO_ACTIVATION_OFFLOAD" in train_source
