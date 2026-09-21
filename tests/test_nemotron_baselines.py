from scripts.run_nemotron_baselines import ARMS, arguments, commands


def test_only_three_requested_arms_are_planned():
    args = arguments([
        "--model", "/tmp/student", "--teacher-model", "/tmp/teacher",
        "--train-data", "/tmp/pool/train.jsonl",
        "--data-manifest", "/tmp/eval/manifest.json",
        "--output-dir", "/tmp/nemotron-baselines",
    ])
    plan = commands(args)
    assert ARMS == (("control_ref", "none"), ("vanilla_opd", "vanilla"), ("opsd", "opsd"))
    assert list(plan["train_commands"]) == ["control_ref", "vanilla_opd", "opsd"]
    assert plan["roles"] == {
        "student": ["0", "1", "2"],
        "hindsight": ["3"],
        "teacher_tp4": ["4", "5", "6", "7"],
    }
    for arm, ablation in ARMS:
        cmd = plan["train_commands"][arm]
        assert cmd[cmd.index("--reasoning-ablation") + 1] == ablation
        assert cmd[cmd.index("--model-family") + 1] == "nemotron"
        assert cmd[cmd.index("--rounds") + 1] == "4"
        assert cmd[cmd.index("--global-batch-prompts") + 1] == "256"
        assert cmd[cmd.index("--teacher-gpus-per-worker") + 1] == "4"
        assert ("--teacher-gpus" in cmd) == (arm != "opsd")
    evaluation = plan["eval_command"]
    assert "--include-base" in evaluation
    assert evaluation.count("--checkpoint") == 3
    assert "--decoding" in evaluation
    assert evaluation[evaluation.index("--decoding") + 1] == "nemotron-thinking"
