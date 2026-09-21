from types import SimpleNamespace

from lulu.nemotron_family import (
    THINKING_SYSTEM_PROMPT, apply_reasoning_template, nemotron_teacher_tp_plan,
    thinking_messages,
)


def test_thinking_messages_adds_system_without_changing_question():
    original = [{"role": "user", "content": "Solve x + 1 = 2"}]
    result = thinking_messages(original)
    assert result == [
        {"role": "system", "content": THINKING_SYSTEM_PROMPT},
        {"role": "user", "content": "Solve x + 1 = 2"},
    ]
    assert original == [{"role": "user", "content": "Solve x + 1 = 2"}]


def test_thinking_messages_rejects_ambiguous_existing_system():
    original = [
        {"role": "system", "content": "Answer briefly"},
        {"role": "user", "content": "What is 2 + 2?"},
    ]
    try:
        thinking_messages(original)
    except ValueError as exc:
        assert "system instructions" in str(exc)
    else:
        raise AssertionError("Ambiguous system policy should be rejected")


def test_nemotron_template_does_not_pass_qwen_flag():
    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert "enable_thinking" not in kwargs
            return messages

    rendered = apply_reasoning_template(FakeTokenizer(),
        [{"role": "user", "content": "Solve"}], family="nemotron",
        enable_thinking=True, tokenize=False, add_generation_prompt=True)
    assert rendered[0]["content"] == THINKING_SYSTEM_PROMPT


def test_nemotron_teacher_plan_is_true_four_way_sharding():
    config = SimpleNamespace(model_type="nemotron-nas", num_attention_heads=64)
    plan = nemotron_teacher_tp_plan(config, 4)
    assert plan["layers.*.self_attn.q_proj"] == "colwise_rep"
    assert plan["layers.*.mlp.down_proj"] == "rowwise_rep"
