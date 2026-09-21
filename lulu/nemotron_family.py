"""Model-family policy shared by Nemotron training and evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


THINKING_SYSTEM_PROMPT = "detailed thinking on"
NON_THINKING_SYSTEM_PROMPT = "detailed thinking off"


def thinking_messages(
    messages: Sequence[Mapping[str, str]], *, enable_thinking: bool = True,
) -> list[dict[str, str]]:
    """Apply NVIDIA's mode protocol without changing the user question."""
    if not messages or messages[-1].get("role") != "user":
        raise ValueError("Nemotron prompt must end in a user message")
    result = [dict(message) for message in messages]
    systems = [i for i, message in enumerate(result) if message.get("role") == "system"]
    if len(systems) > 1 or (systems and systems[0] != 0):
        raise ValueError("At most one leading system message is supported")
    mode = THINKING_SYSTEM_PROMPT if enable_thinking else NON_THINKING_SYSTEM_PROMPT
    if systems:
        existing = result[0].get("content", "").strip()
        if existing and existing != mode:
            raise ValueError("Existing system instructions need an explicit Nemotron prompt policy")
        result[0]["content"] = mode
    else:
        result.insert(0, {"role": "system", "content": mode})
    return result


def apply_reasoning_template(tokenizer, messages, *, family="qwen", enable_thinking=True, **kwargs):
    """Render a chat without assuming Qwen's ``enable_thinking`` template API."""
    if family == "nemotron":
        return tokenizer.apply_chat_template(
            thinking_messages(messages, enable_thinking=enable_thinking), **kwargs
        )
    if family != "qwen":
        raise ValueError(f"Unsupported model family: {family}")
    return tokenizer.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)


def nemotron_teacher_tp_plan(config, world):
    """Keep NAS decoder activations replicated while sharding large projections.

    Gathered column outputs let DeciLM's unmodified attention keep its original
    head counts. Rowwise layers split their replicated inputs and reduce their
    outputs. These are paths relative to the decoder base model; Transformers
    prefixes ``model.`` when it builds the top-level CausalLM plan. True
    sharding is verified after loading by the Teacher service.
    """
    if world != 4 or getattr(config, "model_type", None) != "nemotron-nas":
        raise ValueError("Nemotron Super Teacher requires four TP ranks and nemotron-nas config")
    if config.num_attention_heads % world:
        raise ValueError("Teacher attention head count must divide TP size")
    return {
        "layers.*.self_attn.q_proj": "colwise_rep",
        "layers.*.self_attn.k_proj": "colwise_rep",
        "layers.*.self_attn.v_proj": "colwise_rep",
        "layers.*.self_attn.o_proj": "rowwise_rep",
        "layers.*.mlp.gate_proj": "colwise_rep",
        "layers.*.mlp.up_proj": "colwise_rep",
        "layers.*.mlp.down_proj": "rowwise_rep",
    }
