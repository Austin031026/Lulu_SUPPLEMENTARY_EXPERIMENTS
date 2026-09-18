"""Standalone batched evaluation utilities for Lulu."""
from __future__ import annotations

import importlib.util
import json
import time
from pathlib import Path


def load_parser(path):
    path = Path(path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location("_lulu_benchmark_parser", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import benchmark parser: {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    for name in ("extract_answer", "math_equal"):
        if not hasattr(module, name):
            raise RuntimeError(f"benchmark parser missing {name}: {path}")

    return module


def resolve_dtype(name):
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


class PlainModelRunner:
    """Batched evaluator with explicit, recorded decoding settings."""

    def __init__(
        self,
        *,
        parser_path,
        batch_size=8,
        max_response_tokens=8192,
        max_prompt_tokens=4096,
        dtype="bfloat16",
        sampling=None,
    ):
        self.parser = load_parser(parser_path)
        self.batch_size = int(batch_size)
        self.max_response_tokens = int(max_response_tokens)
        self.max_prompt_tokens = int(max_prompt_tokens)
        self.dtype = resolve_dtype(dtype)
        self.sampling = sampling or {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "seed": 42}

        self.model = None
        self.tokenizer = None
        self.model_id = None

    def close(self):
        self.model = None
        self.tokenizer = None
        self.model_id = None

        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def load_model(self, model_id, **kwargs):
        raise NotImplementedError

    @staticmethod
    def _messages(row):
        prompt = row.get("prompt")
        if not isinstance(prompt, list):
            raise ValueError("benchmark row must contain chat-style prompt")
        return prompt

    @staticmethod
    def _ground_truth(row):
        reward_model = row.get("reward_model") or {}
        return reward_model.get("ground_truth")

    def evaluate_shard(
        self,
        *,
        model_id,
        input_parquet,
        output,
        shard_id,
        num_shards,
        max_examples=0,
        store_text=False,
        progress_every=8,
    ):
        import pyarrow.parquet as pq
        import torch

        self.load_model(model_id)

        rows = pq.read_table(input_parquet).to_pylist()
        if max_examples:
            rows = rows[:max_examples]

        selected = [
            (i, row)
            for i, row in enumerate(rows)
            if i % num_shards == shard_id
        ]

        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        timing = {'backend': 'hf', 'examples': 0, 'response_tokens': 0,
                  'generation_seconds': 0.0, 'scoring_seconds': 0.0}
        with output.open("w") as handle, torch.inference_mode():
            for start in range(0, len(selected), self.batch_size):
                chunk = selected[start:start + self.batch_size]

                texts = [
                    self.tokenizer.apply_chat_template(
                        self._messages(row),
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for _, row in chunk
                ]

                encoded = self.tokenizer(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=False,
                    return_token_type_ids=False,
                    add_special_tokens=False,
                )
                encoded.pop("token_type_ids", None)

                prompt_lengths = encoded["attention_mask"].sum(dim=1).tolist()
                for (index, _), length in zip(chunk, prompt_lengths):
                    if length > self.max_prompt_tokens:
                        raise ValueError(
                            f"benchmark prompt_index={index} has {length} tokens, "
                            f"exceeding --max-prompt-tokens={self.max_prompt_tokens}; "
                            "increase --max-prompt-tokens to preserve the complete prompt"
                        )

                encoded = {
                    key: value.to(self.model.device)
                    for key, value in encoded.items()
                }

                generation_started = time.monotonic()
                sampling = self.sampling
                generation = {"do_sample": sampling["temperature"] > 0}
                if generation["do_sample"]:
                    # HF sampling is reproducible for fixed batching/sharding.
                    # vLLM additionally supplies an independent seed per request.
                    import hashlib
                    key = f"{sampling['seed']}:{chunk[0][1].get('data_source', '')}:{chunk[0][0]}"
                    torch.manual_seed(int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big"))
                    generation.update(temperature=sampling["temperature"], top_p=sampling["top_p"],
                                      top_k=max(0, sampling["top_k"]), min_p=0.0)
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=self.max_response_tokens,
                    **generation,
                    num_beams=1,
                )

                timing['generation_seconds'] += time.monotonic()-generation_started
                scoring_started = time.monotonic()
                prompt_width = encoded["input_ids"].shape[1]
                completions = generated[:, prompt_width:]

                for offset, ((index, row), tokens) in enumerate(zip(chunk, completions)):
                    token_ids = tokens.tolist()

                    eos = self.tokenizer.eos_token_id
                    if eos is not None and eos in token_ids:
                        token_ids = token_ids[:token_ids.index(eos) + 1]

                    text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                    source = str(row.get("data_source", ""))
                    prediction = self.parser.extract_answer(text, source)
                    ground_truth = self._ground_truth(row)

                    reward = None
                    if ground_truth is not None:
                        reward = float(self.parser.math_equal(prediction, ground_truth))

                    response_tokens = len(token_ids)
                    timing['examples'] += 1
                    timing['response_tokens'] += response_tokens

                    result = {
                        "prompt_index": index,
                        "data_source": source,
                        "prediction": prediction,
                        "ground_truth": ground_truth,
                        "reward": reward,
                        "prompt_tokens": int(prompt_lengths[offset]),
                        "response_tokens": response_tokens,
                        "hit_cap": response_tokens >= self.max_response_tokens,
                    }

                    if store_text:
                        result["response"] = text

                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")

                handle.flush()
                timing['scoring_seconds'] += time.monotonic()-scoring_started
                timing['generation_tokens_per_second'] = timing['response_tokens']/max(timing['generation_seconds'], 1e-9)
                timing_path = output.with_suffix('.timing.json')
                temp = timing_path.with_suffix('.tmp')
                temp.write_text(json.dumps(timing, indent=2)+'\n')
                temp.replace(timing_path)

                if progress_every and start % (progress_every * self.batch_size) == 0:
                    print(
                        f"[eval] shard={shard_id} "
                        f"completed={min(start + len(chunk), len(selected))}/{len(selected)}",
                        flush=True,
                    )