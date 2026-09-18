"""Continuous-batch evaluation, sharing Lulu's prompts, parser and row schema."""
from __future__ import annotations

from contextlib import ExitStack
import gc
import hashlib
import json
from pathlib import Path
import time

from lulu.evaluation_runner import load_parser


def response_budget(plan, prompt_tokens):
    """Reserve context for the complete prompt; never truncate input tokens."""
    requested = plan['max_response_tokens']
    limit = plan.get('max_model_len')
    if limit is None:
        return requested
    available = limit - prompt_tokens - plan.get('context_safety_margin', 0)
    if available < 1:
        raise ValueError('Complete prompt plus context safety margin leaves no generation budget')
    return min(requested, available)


def prepare_models(plan, load_assets):
    """Merge each PEFT checkpoint once on CPU before the GPU workers start."""
    from lulu.evaluation_runner import resolve_dtype
    for model in plan['models']:
        source = Path(model['model'])
        if not (source/'adapter_config.json').is_file():
            model['inference_model'] = model['model']
            continue
        destination = Path(plan['output_dir'])/'merged_models'/model['name']
        merged, tokenizer = load_assets(model['model'], dtype=resolve_dtype(plan['dtype']),
            device='cpu', thinking=plan['thinking'], trust_remote_code=plan['trust_remote_code'],
            adapter_base_model=plan['adapter_base_model'])
        merged.save_pretrained(destination, safe_serialization=True)
        tokenizer.save_pretrained(destination)
        model['inference_model'] = str(destination)
        del merged, tokenizer
        gc.collect()


def build_jobs(plan, shard_id, tokenizer):
    """Cap before sharding; queue every benchmark together to refill idle slots."""
    import pyarrow.parquet as pq
    jobs = []
    for benchmark in plan['benchmarks']:
        rows = pq.read_table(benchmark['path']).to_pylist()
        if plan['max_examples']:
            rows = rows[:plan['max_examples']]
        for index, row in enumerate(rows):
            if index % len(plan['devices']) != shard_id:
                continue
            text = tokenizer.apply_chat_template(row['prompt'], tokenize=False,
                add_generation_prompt=True, enable_thinking=plan['thinking'])
            ids = tokenizer(text, add_special_tokens=False)['input_ids']
            if len(ids) > plan['max_prompt_tokens']:
                raise ValueError(f"{benchmark['name']} prompt_index={index} has {len(ids)} tokens; "
                                 f"exceeds --max-prompt-tokens={plan['max_prompt_tokens']}")
            jobs.append({'benchmark': benchmark['name'], 'index': index,
                         'row': row, 'prompt_token_ids': ids,
                         'max_response_tokens': response_budget(plan, len(ids))})
    return jobs


def run_worker(plan, shard_id):
    # vLLM workers run one model per process. Process exit releases its engine,
    # CUDA graphs and KV cache before the next model starts on these GPUs.
    if len(plan['models']) != 1:
        raise ValueError('vLLM workers require exactly one model per process')
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model = plan['models'][0]
    source = model.get('inference_model', model['model'])
    tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=plan['trust_remote_code'])
    jobs = build_jobs(plan, shard_id, tokenizer)
    parser = load_parser(plan['parser_path'])
    output_root = Path(plan['output_dir'])/model['name']
    with ExitStack() as stack:
        files = {}
        for benchmark in plan['benchmarks']:
            path = output_root/benchmark['name']/f'shard-{shard_id:03d}.jsonl'
            path.parent.mkdir(parents=True, exist_ok=True)
            files[benchmark['name']] = stack.enter_context(path.open('w'))
        if not jobs:
            return
        settings = plan['vllm']
        engine = LLM(model=source, tokenizer=source, dtype=plan['dtype'],
            trust_remote_code=plan['trust_remote_code'], tensor_parallel_size=1,
            max_model_len=plan.get('max_model_len') or plan['max_prompt_tokens']+plan['max_response_tokens'],
            gpu_memory_utilization=settings['gpu_memory_utilization'],
            max_num_seqs=settings['max_num_seqs'],
            max_num_batched_tokens=settings['max_num_batched_tokens'],
            enable_chunked_prefill=True, enforce_eager=settings['enforce_eager'],
            swap_space=0, seed=plan.get('sampling', {}).get('seed', 42), generation_config='vllm')
        # vLLM V1 owns a non-daemon engine process. Explicitly shut it down
        # before interpreter teardown, which otherwise waits on that live child.
        core = getattr(getattr(engine, 'llm_engine', None), 'engine_core', None)
        if core is not None:
            stack.callback(core.shutdown)
        policy = plan.get("sampling", {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "seed": 42})
        sampling_args = dict(n=1, temperature=policy["temperature"], top_p=policy["top_p"],
            top_k=policy["top_k"], min_p=0.0, repetition_penalty=1.0,
            max_tokens=plan["max_response_tokens"], ignore_eos=False, detokenize=False)
        if policy["temperature"] > 0:
            # Stable across checkpoints and GPU shard counts; Python hash is not stable.
            for job in jobs:
                key = f"{policy['seed']}:{job['benchmark']}:{job['index']}"
                job["generation_seed"] = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")
            sampling = [SamplingParams(**dict(sampling_args, max_tokens=job['max_response_tokens']),
                                       seed=job["generation_seed"]) for job in jobs]
        elif plan.get('max_model_len') is not None:
            sampling = [SamplingParams(**dict(sampling_args, max_tokens=job['max_response_tokens'])) for job in jobs]
        else:
            sampling = SamplingParams(**sampling_args)
        started = time.monotonic()
        # One queue spans ALL datasets. max_num_seqs controls resident requests;
        # completed requests release slots immediately instead of padding a batch.
        outputs = engine.generate([{'prompt_token_ids': j['prompt_token_ids']} for j in jobs],
                                  sampling_params=sampling, use_tqdm=True)
        generation_seconds = time.monotonic()-started
        if len(outputs) != len(jobs):
            raise RuntimeError('vLLM returned incomplete generation coverage')
        response_tokens = 0
        for job, result in zip(jobs, outputs):
            if list(result.prompt_token_ids) != job['prompt_token_ids'] or len(result.outputs) != 1 or not result.finished:
                raise RuntimeError('vLLM response does not match its requested prompt')
            completion = result.outputs[0]
            tokens = list(completion.token_ids)
            if len(tokens) > job['max_response_tokens']:
                raise RuntimeError('vLLM exceeded the per-request response budget')
            text = tokenizer.decode(tokens, skip_special_tokens=True)
            row = job['row']
            source_name = str(row.get('data_source', ''))
            prediction = parser.extract_answer(text, source_name)
            gold = (row.get('reward_model') or {}).get('ground_truth')
            record = {'prompt_index': job['index'], 'data_source': source_name,
                'prediction': prediction, 'ground_truth': gold,
                'reward': float(parser.math_equal(prediction, gold)) if gold is not None else None,
                'prompt_tokens': len(job['prompt_token_ids']), 'response_tokens': len(tokens),
                'hit_cap': completion.finish_reason == 'length',
                'max_response_tokens': job['max_response_tokens'],
                'context_limited': job['max_response_tokens'] < plan['max_response_tokens'],
                'finish_reason': completion.finish_reason}
            if 'generation_seed' in job:
                record['generation_seed'] = job['generation_seed']
            if plan['store_text']:
                record['response'] = text
            if plan.get('store_token_ids', False):
                record['response_token_ids'] = tokens
                record['prompt_sha256'] = hashlib.sha256(
                    json.dumps(job['prompt_token_ids'], separators=(',', ':')).encode()).hexdigest()
            files[job['benchmark']].write(json.dumps(record, ensure_ascii=False)+'\n')
            response_tokens += len(tokens)
        elapsed = time.monotonic()-started
        timing = {'backend': 'vllm', 'model': model['name'], 'shard_id': shard_id,
            'examples': len(jobs), 'generation_seconds': generation_seconds,
            'scoring_seconds': elapsed-generation_seconds, 'response_tokens': response_tokens,
            'generation_tokens_per_second': response_tokens/generation_seconds}
        path = Path(plan['output_dir'])/f'timing-{model["name"]}-{shard_id:03d}.json'
        path.write_text(json.dumps(timing, indent=2)+'\n')
        print(json.dumps(timing), flush=True)
