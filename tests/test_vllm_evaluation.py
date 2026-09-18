"""CPU contract checks for vLLM scheduling, prompt parity and shared scoring."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from lulu import vllm_evaluation as evaluation


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] and kwargs['add_generation_prompt']
        return messages[0]['content']

    def __call__(self, text, **kwargs):
        assert kwargs['add_special_tokens'] is False
        return {'input_ids': [int(text)+10]}

    def decode(self, tokens, **kwargs):
        return 'answer'


@pytest.fixture
def plan(tmp_path):
    benchmarks = []
    for name, count in [('math500', 210), ('aime25', 30)]:
        path = tmp_path/f'{name}.parquet'
        pq.write_table(pa.Table.from_pylist([
            {'prompt': [{'role': 'user', 'content': str(i)}], 'data_source': name,
             'reward_model': {'ground_truth': 'answer'}} for i in range(count)]), path)
        benchmarks.append({'name': name, 'path': str(path)})
    parser = tmp_path/'parser.py'
    parser.write_text('def extract_answer(text, source): return text\ndef math_equal(a, b): return a == b\n')
    return {'models': [{'name': 'final', 'model': 'adapter', 'inference_model': 'merged'}],
        'benchmarks': benchmarks, 'devices': [str(i) for i in range(8)], 'max_examples': 199,
        'max_prompt_tokens': 4096, 'max_response_tokens': 8192, 'thinking': True,
        'output_dir': str(tmp_path/'out'), 'parser_path': str(parser), 'store_text': True,
        'trust_remote_code': False, 'adapter_base_model': None, 'dtype': 'bfloat16',
        'vllm': {'gpu_memory_utilization': .8, 'max_num_seqs': 64,
                 'max_num_batched_tokens': 4096, 'enforce_eager': False}}


def test_cap_precedes_sharding_and_all_benchmarks_share_one_queue(plan):
    queues = [evaluation.build_jobs(plan, i, Tokenizer()) for i in range(8)]
    pairs = [(job['benchmark'], job['index']) for queue in queues for job in queue]
    assert len(pairs) == len(set(pairs)) == 229
    assert {i for bench, i in pairs if bench == 'math500'} == set(range(199))
    assert {i for bench, i in pairs if bench == 'aime25'} == set(range(30))
    for shard, queue in enumerate(queues):
        assert all(job['index'] % 8 == shard for job in queue)
        assert {job['benchmark'] for job in queue} == {'math500', 'aime25'}


def fake_engine(monkeypatch, invalid=False):
    calls = []
    class Engine:
        def __init__(self, **kwargs):
            calls.append(('init', kwargs))
        def generate(self, prompts, sampling_params, use_tqdm):
            calls.append(('generate', prompts, sampling_params))
            return [SimpleNamespace(prompt_token_ids=[-1] if invalid else p['prompt_token_ids'], finished=True,
                outputs=[SimpleNamespace(token_ids=[77], finish_reason='length' if i == 0 else 'stop')])
                for i, p in enumerate(prompts)]
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(LLM=Engine, SamplingParams=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, 'transformers', SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: Tokenizer())))
    return calls


def test_continuous_batch_worker_uses_one_queue_and_writes_compatible_rows(plan, monkeypatch):
    calls = fake_engine(monkeypatch)
    evaluation.run_worker(plan, 0)
    assert len(calls) == 2
    assert calls[0][1]['model'] == 'merged'
    assert calls[0][1]['max_num_seqs'] == 64
    assert calls[0][1]['enable_chunked_prefill'] is True
    assert calls[0][1]['generation_config'] == 'vllm'
    assert len(calls[1][1]) == 29
    assert calls[1][2]['temperature'] == 0 and calls[1][2]['max_tokens'] == 8192
    assert calls[1][2]['ignore_eos'] is False
    root = Path(plan['output_dir'])
    rows = [json.loads(line) for file in (root/'final').glob('*/shard-000.jsonl') for line in file.read_text().splitlines()]
    assert len(rows) == 29 and all(row['reward'] == 1 for row in rows)
    assert sum(row['hit_cap'] for row in rows) == 1
    assert json.loads((root/'timing-final-000.json').read_text())['examples'] == 29


def test_response_order_mismatch_is_rejected(plan, monkeypatch):
    fake_engine(monkeypatch, invalid=True)
    with pytest.raises(RuntimeError, match='does not match'):
        evaluation.run_worker(plan, 0)


def test_overlong_prompt_fails_before_engine_initialization(plan, monkeypatch):
    calls = fake_engine(monkeypatch)
    plan['max_prompt_tokens'] = 0
    with pytest.raises(ValueError, match='max-prompt-tokens'):
        evaluation.run_worker(plan, 0)
    assert calls == []


def test_adapter_is_merged_once_on_cpu_before_any_workers(plan, tmp_path):
    adapter = tmp_path/'adapter'
    adapter.mkdir()
    (adapter/'adapter_config.json').write_text('{}')
    plan['models'] = [{'name': 'base', 'model': 'original-base'}, {'name': 'final', 'model': str(adapter)}]
    calls = []
    class Saved:
        def save_pretrained(self, path, **kwargs):
            Path(path).mkdir(parents=True, exist_ok=True)
    def load(model, **kwargs):
        calls.append((model, kwargs))
        return Saved(), Saved()
    evaluation.prepare_models(plan, load)
    assert len(calls) == 1 and calls[0][1]['device'] == 'cpu'
    assert plan['models'][0]['inference_model'] == 'original-base'
    assert plan['models'][1]['model'] == str(adapter)
    assert plan['models'][1]['inference_model'].endswith('merged_models/final')


def test_thinking_sampling_is_seeded_per_question_across_checkpoints_and_shards(plan, monkeypatch):
    plan['sampling'] = {'temperature': .6, 'top_p': .95, 'top_k': 20, 'seed': 42}
    calls = fake_engine(monkeypatch)
    evaluation.run_worker(plan, 0)
    original = [json.loads(line) for f in (Path(plan['output_dir'])/'final').glob('*/shard-000.jsonl')
                for line in f.read_text().splitlines()]
    seeds = {(r['data_source'], r['prompt_index']): r['generation_seed'] for r in original}
    assert len(set(seeds.values())) == len(seeds)
    params = calls[1][2]
    assert len(params) == len(original)
    assert all(p['temperature'] == .6 and p['top_p'] == .95 and p['top_k'] == 20 and p['min_p'] == 0 for p in params)
    plan['models'][0]['name'] = 'step20'
    plan['devices'] = plan['devices'][:4]
    evaluation.run_worker(plan, 0)
    changed = [json.loads(line) for f in (Path(plan['output_dir'])/'step20').glob('*/shard-000.jsonl')
               for line in f.read_text().splitlines()]
    assert all(r['generation_seed'] == seeds[(r['data_source'], r['prompt_index'])]
               for r in changed if (r['data_source'], r['prompt_index']) in seeds)


@pytest.mark.parametrize('invalid', [False, True])
def test_v1_engine_is_closed_after_success_or_scoring_failure(plan, monkeypatch, invalid):
    calls = fake_engine(monkeypatch, invalid=invalid)
    original = sys.modules['vllm'].LLM
    class ManagedEngine(original):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.llm_engine = SimpleNamespace(engine_core=SimpleNamespace(
                shutdown=lambda: calls.append(('shutdown',))))
    monkeypatch.setattr(sys.modules['vllm'], 'LLM', ManagedEngine)
    if invalid:
        with pytest.raises(RuntimeError, match='does not match'):
            evaluation.run_worker(plan, 0)
    else:
        evaluation.run_worker(plan, 0)
    assert calls[-1] == ('shutdown',)
    assert calls.count(('shutdown',)) == 1


def test_long_horizon_caps_output_without_truncating_prompt(plan, monkeypatch):
    calls = fake_engine(monkeypatch)
    monkeypatch.setattr(Tokenizer, '__call__', lambda self, text, **kw:
                        {'input_ids': [17] * (2807 if text == '0' else 200)})
    plan.update(max_model_len=40960, max_response_tokens=38912,
                context_safety_margin=128, store_token_ids=True,
                sampling={'temperature': .6, 'top_p': .95, 'top_k': 20, 'seed': 42})
    evaluation.run_worker(plan, 0)
    assert calls[0][1]['max_model_len'] == 40960
    for prompt, params in zip(calls[1][1], calls[1][2]):
        size = len(prompt['prompt_token_ids'])
        assert size in (200, 2807)
        assert params['max_tokens'] == (38025 if size == 2807 else 38912)
        assert size + params['max_tokens'] + 128 <= 40960
    rows = [json.loads(line) for f in (Path(plan['output_dir'])/'final').glob('*/shard-000.jsonl')
            for line in f.read_text().splitlines()]
    assert all(r['response_token_ids'] == [77] for r in rows)
    assert all(r['max_response_tokens'] == 38025 for r in rows if r['prompt_index'] == 0)
    assert all(r['context_limited'] == (r['prompt_index'] == 0) for r in rows)


def test_complete_prompt_with_no_remaining_context_is_rejected(plan):
    plan.update(max_model_len=40960, context_safety_margin=128)
    with pytest.raises(ValueError, match='leaves no generation budget'):
        evaluation.response_budget(plan, 40832)
