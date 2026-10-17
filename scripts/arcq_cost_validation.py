"""Fail-closed validation shared by the launcher and report generator."""
import hashlib
import json
import math
from pathlib import Path

COUNTS = {'math500': 500, 'aime25': 30, 'olympiadbench': 512, 'mmlu_pro': 512, 'gpqa_diamond': 198}
ARMS = {'vanilla_opd': 'vanilla', 'hind': 'ren', 'opsd': 'opsd'}


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def number(value, name, *, positive=False, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f'{name}: missing or non-finite number')
    if value < 0 or (positive and value == 0) or (maximum is not None and value > maximum):
        raise ValueError(f'{name}: invalid value {value}')
    return value


def validate_training(label, root, metrics, telemetry):
    cfg = load(root/'train/run_config.json')
    if cfg.get('method') != 'ren_balanced' or cfg.get('reasoning_ablation') != ARMS[label] or cfg.get('cost_mode') is not True:
        raise ValueError('Algorithm label/configuration mismatch or cost-mode disabled')
    if cfg.get('rounds') != 4 or len(metrics) != 4 or [r.get('round') for r in metrics] != list(range(4)):
        raise ValueError('Expected complete, ordered four-round training')
    checkpoint = (root/'train/checkpoints/round_000004').resolve()
    state = load(checkpoint/'lulu_state.json')
    if state.get('completed_rounds') != 4 or state.get('completed_updates') != 4:
        raise ValueError('Final checkpoint is incomplete')
    if not list(checkpoint.glob('*.safetensors')):
        raise ValueError('Final checkpoint weights missing')
    if digest(cfg['train_data']) != cfg['train_sha256']:
        raise ValueError('Training pool changed')
    for row in metrics:
        if row.get('skipped_update') is not False:
            raise ValueError('Skipped or unverified update')
        for field in ('response_tokens', 'round_seconds', 'pipeline_seconds', 'update_seconds', 'weight_scoring_seconds',
                      'student_peak_memory_allocated_mib', 'student_peak_memory_reserved_mib'):
            number(row.get(field), field, positive=True)
        for group in ('collection', 'update', 'services'):
            if not row.get('cost_phases', {}).get(group):
                raise ValueError(f'Missing phase measurements: {group}')
        for field in ('generation_seconds', 'generation_request_seconds', 'causal_scoring_seconds'):
            number(row['cost_phases']['collection'].get(field, {}).get('sum_worker_seconds'), field, positive=True)
        for field in ('hindsight_forward_seconds', 'reference_forward_seconds', 'teacher_forward_seconds'):
            number(row['cost_phases']['services'].get(field), field)
        rollout_rows = []
        for shard in sorted((root/'train/rollouts'/f'round_{row["round"]:04d}').glob('shard-*.jsonl')):
            rollout_rows.extend(json.loads(line) for line in shard.read_text().splitlines())
        expected = cfg['global_batch_prompts']*cfg['rollouts_per_prompt']
        if len(rollout_rows) != expected or len({r['index'] for r in rollout_rows}) != expected:
            raise ValueError('Rollout trajectory coverage mismatch')
        if sum(len(r['response_ids']) for r in rollout_rows) != row['response_tokens']:
            raise ValueError('Generated token count disagrees with saved trajectories')
    validate_telemetry(telemetry, 'train')
    command = telemetry.get('command', [])
    if '--output-dir' not in command or Path(command[command.index('--output-dir')+1]).resolve() != (root/'train').resolve():
        raise ValueError('Telemetry does not belong to this training directory')
    return cfg, checkpoint


def validate_telemetry(value, scope):
    if value.get('status') != 'complete' or value.get('exit_code') != 0 or value.get('monitoring_complete') is not True or value.get('scope') != scope:
        raise ValueError(f'{scope} monitoring incomplete')
    elapsed = number(value.get('wall_seconds'), 'wall_seconds', positive=True)
    count = number(value.get('allocated_gpu_count'), 'gpu_count', positive=True)
    hours = number(value.get('gpu_hours'), 'gpu_hours', positive=True)
    if not math.isclose(hours, elapsed*count/3600, rel_tol=1e-9):
        raise ValueError('GPU-hours arithmetic mismatch')
    if len(value.get('gpus', {})) != count:
        raise ValueError('Missing GPU inventory')


def verify_evaluation(root, selected_name, checkpoint):
    import pyarrow.parquet as pq
    plan = load(root/'evaluation/eval_plan.json')
    selected = [m for m in plan['models'] if m['name'] == selected_name]
    if len(selected) != 1 or Path(selected[0]['model']).resolve() != checkpoint:
        raise ValueError('Evaluation checkpoint mismatch')
    if plan.get('decoding') != 'qwen-thinking' or plan.get('max_examples') != 512 or plan.get('max_response_tokens') != 32768:
        raise ValueError('Evaluation protocol mismatch')
    if set(b['name'] for b in plan['benchmarks']) != set(COUNTS):
        raise ValueError('Evaluation benchmarks mismatch')
    actual, identities = {}, {}
    for benchmark in plan['benchmarks']:
        name = benchmark['name']
        rows = {}
        for shard in sorted((root/'evaluation'/selected_name/name).glob('shard-*.jsonl')):
            for line in shard.read_text().splitlines():
                row = json.loads(line)
                index = row['prompt_index']
                if type(index) is not int or index in rows:
                    raise ValueError(f'{name}: duplicate/invalid prompt index')
                reward = number(row.get('reward'), f'{name}/reward', maximum=1)
                if reward not in (0, 1):
                    raise ValueError('Expected binary Pass@1 rewards')
                rows[index] = row
        if set(rows) != set(range(COUNTS[name])):
            raise ValueError(f'{name}: incomplete evaluation coverage')
        source_rows = pq.read_table(benchmark['path']).slice(0, COUNTS[name]).to_pylist()
        if len(source_rows) != COUNTS[name]:
            raise ValueError(f'{name}: evaluation source too short')
        for index, source in enumerate(source_rows):
            if rows[index].get('ground_truth') != source['reward_model']['ground_truth']:
                raise ValueError(f'{name}: source answer/index mismatch at {index}')
        actual[name] = sum(row['reward'] for row in rows.values())/len(rows)
        identities[name] = {'parquet_sha256': digest(benchmark['path']), 'rows': len(rows),
                            'answers_sha256': hashlib.sha256(json.dumps([rows[i]['ground_truth'] for i in sorted(rows)], sort_keys=True).encode()).hexdigest()}
    return actual, identities, {k: plan.get(k) for k in ('decoding','sampling','max_examples','max_response_tokens','max_prompt_tokens','max_model_len','context_safety_margin','devices','vllm','backend','thinking','split','parser_path')}


def compare_runs(rows):
    reference = rows[0]
    for row in rows[1:]:
        for key in ('comparison_config', 'evaluation_identity', 'evaluation_protocol', 'hardware', 'source_sha256'):
            if row[key] != reference[key]:
                raise ValueError(f'Not comparable: {row["label"]} differs in {key}')
    for row in rows:
        row['relative_opd_cost'] = row['gpu_hours']/reference['gpu_hours']
        row['accuracy_gain_pp'] = 100*(row['macro_accuracy']-reference['macro_accuracy'])
        row['gpu_hours_per_million_tokens'] = row['gpu_hours']*1e6/row['generated_tokens']
