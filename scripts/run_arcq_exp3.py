#!/usr/bin/env python3
"""Freeze, smoke-test, train, evaluate and verify the three ArcQ cost arms.

Default: prepare only. --run executes sequentially on eight reserved GPUs.
Re-execution skips verified completed stages; failed runs are never auto-resumed.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from arcq_cost_validation import ARMS, COUNTS, digest, load, validate_telemetry
from run_with_cost_telemetry import atomic_json, query_gpus, git_provenance

CODE = Path(__file__).resolve().parents[1]
POOL_SHA = '7d8e0ef07b341b90a71958e4513bacdc3e086e152f55d47245c7dd9871bc5cc4'


def train_command(code, assets, out, arm, gpus, smoke=False):
    command = [sys.executable, '-u', str(code/'scripts/train_lulu.py'),
        '--model', assets['student'], '--teacher-model', assets['teacher'],
        '--train-data', assets['pool'], '--output-dir', str(out),
        '--method', 'ren_balanced', '--backend', 'persistent', '--cost-mode',
        '--reasoning-ablation', arm, '--gpus', ','.join(gpus),
        '--student-gpus', ','.join(gpus[:5]), '--hindsight-gpus', gpus[5],
        '--teacher-tp-mode', 'eager-local', '--teacher-gpus-per-worker', '2',
        '--rounds', '1' if smoke else '4', '--global-batch-prompts', '5' if smoke else '256',
        '--rollouts-per-prompt', '1', '--update-passes', '1',
        '--rollout-backend', 'vllm', '--rollout-batch-size', '16', '--rollout-top-k', '20',
        '--rollout-vllm-memory', '0.42', '--rollout-vllm-max-seqs', '32',
        '--score-batch-size', '2', '--train-micro-batch-size', '1', '--logit-chunk-size', '128',
        '--max-new-tokens', '64' if smoke else '8192', '--max-prompt-tokens', '4096',
        '--max-sequence-tokens', '16384', '--temperature', '0.6', '--top-p', '0.95', '--top-k', '32',
        '--learning-rate', '1e-6', '--weight-decay', '0', '--max-grad-norm', '1',
        '--control-loss-coef', '0.5', '--reference-kl-coef', '0.1',
        '--lora-rank', '0', '--master-weights-fp32', '--dtype', 'bfloat16',
        '--gradient-checkpointing', '--match-causal-update', '--gradient-norm-every', '0',
        '--validation-every', '0', '--no-stability-early-stop', '--save-every', '20',
        '--retain-checkpoints', '0,1' if smoke else '0,1,2,3,4', '--worker-timeout', '7200', '--seed', '42']
    if arm != 'opsd':
        command += ['--teacher-gpus', ','.join(gpus[6:])]
    return command


def eval_command(code, assets, out, checkpoint, gpus):
    command = [sys.executable, '-u', str(code/'scripts/evaluate_lulu.py'),
        '--model', assets['student'], '--checkpoint', f'final={checkpoint}',
        '--output-dir', str(out), '--gpus', ','.join(gpus), '--backend', 'vllm',
        '--benchmarks', ','.join(COUNTS), '--max-examples', '512',
        '--decoding', 'qwen-thinking', '--thinking', '--seed', '42',
        '--max-response-tokens', '32768', '--max-prompt-tokens', '4096',
        '--max-model-len', '40960', '--context-safety-margin', '128',
        '--vllm-gpu-memory-utilization', '0.8', '--vllm-max-num-seqs', '32',
        '--vllm-max-num-batched-tokens', '4096', '--store-text', '--store-token-ids',
        '--parser-path', str(code/'lulu/thinking_final_parser.py')]
    for name in COUNTS:
        command += ['--benchmark', f'{name}={assets["benchmarks"][name]}']
    return command


def input_assets(root):
    import pyarrow.parquet as pq
    assets = {'student': str(root/'models/Qwen3-1.7B'), 'teacher': str(root/'models/Qwen3-32B'),
              'pool': str(root/'data/prepared/dapo_pool2048_s42/train.jsonl'),
              'benchmarks': {name: str(root/'data/prepared/rq1_eval_v1'/name/'full.parquet') for name in COUNTS}}
    hashes = {}
    for key in ('student', 'teacher'):
        directory = Path(assets[key])
        cfg = load(directory/'config.json')
        if cfg.get('model_type') != 'qwen3':
            raise ValueError(f'{key}: expected Qwen3 model')
        for name in ('config.json','tokenizer_config.json','tokenizer.json'):
            hashes[str(directory/name)] = digest(directory/name)
        index = directory/'model.safetensors.index.json'
        names = set(load(index)['weight_map'].values()) if index.exists() else {'model.safetensors'}
        if index.exists():
            hashes[str(index)] = digest(index)
        from safetensors import safe_open
        for name in names:
            with safe_open(str(directory/name), framework='pt', device='cpu') as weights:
                if not list(weights.keys()):
                    raise ValueError(f'Empty model shard: {directory/name}')
    hashes[assets['pool']] = digest(assets['pool'])
    if hashes[assets['pool']] != POOL_SHA:
        raise ValueError('Historical 2048-question pool SHA256 mismatch')
    for name, path in assets['benchmarks'].items():
        table = pq.read_table(path)
        expected = 1400 if name == 'mmlu_pro' else COUNTS[name]
        if len(table) != expected or not {'prompt','reward_model'} <= set(table.column_names):
            raise ValueError(f'{name}: invalid evaluation source')
        hashes[path] = digest(path)
    return assets, hashes


def freeze_code(target):
    hashes = {}
    for folder in ('lulu', 'scripts'):
        for source in sorted((CODE/folder).rglob('*.py')):
            relative = source.relative_to(CODE)
            destination = target/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            hashes[str(destination)] = digest(destination)
    return hashes


def ensure_idle(gpus):
    rows = query_gpus(gpus)
    if any(row['memory_used_mib'] > 1024 or row['utilization_percent'] > 5 for row in rows):
        raise RuntimeError('The eight requested GPUs must be idle before each stage; no processes were stopped')
    if len({(r['name'], r['memory_total_mib']) for r in rows}) != 1:
        raise RuntimeError('Expected identical GPU models and capacities')


def execute_stage(root, plan, label, scope, command, telemetry, env):
    if (telemetry/'summary.json').exists():
        summary = load(telemetry/'summary.json')
        validate_telemetry(summary, scope)
        if summary['command'] != command:
            raise ValueError('Recorded stage command differs from frozen plan')
        print(f'SKIP completed {label}/{scope}', flush=True)
        return
    if telemetry.exists() and any(telemetry.iterdir()):
        raise RuntimeError(f'Incomplete attempt retained at {telemetry}; use a new suite directory for a fresh attempt')
    for path, expected in plan['sha256'].items():
        if digest(path) != expected:
            raise ValueError(f'Frozen input/code changed: {path}')
    ensure_idle(plan['gpus'])
    log = root/'logs'/f'{label}_{scope}.log'
    log.parent.mkdir(exist_ok=True)
    atomic_json(root/'status.json', {'state':'running','algorithm':label,'stage':scope,'log':str(log)})
    wrapper = [sys.executable, '-u', str(root/'code/scripts/run_with_cost_telemetry.py'),
               '--output-dir', str(telemetry), '--label', label, '--scope', scope,
               '--gpus', ','.join(plan['gpus']), '--poll-seconds', '1', '--', *command]
    print(f'RUN {label}/{scope}; log={log}', flush=True)
    stage_env = dict(env)
    for variable, subdir in [('TRITON_CACHE_DIR','triton'),('VLLM_CACHE_ROOT','vllm'),('TORCHINDUCTOR_CACHE_DIR','inductor')]:
        stage_env[variable] = str(root/'cache'/label/scope/subdir)
    with log.open('x') as stream:
        child = subprocess.Popen(wrapper, env=stage_env, cwd=root/'code', stdout=stream, stderr=subprocess.STDOUT)
        try:
            result = child.wait()
            if result:
                raise subprocess.CalledProcessError(result, wrapper)
        finally:
            if child.poll() is None:
                # The telemetry wrapper forwards termination to its owned worker group.
                child.terminate()
                child.wait()
    validate_telemetry(load(telemetry/'summary.json'), scope)


def attempt_inventory(root):
    """Retain smoke/evaluation/failed costs separately, never hide retries."""
    attempts = []
    for path in sorted(root.rglob('telemetry/manifest.json')) + sorted(root.rglob('evaluation_telemetry/manifest.json')):
        manifest = load(path)
        summary_path = path.parent/'summary.json'
        summary = load(summary_path) if summary_path.exists() else {}
        attempts.append({'path': str(path.parent.relative_to(root)), 'label': manifest['label'],
                         'scope': manifest['scope'], 'status': summary.get('status', 'unknown_incomplete'),
                         'wall_seconds': summary.get('wall_seconds'), 'gpu_hours': summary.get('gpu_hours'),
                         'monitoring_complete': summary.get('monitoring_complete', False)})
    atomic_json(root/'attempt_costs.json', {'attempts': attempts,
        'known_gpu_hours': sum(a['gpu_hours'] for a in attempts if a['gpu_hours'] is not None),
        'unknown_cost_attempts': sum(a['gpu_hours'] is None for a in attempts)})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arcq-root', default=os.environ.get('ARCQ_ROOT'))
    p.add_argument('--output-dir')
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    p.add_argument('--run', action='store_true')
    args = p.parse_args()
    if not args.arcq_root:
        p.error('Source env/arcq_env.sh or pass --arcq-root')
    workspace = Path(args.arcq_root).resolve(strict=True)
    root = Path(args.output_dir).resolve() if args.output_dir else workspace/'experiments/exp3_cost'/time.strftime('suite_%Y%m%dT%H%M%S')
    if not root.is_relative_to(workspace/'experiments'):
        raise ValueError('Output must be under ARCQ_ROOT/experiments')
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan_path = root/'experiment_plan.json'
        if plan_path.exists():
            plan = load(plan_path)
            if plan['gpus'] != args.gpus.split(','):
                raise ValueError('GPU placement changed from frozen plan')
        else:
            assets, hashes = input_assets(workspace)
            gpus = args.gpus.split(',')
            if len(set(gpus)) != 8 or len(gpus) != 8:
                raise ValueError('This matched protocol reserves eight distinct GPUs')
            hashes.update(freeze_code(root/'code'))
            commands = {}
            for label, arm in ARMS.items():
                run = root/'runs'/label
                commands[label] = {
                    'smoke': train_command(root/'code', assets, root/'smoke'/label/'train', arm, gpus, True),
                    'train': train_command(root/'code', assets, run/'train', arm, gpus),
                    'evaluation': eval_command(root/'code', assets, run/'evaluation', run/'train/checkpoints/round_000004', gpus)}
            plan = {'schema_version':2, 'gpus':gpus,'assets':assets,'sha256':hashes,'commands':commands,
                    'git':git_provenance(str(CODE)), 'protocol':'matched control/reference; fixed 8 reserved GPUs',
                    'checkpoint_reuse':'vanilla_opd/checkpoints/round_000004 is available for Experiment 1',
                    'model_identity_scope':'local paths, config/tokenizer/index hashes and safetensors readability; Hub revision not independently proven'}
            atomic_json(plan_path, plan)
        print(f'PLAN={plan_path}', flush=True)
        if not args.run:
            return
        allocation_dir = workspace/'experiments/exp3_cost'
        allocation_dir.mkdir(parents=True, exist_ok=True)
        allocation_lock = (allocation_dir/'.allocation.lock').open('a')
        fcntl.flock(allocation_lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        def interrupted(signum, frame):
            raise InterruptedError(f'Controller received signal {signum}')
        signal.signal(signal.SIGTERM, interrupted)
        env = dict(os.environ, HF_HUB_OFFLINE='1', TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        env.pop('TRANSFORMERS_CACHE', None)
        env.pop('PYTHONPATH', None)
        try:
            # All three real-GPU smoke updates must succeed before any production training.
            for label in ARMS:
                execute_stage(root, plan, label, 'smoke', plan['commands'][label]['smoke'], root/'smoke'/label/'telemetry', env)
            for label in ARMS:
                run = root/'runs'/label
                execute_stage(root, plan, label, 'train', plan['commands'][label]['train'], run/'telemetry', env)
                state = load(run/'train/checkpoints/round_000004/lulu_state.json')
                if state.get('completed_updates') != 4:
                    raise ValueError('Incomplete training checkpoint')
                execute_stage(root, plan, label, 'evaluation', plan['commands'][label]['evaluation'], run/'evaluation_telemetry', env)
            command = [sys.executable, str(root/'code/scripts/summarize_arcq_exp3_cost.py'), '--output-dir', str(root/'analysis')]
            for label in ARMS:
                command += ['--run', f'{label}={root}/runs/{label}', '--evaluation-model', f'{label}=final']
            subprocess.run(command, check=True, env=env)
            atomic_json(root/'status.json', {'state':'complete','report':str(root/'analysis/REPORT.md')})
            print(f'ARCQ_EXP3=PASS\nREPORT={root}/analysis/REPORT.md', flush=True)
        except BaseException as exc:
            atomic_json(root/'status.json', {'state':'failed','error':repr(exc),'preserved_directory':str(root)})
            raise
        finally:
            attempt_inventory(root)
            allocation_lock.close()


if __name__ == '__main__':
    main()
