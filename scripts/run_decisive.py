"""Freeze and queue the fixed-pool ReN experiment, then evaluate base and final.

Default invocation prepares a reviewable plan only. --run waits for complete
Teacher weights and eight idle GPUs before starting. Existing GPU jobs are never
stopped. The queued job executes a source snapshot, not a moving checkout.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
BENCH_ROOT = WORKSPACE/'Soraka_rlrl/experiments/v6_5-success-q-scale-pool4096-phase11024-seed42/crossbench_v631/data'
BENCHMARKS = ('math500', 'aime25', 'olympiadbench', 'mmlu_pro', 'gpqa_diamond')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')
    temp.replace(path)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', default=str(WORKSPACE/'LuLu_outputs/experiments/ren_qwen1p7_teacher32b_pool2048_s42'))
    p.add_argument('--pool-dir', default=str(WORKSPACE/'LuLu_outputs/data/dapo_pool2048_s42'))
    p.add_argument('--model', default='Qwen/Qwen3-1.7B')
    p.add_argument('--teacher-model', default=str(WORKSPACE/'LuLu_outputs/models/Qwen3-32B'),
                   help='Local directory containing the staged Teacher checkpoint')
    p.add_argument('--hf-cache', default=str(WORKSPACE.parent/'huggingface_cache/hub'))
    p.add_argument('--data-manifest', default=str(BENCH_ROOT/'manifest.json'))
    p.add_argument('--method', choices=('ren_opd', 'ren_resolved', 'ren_stable', 'ren_balanced'), default='ren_opd')
    p.add_argument('--lora-rank', type=int, default=16)
    p.add_argument('--learning-rate', type=float, default=None, help='Stable ReN defaults to 1e-6; legacy methods 1e-5')
    p.add_argument('--reference-kl-coef', type=float, default=.1)
    p.add_argument('--stable-ratio-epsilon', type=float, default=1e-6)
    p.add_argument('--validation-data', default=str(WORKSPACE/'LuLu_outputs/data/dapo_dev256_balanced_s42/dev.jsonl'))
    p.add_argument('--validation-every', type=int, default=4)
    p.add_argument('--gradient-norm-every', type=int, default=4)
    p.add_argument('--rollout-vllm-max-seqs', type=int, default=32)
    p.add_argument('--reference-evaluation', default=str(WORKSPACE/'LuLu_outputs/experiments/ren_stable_full_qwen1p7_teacher32b_pool2048_s42/evaluation_rescored_20260916'))
    p.add_argument('--rollout-backend', choices=('hf','vllm'), default='hf')
    p.add_argument('--eval-rounds', default='', help='Comma-separated intermediate round checkpoints')
    p.add_argument('--rounds', type=int, default=32)
    p.add_argument('--global-batch-prompts', type=int, default=64)
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    p.add_argument('--rollout-batch-size', type=int, default=8)
    p.add_argument('--score-batch-size', type=int, default=1)
    p.add_argument('--train-micro-batch-size', type=int, default=1)
    p.add_argument('--eval-batch-size', type=int, default=16)
    p.add_argument('--eval-decoding', choices=('qwen-thinking', 'greedy'), default='qwen-thinking')
    p.add_argument('--eval-split', choices=('probe', 'full'), default='full')
    p.add_argument('--eval-max-examples', type=int, default=199,
                   help='per-benchmark evaluation cap; 0 means the complete split')
    p.add_argument('--adopt-training-pid', type=int,
                   help='supervise this already-running training process using the saved plan; requires --run')
    p.add_argument('--max-new-tokens', type=int, default=8192)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--save-every', type=int, default=20)
    p.add_argument('--run', action='store_true')
    p.add_argument('--detach', action='store_true',
                   help='run the controller in an independent session with file-backed logs')
    p.add_argument('--poll-seconds', type=float, default=60)
    p.add_argument('--idle-checks', type=int, default=3)
    p.add_argument('--idle-memory-mib', type=int, default=1024)
    p.add_argument('--idle-utilization', type=int, default=5)
    p.add_argument('--wait-for-progress', action='append', default=[],
                   help='Optional existing progress JSON files that must report status=complete before GPU use')
    return p


def make_plan(a, root):
    gpus = [x.strip() for x in a.gpus.split(',') if x.strip()]
    if len(gpus) != 8 or len(set(gpus)) != 8:
        raise ValueError('This experiment requires exactly eight distinct GPU indices')
    for name in ('rounds', 'global_batch_prompts', 'rollout_batch_size', 'score_batch_size',
                 'train_micro_batch_size', 'eval_batch_size', 'max_new_tokens', 'save_every',
                 'idle_checks', 'poll_seconds'):
        if getattr(a, name) <= 0:
            raise ValueError(f'{name} must be positive')
    middle = sorted(set(int(x) for x in a.eval_rounds.split(',') if x.strip()))
    if any(x <= 0 or x >= a.rounds for x in middle):
        raise ValueError('Intermediate checkpoints must be strictly inside training')
    if a.method in ('ren_stable', 'ren_balanced') and a.lora_rank != 0:
        raise ValueError('ren_stable requires --lora-rank 0')
    rate = a.learning_rate if a.learning_rate is not None else (1e-6 if a.method in ('ren_stable', 'ren_balanced') else 1e-5)
    if rate <= 0 or a.reference_kl_coef <= 0 or a.stable_ratio_epsilon <= 0:
        raise ValueError('Learning rate and stable coefficients must be positive')
    if a.lora_rank < 0:
        raise ValueError('lora-rank must be nonnegative')
    if a.eval_max_examples < 0:
        raise ValueError('eval-max-examples must be nonnegative')
    if a.max_new_tokens+4096 > 16384:
        raise ValueError('Response plus 4096 prompt budget must fit 16384 total tokens')
    pool = Path(a.pool_dir).resolve()
    meta = json.loads((pool/'manifest.json').read_text())
    if meta['pool_size'] != 2048 or meta['seed'] != a.seed:
        raise ValueError('Expected the fixed 2048-question pool with matching seed')
    if digest(pool/'train.jsonl') != meta['outputs']['train']['sha256']:
        raise ValueError('Pool bytes disagree with immutable manifest')
    ids = json.loads((pool/'question_ids.json').read_text())
    if len(ids) != 2048 or len(set(ids)) != 2048:
        raise ValueError('Pool must contain exactly 2048 unique IDs')
    if a.model == a.teacher_model:
        raise ValueError('Identical initial Teacher and Student provide no ReN learning signal')
    manifest = Path(a.data_manifest).resolve()
    source = json.loads(manifest.read_text())['benchmarks']
    benchmarks = []
    for name in BENCHMARKS:
        location = Path(source[name][a.eval_split])
        location = (manifest.parent/location).resolve() if not location.is_absolute() else location
        benchmarks.append({'name': name, 'path': str(location), 'sha256': digest(location),
                           'available_examples': source[name][a.eval_split+'_examples'],
                           'examples': min(source[name][a.eval_split+'_examples'], a.eval_max_examples)
                                       if a.eval_max_examples else source[name][a.eval_split+'_examples']})
    validation = None
    baseline = None
    if a.method == 'ren_balanced':
        if middle or a.rounds > 12 or a.validation_every < 1 or a.gradient_norm_every < 1:
            raise ValueError('Balanced run uses <=12 rounds, dev selection, and no intermediate test evaluation')
        dev=Path(a.validation_data).resolve()
        dev_meta=json.loads((dev.parent/'manifest.json').read_text())
        if dev_meta['size']!=256 or digest(dev)!=dev_meta['sha256']:
            raise ValueError('Expected immutable disjoint dev256')
        validation={'path':str(dev),'sha256':digest(dev),'examples':256,'every':a.validation_every,
                    'temperature':0.,'max_new_tokens':a.max_new_tokens,'selection':'highest dev accuracy, earlier round on tie; round0 eligible'}
        source_eval=Path(a.reference_evaluation).resolve()
        correction=json.loads((source_eval/'rescore_audit.json').read_text())
        if correction['parser_sha256']!=digest(PROJECT/'lulu/benchmark_parser.py'):
            raise ValueError('Reference evaluation uses a different parser')
        baseline={'path':str(source_eval),'summary_sha256':digest(source_eval/'summary.json'),
                  'shard_sha256':{str(f.relative_to(source_eval)):digest(f) for f in sorted((source_eval/'base').rglob('shard-*.jsonl'))},
                  'reuse_only':True}
    frozen = root/'code'
    train_dir = root/'train'
    train = [sys.executable, '-u', str(frozen/'scripts/train_lulu.py'),
        '--train-data', str(pool/'train.jsonl'), '--output-dir', str(train_dir),
        '--model', a.model, '--teacher-model', str(Path(a.teacher_model).resolve()),
        '--method', a.method, '--backend', 'persistent', '--gpus', ','.join(gpus),
        '--student-gpus', ','.join(gpus[:5]), '--hindsight-gpus', gpus[5], '--teacher-gpus', ','.join(gpus[6:]),
        '--global-batch-prompts', str(a.global_batch_prompts), '--rollouts-per-prompt', '1',
        '--rounds', str(a.rounds), '--update-passes', '1', '--rollout-batch-size', str(a.rollout_batch_size),
        '--score-batch-size', str(a.score_batch_size), '--train-micro-batch-size', str(a.train_micro_batch_size),
        '--max-new-tokens', str(a.max_new_tokens), '--max-prompt-tokens', '4096', '--max-sequence-tokens', '16384',
        '--logit-chunk-size', '128' if a.method in ('ren_resolved', 'ren_stable', 'ren_balanced') else '32', '--top-k', '32', '--learning-rate', str(rate), '--lora-rank', str(a.lora_rank),
        '--lora-alpha', '32', '--temperature', '.6' if a.method in ('ren_resolved', 'ren_stable', 'ren_balanced') else '1',
        '--top-p', '.95' if a.method in ('ren_resolved', 'ren_stable', 'ren_balanced') else '1', '--seed', str(a.seed),
        '--save-every', str(a.save_every)]
    train += ['--rollout-backend', a.rollout_backend, '--rollout-top-k', '20' if a.method in ('ren_resolved', 'ren_stable', 'ren_balanced') else '0',
              '--rollout-vllm-memory', '.42' if not a.lora_rank else '.35',
              '--retain-checkpoints', ','.join(map(str,sorted(set(middle+([4] if a.method in ('ren_stable', 'ren_balanced') and a.rounds>4 else [])))))]
    if a.method in ('ren_resolved', 'ren_stable', 'ren_balanced'):
        train += ['--worker-timeout', '600']
    if a.method in ('ren_stable', 'ren_balanced'):
        train += ['--teacher-tp-mode', 'eager-local', '--reference-kl-coef', str(a.reference_kl_coef),
                  '--stable-ratio-epsilon', str(a.stable_ratio_epsilon), '--stability-early-stop']
    if a.method == 'ren_balanced':
        retained=sorted(set(range(a.validation_every,a.rounds+1,a.validation_every)))
        train[train.index('--retain-checkpoints')+1]=','.join(map(str,retained))
        train += ['--validation-data',validation['path'],'--validation-every',str(a.validation_every),
                  '--validation-max-examples','256','--gradient-norm-every',str(a.gradient_norm_every),
                  '--rollout-vllm-max-seqs',str(a.rollout_vllm_max_seqs)]
    if not a.lora_rank:
        train.append('--master-weights-fp32')
    evaluate = [sys.executable, '-u', str(frozen/'scripts/evaluate_lulu.py'),
        '--model', a.model, '--include-base', '--checkpoint', f'final={train_dir}/checkpoints/latest',
        '--data-manifest', str(manifest), '--benchmarks', ','.join(BENCHMARKS), '--split', a.eval_split,
        '--gpus', ','.join(gpus), '--batch-size', str(a.eval_batch_size),
        '--max-response-tokens', str(a.max_new_tokens), '--max-prompt-tokens', '4096', '--thinking', '--store-text',
        '--max-examples', str(a.eval_max_examples), '--decoding', a.eval_decoding, '--seed', str(a.seed),
        '--output-dir', str(root/'evaluation')]
    prefix = 'round' if a.method in ('ren_resolved', 'ren_stable', 'ren_balanced') else 'step'
    for index in middle:
        evaluate += ['--checkpoint', f'round{index}={train_dir}/checkpoints/{prefix}_{index:06d}']
    evaluate += ['--backend', 'vllm']
    if a.method == 'ren_balanced':
        evaluate.remove('--include-base')
        evaluate[evaluate.index('--checkpoint')+1]=f'selected={train_dir}/checkpoints/selected'
    return {'schema_version': 1, 'student': a.model, 'teacher': str(Path(a.teacher_model).resolve()),
        **({'stabilization': {'reference_kl_direction': 'student_to_initial_reference',
             'reference_kl_coef': a.reference_kl_coef, 'rho_epsilon': a.stable_ratio_epsilon,
             'weight_normalization': 'none', 'loss_reduction': ('reasoning_prompt_balanced; control/reference_global_token_mean' if a.method=='ren_balanced' else 'global_generated_token_mean'),
             'mask': 'structural_think_plus_answer_and_stop', 'golden_mixture': False,
             'learning_rate': rate, 'early_stop': 'two-round joint length/cap/repetition guard'}} if a.method in ('ren_stable', 'ren_balanced') else {}),
        **({'validation':validation,'reference_evaluation':baseline,
            'budget_allocation':{'reasoning':'mean over reasoning tokens, then rollouts within prompt, then prompts',
                                'weight':'relu(DC-DH)/(1+relu(DC-DH))','weight_renormalization':False,
                                'control_and_reference':'unchanged global generated-token mean',
                                'component_gradient_norm_every':a.gradient_norm_every}} if a.method=='ren_balanced' else {}),
        'method': a.method, 'trainable_parameters': 'full_model' if not a.lora_rank else 'lora',
        'lora_rank': a.lora_rank, 'rollout_backend': a.rollout_backend, 'intermediate_rounds': middle, 'thinking': True, 'pool_manifest': str(pool/'manifest.json'),
        'pool_sha256': meta['outputs']['train']['sha256'], 'pool_size': 2048,
        'rounds': a.rounds, 'prompt_exposures': a.rounds*a.global_batch_prompts,
        'pool_passes': a.rounds*a.global_batch_prompts/2048,
        'global_batch_prompts': a.global_batch_prompts, 'max_new_tokens': a.max_new_tokens,
        'gpu_roles': {'student': gpus[:5], 'hindsight': gpus[5:6], 'teacher_tp': gpus[6:]},
        'data_manifest': str(manifest), 'data_manifest_sha256': digest(manifest),
        'evaluation': {'split': a.eval_split, 'benchmarks': benchmarks,
                       'decoding': a.eval_decoding, 'seed': a.seed,
                       'max_examples_per_benchmark': a.eval_max_examples,
                       'selection': 'first_rows_before_sharding',
                       'examples_per_model': sum(b['examples'] for b in benchmarks), 'models': ['selected'] if a.method=='ren_balanced' else ['base', 'final'] + [f'round{x}' for x in middle]},
        'train_command': train, 'eval_command': evaluate,
        'hf_cache': str(Path(a.hf_cache).resolve()), 'source_root': str(PROJECT),
        'resource_policy': {'gpus': gpus, 'idle_checks': a.idle_checks, 'poll_seconds': a.poll_seconds,
                            'idle_memory_mib': a.idle_memory_mib, 'idle_utilization': a.idle_utilization,
                            'wait_for_progress': [str(Path(p).resolve()) for p in a.wait_for_progress]}}


def freeze_code(root):
    destination = root/'code'
    snapshot = {}
    for directory in ('lulu', 'scripts'):
        for source in sorted((PROJECT/directory).rglob('*.py')):
            relative = source.relative_to(PROJECT)
            snapshot[str(relative)] = digest(source)
            target = destination/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
    return snapshot


def verify_inputs(plan, root):
    if digest(plan['data_manifest']) != plan['data_manifest_sha256']:
        raise ValueError('Benchmark manifest changed')
    if digest(Path(plan['pool_manifest']).parent/'train.jsonl') != plan['pool_sha256']:
        raise ValueError('Frozen training pool changed')
    for benchmark in plan['evaluation']['benchmarks']:
        if digest(benchmark['path']) != benchmark['sha256']:
            raise ValueError(f'Benchmark changed: {benchmark["name"]}')
    if plan.get('validation') and digest(plan['validation']['path'])!=plan['validation']['sha256']:
        raise ValueError('Frozen validation data changed')
    if plan.get('reference_evaluation'):
        baseline=plan['reference_evaluation'];source=Path(baseline['path'])
        if digest(source/'summary.json')!=baseline['summary_sha256']:
            raise ValueError('Frozen reference evaluation summary changed')
        for relative,expected in baseline['shard_sha256'].items():
            if digest(source/relative)!=expected:raise ValueError('Frozen reference generations changed')
    for relative, expected in plan['code_sha256'].items():
        if digest(root/'code'/relative) != expected:
            raise ValueError(f'Queued source snapshot changed: {relative}')


def teacher_ready(path):
    path = Path(path)
    required = ('config.json', 'tokenizer_config.json', 'tokenizer.json', 'model.safetensors.index.json')
    if any(not (path/name).is_file() for name in required):
        return False
    try:
        index = json.loads((path/'model.safetensors.index.json').read_text())
        names = set(index['weight_map'].values())
        if not names or any(not isinstance(name, str) or Path(name).name != name or not name.endswith('.safetensors') for name in names):
            return False
        for name in names:
            shard = path/name
            with shard.open('rb') as stream:
                raw = stream.read(8)
                if len(raw) != 8:
                    return False
                length = struct.unpack('<Q', raw)[0]
                if length > 16*1024*1024:
                    return False
                header = json.loads(stream.read(length))
            size = max(t['data_offsets'][1] for key, t in header.items() if key != '__metadata__')
            if shard.stat().st_size != 8+length+size:
                return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return True


def gpu_snapshot():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
                                      '--format=csv,noheader,nounits'], text=True, timeout=15)
    return {line.split(',')[0].strip(): {'memory_mib': int(line.split(',')[1]),
            'utilization': int(line.split(',')[2])} for line in output.splitlines() if line.strip()}


def resources_idle(snapshot, policy):
    return all(gpu in snapshot and snapshot[gpu]['memory_mib'] <= policy['idle_memory_mib']
               and snapshot[gpu]['utilization'] <= policy['idle_utilization'] for gpu in policy['gpus'])


def progress_complete(policy):
    for filename in policy['wait_for_progress']:
        try:
            if json.loads(Path(filename).read_text()).get('status') != 'complete':
                return False
        except (OSError, ValueError):
            return False
    return True


def status(root, phase, **details):
    record = {'status': phase, 'controller_pid': os.getpid(),
              'updated_at_utc': datetime.now(timezone.utc).isoformat(), **details}
    write_json(root/'live_progress.json', record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def evaluation_progress(plan, root):
    by_model = {}
    for model in plan['evaluation']['models']:
        counts = {}
        for benchmark in plan['evaluation']['benchmarks']:
            ids = set()
            for path in (root/'evaluation'/model/benchmark['name']).glob('shard-*.jsonl'):
                for line in path.read_text().split('\n'):
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue  # A worker may be writing its last line.
                    ids.add(row['prompt_index'])
            counts[benchmark['name']] = len(ids)
        by_model[model] = counts
    return {'evaluation_completed_examples': sum(sum(x.values()) for x in by_model.values()),
            'evaluation_total_examples': plan['evaluation']['examples_per_model']*len(by_model),
            'evaluation_by_model': by_model}


def child_run(command, root, phase, env, plan):
    with (root/'logs'/f'{phase}.log').open('a') as log:
        child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            while child.poll() is None:
                details = {'child_pid': child.pid, 'log': str(root/'logs'/f'{phase}.log')}
                latest = root/'train/latest.json'
                if latest.is_file():
                    state = json.loads(latest.read_text())
                    details['completed_updates'] = state.get('completed_updates')
                    details['completed_rounds'] = state.get('completed_rounds')
                    details['total_rounds'] = plan['rounds']
                    details['total_updates'] = plan['rounds']
                    details['latest_metrics'] = state.get('metrics', [])
                phase_file = root/'train/phase_progress.json'
                if phase == 'training' and phase_file.exists():
                    details['round_progress'] = json.loads(phase_file.read_text())
                if phase == 'evaluation':
                    details.update(evaluation_progress(plan, root))
                status(root, phase, **details)
                time.sleep(15)
            if child.returncode:
                raise RuntimeError(f'{phase} exited with {child.returncode}; see {root}/logs/{phase}.log')
        finally:
            if child.poll() is None or child.returncode != 0:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    pass
                finally:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(child.pid, signal.SIGKILL)
                    child.wait()


def evaluation_plan(plan, root):
    """Apply the explicit evaluation-only override without changing frozen training."""
    effective = json.loads(json.dumps(plan))
    settings = root/'evaluation_settings.json'
    limit = effective['evaluation'].get('max_examples_per_benchmark', 0)
    if settings.exists():
        override = json.loads(settings.read_text())
        if set(override) != {'max_examples_per_benchmark'}:
            raise ValueError('Evaluation settings may only override max_examples_per_benchmark')
        limit = override['max_examples_per_benchmark']
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 0:
        raise ValueError('Evaluation max_examples_per_benchmark must be a nonnegative integer')
    command = effective['eval_command']
    if '--max-examples' in command:
        command[command.index('--max-examples')+1] = str(limit)
    else:
        command.extend(['--max-examples', str(limit)])
    evaluation = effective['evaluation']
    evaluation.update(max_examples_per_benchmark=limit, selection='first_rows_before_sharding')
    for benchmark in evaluation['benchmarks']:
        available = benchmark.setdefault('available_examples', benchmark['examples'])
        benchmark['examples'] = min(available, limit) if limit else available
    evaluation['examples_per_model'] = sum(b['examples'] for b in evaluation['benchmarks'])
    if effective['method'] == 'ren_stable' and (root/'train/early_stop.json').exists():
        state = json.loads((root/'train/checkpoints/latest/lulu_state.json').read_text())
        final_round = state['completed_rounds']
        directory = root/'train/checkpoints'
        available = {int(path.name.split('_')[1]): path for path in directory.glob('round_*')
                     if path.is_dir() and int(path.name.split('_')[1]) < final_round and int(path.name.split('_')[1]) > 0}
        selected = [i for i in effective['intermediate_rounds'] if i in available]
        if len(selected) < min(2,len(available)):
            selected = sorted(set(selected+list(available)))[-2:]
        rebuilt=[]; cursor=0
        while cursor < len(command):
            if command[cursor] == '--checkpoint':
                label = command[cursor+1].split('=',1)[0]
                if label == 'final': rebuilt.extend(command[cursor:cursor+2])
                cursor += 2
            else:
                rebuilt.append(command[cursor]); cursor += 1
        for index in selected:
            rebuilt += ['--checkpoint', f'round{index}={available[index]}']
        effective['eval_command'] = rebuilt
        evaluation.update(models=['base','final']+[f'round{i}' for i in selected],
                          early_stopped=True, completed_rounds=final_round, selected_intermediate_rounds=selected)
    if effective['method']=='ren_balanced':
        selected=json.loads((root/'train/validation/selected.json').read_text())
        path=Path(selected['checkpoint'])
        state=json.loads((path/'lulu_state.json').read_text())
        if state['completed_rounds']!=selected['round']:
            raise ValueError('Dev selection does not identify a retained checkpoint')
        effective['eval_command'][effective['eval_command'].index('--checkpoint')+1]=f'selected={path}'
        evaluation.update(models=['selected'],selected_round=selected['round'],selected_dev_accuracy=selected['accuracy'])
    return effective


def process_identity(pid):
    """Read Linux start time and state; PID reuse must not attach a new job."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    except (FileNotFoundError, ProcessLookupError):
        return None
    return fields[19], fields[0]


def monitor_training(pid, plan, root):
    """Adopt monitoring only; never signal or restart an existing GPU process."""
    if pid <= 0:
        raise ValueError('adopt-training-pid must be positive')
    identity = process_identity(pid)
    if identity is None or identity[1] in ('Z', 'X'):
        raise ValueError('Adopted training process is not running')
    actual = Path(f'/proc/{pid}/cmdline').read_bytes().decode().rstrip('\0').split('\0')
    expected = plan['train_command']
    if actual not in (expected, expected+['--resume']):
        raise ValueError('Adopted process does not match the saved training command')
    while True:
        current = process_identity(pid)
        if current is None or current[1] in ('Z', 'X'):
            return
        if current[0] != identity[0]:
            raise RuntimeError('Adopted training PID was reused; refusing to follow another process')
        latest = root/'train/latest.json'
        completed = json.loads(latest.read_text()).get('completed_updates', 0) if latest.exists() else 0
        status(root, 'training', child_pid=pid, adopted_training=True,
               completed_updates=completed, total_updates=plan['rounds'],
               evaluation_max_examples_per_benchmark=evaluation_plan(plan, root)['evaluation']['max_examples_per_benchmark'],
               log=str(root/'logs/training.log'))
        time.sleep(15)


def training_complete(plan, root):
    state = root/'train/checkpoints/latest/lulu_state.json'
    if not state.is_file():
        return False
    checkpoint = json.loads(state.read_text())
    if plan['method'] in ('ren_stable', 'ren_balanced') and (root/'train/early_stop.json').is_file():
        stopped = json.loads((root/'train/early_stop.json').read_text())
        return (0 < checkpoint.get('completed_rounds', 0) <= plan['rounds']
                and stopped.get('completed_rounds') == checkpoint['completed_rounds']
                and stopped.get('completed_updates') == checkpoint.get('completed_updates'))
    return (checkpoint.get('completed_rounds') == plan['rounds'] and
            (plan['method'] in ('ren_resolved', 'ren_stable', 'ren_balanced') or checkpoint.get('completed_updates') == plan['rounds']))


def launch_detached(command, root):
    """Keep the CPU controller independent of the invoking shell/session."""
    (root/'logs').mkdir(parents=True, exist_ok=True)
    log_path = root/'logs/controller.detached.log'
    with log_path.open('a') as log:
        child = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    write_json(root/'detached_controller.json', {'pid': child.pid, 'command': command,
               'log': str(log_path), 'started_at_utc': datetime.now(timezone.utc).isoformat()})
    print(json.dumps({'controller_pid': child.pid, 'log': str(log_path)}), flush=True)
    return child.pid


def execute(plan, root, adopt_training_pid=None):
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Controller received signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    policy = plan['resource_policy']
    trained = training_complete(plan, root)
    consecutive = 0
    while adopt_training_pid is None:
        snapshot = gpu_snapshot()
        model_complete = trained or teacher_ready(plan['teacher'])
        prior_complete = progress_complete(policy)
        idle = resources_idle(snapshot, policy) and prior_complete
        consecutive = consecutive+1 if idle and model_complete else 0
        status(root, 'waiting_for_resources', teacher_ready=model_complete, gpus=snapshot,
               prior_jobs_complete=prior_complete, consecutive_idle_checks=consecutive,
               required_idle_checks=policy['idle_checks'],
               completed_updates=plan['rounds'] if trained else None, total_updates=plan['rounds'],
               training_complete=trained)
        if consecutive >= policy['idle_checks']:
            break
        time.sleep(policy['poll_seconds'])
    verify_inputs(plan, root)
    env = dict(os.environ)
    env.pop('TRANSFORMERS_CACHE', None)
    env.update(HF_HOME=str(Path(plan['hf_cache']).parent), HF_HUB_CACHE=plan['hf_cache'],
               HUGGINGFACE_HUB_CACHE=plan['hf_cache'], HF_HUB_OFFLINE='1',
               TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='1')
    train = list(plan['train_command'])
    if (root/'train/run_config.json').exists():
        train.append('--resume')
    if not trained:
        if adopt_training_pid is None:
            child_run(train, root, 'training', env, plan)
        else:
            monitor_training(adopt_training_pid, plan, root)
    checkpoint_state = root/'train/checkpoints/latest/lulu_state.json'
    if not checkpoint_state.is_file():
        raise RuntimeError('Training finished without a committed checkpoint')
    completed = json.loads(checkpoint_state.read_text())
    if not training_complete(plan, root):
        raise RuntimeError('Training checkpoint does not contain every planned update')
    verify_inputs(plan, root)
    if adopt_training_pid is not None:
        if not any(line.startswith('Completed: ') for line in (root/'logs/training.log').read_text().split('\n')):
            raise RuntimeError('Adopted training exited without its completion marker')
        while not resources_idle(gpu_snapshot(), policy):
            status(root, 'waiting_for_evaluation_resources', completed_updates=plan['rounds'],
                   total_updates=plan['rounds'])
            time.sleep(policy['poll_seconds'])
    if plan['method'] == 'ren_resolved':
        subprocess.run([sys.executable, str(root/'code/scripts/analyze_resolved_run.py'),
                        '--train-dir', str(root/'train')], env=env, check=True)
    effective = evaluation_plan(plan, root)
    write_json(root/'effective_evaluation_plan.json', effective['evaluation'] | {'command': effective['eval_command']})
    if (root/'evaluation/summary.json').is_file():
        previous = json.loads((root/'evaluation/eval_plan.json').read_text())
        if previous['max_examples'] != effective['evaluation']['max_examples_per_benchmark']:
            raise RuntimeError('Existing evaluation uses a different sample cap; preserve it and select a fresh output')
    if plan['method']=='ren_balanced' and not (root/'evaluation').exists():
        from reuse_initial_evaluation import reuse
        reuse(root, plan, effective)
    if not (root/'evaluation/summary.json').is_file():
        if (root/'evaluation').exists() and any((root/'evaluation').iterdir()):
            raise RuntimeError('Partial evaluation exists; preserve it and select a fresh evaluation output before retry')
        child_run(effective['eval_command'], root, 'evaluation', env, effective)
    if plan['method'] in ('ren_stable', 'ren_balanced'):
        reporter='summarize_balanced_run.py' if plan['method']=='ren_balanced' else 'summarize_stable_run.py'
        subprocess.run([sys.executable,str(root/'code/scripts'/reporter),
                        '--experiment-dir',str(root)],env=env,check=True)
    selection_metadata = {}
    if plan['method']=='ren_balanced':
        chosen=json.loads((root/'train/validation/selected.json').read_text())
        reused=(root/'evaluation/reuse_provenance.json').exists()
        selection_metadata={'selected_round':chosen['round'],'selected_checkpoint':chosen['checkpoint'],
                            'evaluation_reused_initial_model':reused}
    status(root, 'complete', summary=str(root/'evaluation/summary.json'),
           checkpoint=str(root/'train/checkpoints/latest'),
           completed_rounds=completed['completed_rounds'], completed_updates=completed['completed_updates'],
           early_stopped=(root/'train/early_stop.json').exists(), **selection_metadata)


def main():
    a = arguments().parse_args()
    root = Path(a.output_dir).resolve()
    if a.detach:
        if not a.run:
            raise ValueError('--detach requires --run')
        launch_detached([sys.executable, '-u', str(Path(__file__).resolve()),
                         *[arg for arg in sys.argv[1:] if arg != '--detach']], root)
        return
    if a.adopt_training_pid is not None:
        if not a.run or a.eval_max_examples < 0:
            raise ValueError('Adoption requires --run and a nonnegative evaluation cap')
        plan = json.loads((root/'experiment_plan.json').read_text())
        verify_inputs(plan, root)
        print(f'Waiting for controller lock to adopt training PID {a.adopt_training_pid}', flush=True)
    else:
        plan = make_plan(a, root)
    root.mkdir(parents=True, exist_ok=True)
    with (root/'.controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | (0 if a.adopt_training_pid is not None else fcntl.LOCK_NB))
        existing = root/'experiment_plan.json'
        if a.adopt_training_pid is not None:
            write_json(root/'evaluation_settings.json', {'max_examples_per_benchmark': a.eval_max_examples})
            effective = evaluation_plan(plan, root)
            write_json(root/'effective_evaluation_plan.json', effective['evaluation'] | {'command': effective['eval_command']})
        elif existing.exists():
            saved = json.loads(existing.read_text())
            def comparable(value):
                result = {k: v for k, v in value.items() if k not in ('code_sha256', 'evaluation')}
                command = list(result['eval_command'])
                if '--max-examples' in command:
                    offset = command.index('--max-examples')
                    del command[offset:offset+2]
                result['eval_command'] = command
                return result
            if comparable(saved) != comparable(plan):
                raise ValueError('Prepared experiment configuration differs; choose a new output directory')
            plan = saved
            # Evaluation size may change while the training snapshot stays immutable.
            write_json(root/'evaluation_settings.json', {'max_examples_per_benchmark': a.eval_max_examples})
        else:
            plan['code_sha256'] = freeze_code(root)
            write_json(existing, plan)
        (root/'logs').mkdir(exist_ok=True)
        verify_inputs(plan, root)
        if not a.run:
            status(root, 'prepared', plan=str(existing), gpu_jobs_launched=False)
            return
        try:
            execute(plan, root, adopt_training_pid=a.adopt_training_pid)
        except BaseException as error:
            status(root, 'interrupted' if isinstance(error, KeyboardInterrupt) else 'failed', error=str(error))
            raise


if __name__ == '__main__':
    main()
