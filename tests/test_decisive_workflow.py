"""CPU-only checks for the immutable decisive experiment and its resource gate."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import struct

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/run_decisive.py'
SPEC = importlib.util.spec_from_file_location('run_decisive', SCRIPT)
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def experiment(tmp_path):
    pool = tmp_path/'pool'
    pool.mkdir()
    ids = [f'problem-{i}' for i in range(2048)]
    (pool/'train.jsonl').write_text(''.join(json.dumps({'id': i, 'messages': [{'role': 'user', 'content': i}],
                                                     'gold_answer': '42'})+'\n' for i in ids))
    _json(pool/'question_ids.json', ids)
    _json(pool/'manifest.json', {'pool_size': 2048, 'seed': 42, 'outputs': {
        'train': {'sha256': workflow.digest(pool/'train.jsonl')},
        'question_ids': {'sha256': workflow.digest(pool/'question_ids.json')}}})
    benchmarks = tmp_path/'benchmarks'
    counts = dict(zip(workflow.BENCHMARKS, (500, 30, 512, 1400, 198)))
    source = {}
    for name, count in counts.items():
        source[name] = {'full': f'{name}.parquet', 'full_examples': count,
                        'probe': f'{name}_probe.parquet', 'probe_examples': min(count, 16)}
        for split in ('full', 'probe'):
            path = benchmarks/source[name][split]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f'tiny fixture for {name} {split}'.encode())
    _json(benchmarks/'manifest.json', {'benchmarks': source})
    root = tmp_path/'experiment'
    root.mkdir()
    args = workflow.arguments().parse_args(['--pool-dir', str(pool), '--output-dir', str(root),
        '--data-manifest', str(benchmarks/'manifest.json'), '--teacher-model', str(tmp_path/'teacher'),
        '--hf-cache', str(tmp_path/'cache/hub')])
    plan = workflow.make_plan(args, root)
    code = root/'code/lulu/source.py'
    code.parent.mkdir(parents=True)
    code.write_text('FROZEN = True\n')
    plan['code_sha256'] = {'lulu/source.py': workflow.digest(code)}
    (root/'logs').mkdir()
    return args, plan, root


def _option(command, name):
    return command[command.index(name)+1]


def test_decisive_plan_covers_fixed_pool_once_with_explicit_5_1_2_roles(experiment):
    args, plan, root = experiment
    assert plan['pool_size'] == plan['prompt_exposures'] == 2048
    assert plan['rounds'] == 32 and plan['pool_passes'] == 1
    train, evaluate = plan['train_command'], plan['eval_command']
    assert _option(train, '--method') == 'ren_opd'
    assert _option(train, '--backend') == 'persistent'
    assert _option(train, '--student-gpus') == '0,1,2,3,4'
    assert _option(train, '--hindsight-gpus') == '5'
    assert _option(train, '--teacher-gpus') == '6,7'
    assert _option(train, '--update-passes') == '1'
    assert _option(train, '--global-batch-prompts') == '64'
    assert _option(train, '--max-new-tokens') == '8192'
    assert _option(train, '--max-sequence-tokens') == '16384'
    assert _option(train, '--save-every') == '20'
    assert _option(evaluate, '--gpus') == '0,1,2,3,4,5,6,7'
    assert _option(evaluate, '--checkpoint') == f'final={root}/train/checkpoints/latest'
    assert '--include-base' in evaluate and '--thinking' in evaluate
    assert _option(evaluate, '--max-response-tokens') == '8192'
    assert plan['evaluation']['models'] == ['base', 'final']
    assert plan['evaluation']['examples_per_model'] == 825
    assert _option(evaluate, '--max-examples') == '199'
    assert [b['examples'] for b in plan['evaluation']['benchmarks']] == [199, 30, 199, 199, 198]
    assert [b['name'] for b in plan['evaluation']['benchmarks']] == list(workflow.BENCHMARKS)


def test_probe_plan_uses_matching_counts_and_files(experiment):
    args, _, root = experiment
    args.eval_split = 'probe'
    plan = workflow.make_plan(args, root)
    assert plan['evaluation']['examples_per_model'] == 80
    assert all(b['path'].endswith('_probe.parquet') for b in plan['evaluation']['benchmarks'])
    assert _option(plan['eval_command'], '--split') == 'probe'


@pytest.mark.parametrize('field,value,match', [
    ('gpus', '0,1,2,3,4,5,6,6', 'eight distinct'),
    ('rounds', 0, 'positive'),
    ('max_new_tokens', 16384, 'fit'),
    ('seed', 7, 'matching seed'),
])
def test_invalid_decisive_plan_rejected_before_execution(experiment, field, value, match):
    args, _, root = experiment
    setattr(args, field, value)
    with pytest.raises(ValueError, match=match):
        workflow.make_plan(args, root)


def _tiny_teacher(path):
    for filename in ('config.json', 'tokenizer_config.json', 'tokenizer.json'):
        _json(path/filename, {})
    _json(path/'model.safetensors.index.json', {'weight_map': {'weight': 'model-00001-of-00001.safetensors'}})
    header = json.dumps({'weight': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
    header += b' ' * (-len(header) % 8)
    shard = path/'model-00001-of-00001.safetensors'
    shard.write_bytes(struct.pack('<Q', len(header))+header+struct.pack('<f', 1.0))
    return shard


def test_teacher_readiness_accepts_complete_tiny_safetensors(tmp_path):
    _tiny_teacher(tmp_path)
    assert workflow.teacher_ready(tmp_path)


@pytest.mark.parametrize('corruption', ['missing_shard', 'short_header', 'short_payload', 'extra_payload',
                                         'broken_json', 'empty_index'])
def test_teacher_readiness_rejects_incomplete_or_invalid_download(tmp_path, corruption):
    shard = _tiny_teacher(tmp_path)
    if corruption == 'missing_shard':
        shard.unlink()
    elif corruption == 'short_header':
        shard.write_bytes(b'1234')
    elif corruption == 'short_payload':
        shard.write_bytes(shard.read_bytes()[:-1])
    elif corruption == 'extra_payload':
        shard.write_bytes(shard.read_bytes()+b'!')
    elif corruption == 'broken_json':
        (tmp_path/'model.safetensors.index.json').write_text('{')
    else:
        _json(tmp_path/'model.safetensors.index.json', {'weight_map': {}})
    assert not workflow.teacher_ready(tmp_path)


def _idle_snapshot():
    return {str(i): {'memory_mib': 0, 'utilization': 0} for i in range(8)}


@pytest.mark.parametrize('change', ['missing', 'memory', 'utilization'])
def test_all_eight_gpus_must_be_idle(experiment, change):
    _, plan, _ = experiment
    snapshot = _idle_snapshot()
    if change == 'missing':
        snapshot.pop('7')
    else:
        snapshot['7']['memory_mib' if change == 'memory' else 'utilization'] = 2048 if change == 'memory' else 90
    assert not workflow.resources_idle(snapshot, plan['resource_policy'])


def _execution_stubs(monkeypatch):
    monkeypatch.setattr(workflow.signal, 'signal', lambda *args: None)
    monkeypatch.setattr(workflow.time, 'sleep', lambda seconds: None)
    monkeypatch.setattr(workflow, 'status', lambda *args, **kwargs: None)


def test_resource_gate_waits_for_weights_prior_jobs_and_consecutive_all_gpu_idle(experiment, monkeypatch):
    _, plan, root = experiment
    _execution_stubs(monkeypatch)
    plan['resource_policy']['idle_checks'] = 2
    progress = root/'prior.json'
    plan['resource_policy']['wait_for_progress'] = [str(progress)]
    # A busy GPU after one ready poll resets the required consecutive streak.
    cases = [(False, True, True), (True, False, True), (True, True, False),
             (True, True, True), (False, True, True), (True, True, True), (True, True, True)]
    current = [-1]
    launched = []
    def snapshot():
        current[0] += 1
        idle, _, prior = cases[current[0]]
        _json(progress, {'status': 'complete' if prior else 'running'})
        result = _idle_snapshot()
        if not idle:
            result['7']['memory_mib'] = 4096
        return result
    monkeypatch.setattr(workflow, 'gpu_snapshot', snapshot)
    monkeypatch.setattr(workflow, 'teacher_ready', lambda path: cases[current[0]][1])
    def child(command, directory, phase, env, saved_plan):
        launched.append((phase, current[0]))
        assert current[0] == 6
        assert 'TRANSFORMERS_CACHE' not in env and env['HF_HUB_OFFLINE'] == '1'
        if phase == 'training':
            _json(root/'train/checkpoints/latest/lulu_state.json', {'completed_updates': 32, 'completed_rounds': 32})
        else:
            _json(root/'evaluation/summary.json', {'models': {}})
    monkeypatch.setattr(workflow, 'child_run', child)
    workflow.execute(plan, root)
    assert launched == [('training', 6), ('evaluation', 6)]


def test_failed_training_prevents_evaluation(experiment, monkeypatch):
    _, plan, root = experiment
    _execution_stubs(monkeypatch)
    plan['resource_policy']['idle_checks'] = 1
    monkeypatch.setattr(workflow, 'gpu_snapshot', _idle_snapshot)
    monkeypatch.setattr(workflow, 'teacher_ready', lambda path: True)
    phases = []
    def fail(command, root, phase, env, plan):
        phases.append(phase)
        raise RuntimeError('training failed intentionally')
    monkeypatch.setattr(workflow, 'child_run', fail)
    with pytest.raises(RuntimeError, match='training failed intentionally'):
        workflow.execute(plan, root)
    assert phases == ['training']


@pytest.mark.parametrize('changed', ['train', 'benchmark', 'code'])
def test_frozen_inputs_reject_changes_before_launch(experiment, changed):
    _, plan, root = experiment
    workflow.verify_inputs(plan, root)
    path = (Path(plan['pool_manifest']).parent/'train.jsonl' if changed == 'train' else
            Path(plan['evaluation']['benchmarks'][0]['path']) if changed == 'benchmark' else
            root/'code/lulu/source.py')
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='changed'):
        workflow.verify_inputs(plan, root)


def test_changed_benchmark_manifest_cannot_redirect_frozen_evaluation(experiment):
    _, plan, root = experiment
    manifest = Path(plan['data_manifest'])
    data = json.loads(manifest.read_text())
    data['benchmarks']['math500']['full'] = 'different.parquet'
    _json(manifest, data)
    with pytest.raises(ValueError, match='manifest changed'):
        workflow.verify_inputs(plan, root)


@pytest.mark.parametrize('completed', [0, 20, 31])
def test_incomplete_checkpoint_prevents_evaluation_even_if_child_exits_successfully(experiment, monkeypatch, completed):
    _, plan, root = experiment
    _execution_stubs(monkeypatch)
    plan['resource_policy']['idle_checks'] = 1
    monkeypatch.setattr(workflow, 'gpu_snapshot', _idle_snapshot)
    monkeypatch.setattr(workflow, 'teacher_ready', lambda path: True)
    phases = []
    def child(command, directory, phase, env, saved_plan):
        phases.append(phase)
        _json(root/'train/checkpoints/latest/lulu_state.json',
              {'completed_updates': completed, 'completed_rounds': completed})
    monkeypatch.setattr(workflow, 'child_run', child)
    with pytest.raises(RuntimeError, match='every planned update'):
        workflow.execute(plan, root)
    assert phases == ['training']


@pytest.mark.parametrize('outcome', ['success', 'failure', 'interrupted'])
def test_child_lifecycle_cleans_its_process_group_on_failure_or_interruption(experiment, monkeypatch, outcome):
    _, plan, root = experiment
    killed, launched = [], []
    class Child:
        pid = 123456789
        returncode = {'success': 0, 'failure': 2, 'interrupted': None}[outcome]
        def poll(self):
            return self.returncode
        def wait(self, timeout=None):
            if self.returncode is None:
                self.returncode = -15
            return self.returncode
    def popen(command, **kwargs):
        launched.append(kwargs)
        return Child()
    def report(*args, **kwargs):
        raise KeyboardInterrupt('simulated controller interruption')
    monkeypatch.setattr(workflow.subprocess, 'Popen', popen)
    monkeypatch.setattr(workflow.os, 'killpg', lambda pid, sig: killed.append((pid, sig)))
    monkeypatch.setattr(workflow, 'status', report)
    if outcome == 'interrupted':
        with pytest.raises(KeyboardInterrupt):
            workflow.child_run(['unused'], root, 'training', {}, plan)
    elif outcome == 'failure':
        with pytest.raises(RuntimeError, match='exited with 2'):
            workflow.child_run(['unused'], root, 'training', {}, plan)
    else:
        workflow.child_run(['unused'], root, 'training', {}, plan)
    assert launched[0]['start_new_session'] is True
    expected = [] if outcome == 'success' else [(Child.pid, workflow.signal.SIGTERM),
                                               (Child.pid, workflow.signal.SIGKILL)]
    assert killed == expected


def test_source_snapshot_is_independent_of_later_checkout_edits(tmp_path, monkeypatch):
    project, root = tmp_path/'project', tmp_path/'run'
    source = project/'lulu/module.py'
    source.parent.mkdir(parents=True)
    source.write_text('ORIGINAL = True\n')
    script = project/'scripts/launch.py'
    script.parent.mkdir()
    script.write_text('print("frozen")\n')
    monkeypatch.setattr(workflow, 'PROJECT', project)
    hashes = workflow.freeze_code(root)
    source.write_text('CHANGED = True\n')
    assert set(hashes) == {'lulu/module.py', 'scripts/launch.py'}
    assert (root/'code/lulu/module.py').read_text() == 'ORIGINAL = True\n'
    assert workflow.digest(root/'code/lulu/module.py') == hashes['lulu/module.py']


def test_evaluation_override_changes_only_eval_and_can_restore_full_split(experiment):
    _, original, root = experiment
    before = json.dumps(original, sort_keys=True)
    for limit, expected in ((50, 230), (199, 825), (0, 2640)):
        _json(root/'evaluation_settings.json', {'max_examples_per_benchmark': limit})
        effective = workflow.evaluation_plan(original, root)
        assert effective['evaluation']['examples_per_model'] == expected
        assert _option(effective['eval_command'], '--max-examples') == str(limit)
        assert effective['train_command'] == original['train_command']
        assert effective['code_sha256'] == original['code_sha256']
    assert json.dumps(original, sort_keys=True) == before


def test_override_supports_original_running_plan_without_explicit_cap(experiment):
    _, plan, root = experiment
    command = plan['eval_command']
    offset = command.index('--max-examples')
    del command[offset:offset+2]
    del plan['evaluation']['max_examples_per_benchmark']
    for benchmark in plan['evaluation']['benchmarks']:
        benchmark['examples'] = benchmark.pop('available_examples')
    _json(root/'evaluation_settings.json', {'max_examples_per_benchmark': 199})
    effective = workflow.evaluation_plan(plan, root)
    assert effective['evaluation']['examples_per_model'] == 825
    assert _option(effective['eval_command'], '--max-examples') == '199'


@pytest.mark.parametrize('override', [{'max_examples_per_benchmark': -1},
                                      {'max_examples_per_benchmark': True},
                                      {'max_examples_per_benchmark': '199'},
                                      {'train_command': []}])
def test_invalid_evaluation_override_is_rejected(experiment, override):
    _, plan, root = experiment
    _json(root/'evaluation_settings.json', override)
    with pytest.raises(ValueError):
        workflow.evaluation_plan(plan, root)


@pytest.mark.parametrize('complete', [False, True])
def test_adopted_training_is_not_restarted_and_incomplete_run_cannot_evaluate(experiment, monkeypatch, complete):
    _, plan, root = experiment
    _execution_stubs(monkeypatch)
    adopted, phases = [], []
    monkeypatch.setattr(workflow, 'gpu_snapshot', _idle_snapshot)
    def monitor(pid, saved, directory):
        adopted.append(pid)
        _json(root/'train/checkpoints/latest/lulu_state.json',
              {'completed_updates': 32 if complete else 6, 'completed_rounds': 32 if complete else 6})
        (root/'logs/training.log').write_text('Completed: final\n' if complete else '')
    monkeypatch.setattr(workflow, 'monitor_training', monitor)
    def child(command, directory, phase, env, effective):
        phases.append(phase)
        assert _option(command, '--max-examples') == '199'
        _json(root/'evaluation/summary.json', {})
    monkeypatch.setattr(workflow, 'child_run', child)
    if complete:
        workflow.execute(plan, root, adopt_training_pid=123)
    else:
        with pytest.raises(RuntimeError, match='every planned update'):
            workflow.execute(plan, root, adopt_training_pid=123)
    assert adopted == [123]
    assert phases == (['evaluation'] if complete else [])


def test_monitor_adopts_real_cpu_process_without_restarting_or_signalling_it(experiment, monkeypatch):
    import subprocess
    import sys
    import time
    _, plan, root = experiment
    command = [sys.executable, '-c', 'import time; time.sleep(0.4)']
    plan['train_command'] = command
    child = subprocess.Popen(command)
    original_sleep = time.sleep
    reports = []
    monkeypatch.setattr(workflow.time, 'sleep', lambda seconds: original_sleep(0.01))
    monkeypatch.setattr(workflow, 'status', lambda *args, **kwargs: reports.append(kwargs))
    try:
        workflow.monitor_training(child.pid, plan, root)
        assert child.wait(timeout=2) == 0
        assert reports and all(r['child_pid'] == child.pid and r['adopted_training'] for r in reports)
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()


def test_monitor_rejects_unrelated_process(experiment):
    import os
    _, plan, root = experiment
    with pytest.raises(ValueError, match='does not match'):
        workflow.monitor_training(os.getpid(), plan, root)


def test_completed_training_goes_directly_to_evaluation_without_teacher_load(experiment, monkeypatch):
    _, plan, root = experiment
    _execution_stubs(monkeypatch)
    plan['resource_policy']['idle_checks'] = 1
    _json(root/'train/checkpoints/latest/lulu_state.json', {'completed_updates': 32, 'completed_rounds': 32})
    monkeypatch.setattr(workflow, 'gpu_snapshot', _idle_snapshot)
    def unexpected_teacher(path):
        raise AssertionError('Finished training must not need Teacher weights')
    monkeypatch.setattr(workflow, 'teacher_ready', unexpected_teacher)
    phases = []
    def child(command, directory, phase, env, effective):
        phases.append(phase)
        assert _option(command, '--max-examples') == '199'
        _json(root/'evaluation/summary.json', {})
    monkeypatch.setattr(workflow, 'child_run', child)
    workflow.execute(plan, root)
    assert phases == ['evaluation']


def test_detached_controller_uses_own_session_and_no_terminal_io(tmp_path, monkeypatch):
    launched = []
    class Child:
        pid = 54321
    def popen(command, **kwargs):
        launched.append((command, kwargs))
        assert kwargs['stdout'].name.endswith('controller.detached.log')
        return Child()
    monkeypatch.setattr(workflow.subprocess, 'Popen', popen)
    command = ['python', 'run_decisive.py', '--run']
    assert workflow.launch_detached(command, tmp_path) == Child.pid
    actual, options = launched[0]
    assert actual == command
    assert options['start_new_session'] is True
    assert options['stdin'] == workflow.subprocess.DEVNULL
    assert options['stderr'] == workflow.subprocess.STDOUT
    assert json.loads((tmp_path/'detached_controller.json').read_text())['pid'] == Child.pid


def test_resolved_full_model_plan_retains_and_evaluates_intermediate_rounds(experiment):
    args, _, root = experiment
    args.method='ren_resolved';args.lora_rank=0;args.rollout_backend='vllm';args.eval_rounds='10,20'
    plan=workflow.make_plan(args,root)
    train,evaluate=plan['train_command'],plan['eval_command']
    assert plan['trainable_parameters']=='full_model'
    assert _option(train,'--lora-rank')=='0'
    assert '--master-weights-fp32' in train
    assert _option(train,'--retain-checkpoints')=='10,20'
    assert _option(train,'--rollout-backend')=='vllm'
    assert _option(train,'--temperature')=='.6'
    assert _option(train,'--top-p')=='.95'
    assert _option(train,'--rollout-top-k')=='20'
    assert _option(evaluate,'--backend')=='vllm'
    assert f'round10={root}/train/checkpoints/round_000010' in evaluate
    assert f'round20={root}/train/checkpoints/round_000020' in evaluate
    assert plan['evaluation']['models']==['base','final','round10','round20']
    path=root/'train/checkpoints/latest/lulu_state.json'
    _json(path,{'completed_rounds':32,'completed_updates':29})
    assert workflow.training_complete(plan,root)


def test_stable_plan_and_early_stop_evaluate_available_checkpoints(experiment):
    args,_,root=experiment
    args.method='ren_stable';args.lora_rank=0;args.rollout_backend='vllm';args.eval_rounds='8,16'
    plan=workflow.make_plan(args,root)
    assert float(_option(plan['train_command'],'--learning-rate'))==1e-6
    assert _option(plan['train_command'],'--reference-kl-coef')=='0.1'
    assert _option(plan['train_command'],'--retain-checkpoints')=='4,8,16'
    assert plan['stabilization']['golden_mixture'] is False
    for index in [0,4,8,9]:
        path=root/f'train/checkpoints/round_{index:06d}';path.mkdir(parents=True)
        _json(path/'lulu_state.json',dict(completed_rounds=index,completed_updates=index))
    (root/'train/checkpoints/latest').symlink_to('round_000009',target_is_directory=True)
    _json(root/'train/early_stop.json',dict(completed_rounds=9,completed_updates=9))
    assert workflow.training_complete(plan,root)
    effective=workflow.evaluation_plan(plan,root)
    assert effective['evaluation']['models']==['base','final','round4','round8']
    assert not any('round_000016' in part for part in effective['eval_command'])
    assert plan['evaluation']['models']==['base','final','round8','round16']
    _json(root/'train/early_stop.json',dict(completed_rounds=8,completed_updates=8))
    assert not workflow.training_complete(plan,root)


def test_balanced_plan_retains_dev_nodes_and_tests_only_dev_selected_model(experiment,tmp_path):
    args,_,root=experiment
    args.method='ren_balanced';args.lora_rank=0;args.rounds=12;args.rollout_backend='vllm'
    dev=tmp_path/'dev/dev.jsonl';_json(dev,{'fixture':True});_json(dev.parent/'manifest.json',{'size':256,'sha256':workflow.digest(dev)})
    args.validation_data=str(dev)
    baseline=tmp_path/'old_eval';_json(baseline/'summary.json',{})
    _json(baseline/'rescore_audit.json',{'parser_sha256':workflow.digest(workflow.PROJECT/'lulu/benchmark_parser.py')})
    args.reference_evaluation=str(baseline)
    plan=workflow.make_plan(args,root)
    assert '--include-base' not in plan['eval_command'] and plan['evaluation']['models']==['selected']
    assert _option(plan['train_command'],'--retain-checkpoints')=='4,8,12'
    assert _option(plan['train_command'],'--gradient-norm-every')=='4'
    selected=root/'train/checkpoints/round_000008';_json(selected/'lulu_state.json',{'completed_rounds':8})
    _json(root/'train/validation/selected.json',{'checkpoint':str(selected),'round':8,'accuracy':.5})
    effective=workflow.evaluation_plan(plan,root)
    assert _option(effective['eval_command'],'--checkpoint')==f'selected={selected}'
    assert effective['evaluation']['models']==['selected'] and effective['evaluation']['selected_round']==8
