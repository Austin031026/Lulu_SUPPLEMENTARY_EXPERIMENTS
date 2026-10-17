"""Release validation must reject plausible but invalid cost/accuracy tables."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from arcq_cost_validation import ARMS, COUNTS, digest, compare_runs, number
from run_arcq_exp3 import train_command, eval_command
from summarize_arcq_exp3_cost import read_run
from lulu.training import parser, validate_args


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def fixture(root):
    pool = root/'pool.jsonl'
    pool.write_text('{}\n')
    source = root/'code/scripts/train_lulu.py'
    source.parent.mkdir(parents=True)
    source.write_text('# frozen synthetic fixture\n')
    assets = {'student': str(root/'student'), 'teacher':str(root/'teacher'), 'pool':str(pool), 'benchmarks':{}}
    hashes = {str(pool):digest(pool), str(source):digest(source)}
    for name, count in COUNTS.items():
        parquet = root/f'{name}.parquet'
        pq.write_table(pa.Table.from_pylist([{'prompt':[{'role':'user','content':f'q{i}'}], 'reward_model':{'ground_truth':str(i)}} for i in range(count)]), parquet)
        assets['benchmarks'][name] = str(parquet)
        hashes[str(parquet)] = digest(parquet)
    commands = {}
    for label, arm in ARMS.items():
        run = root/'runs'/label
        cmd = train_command(root/'code', assets, run/'train', arm, list('01234567'))
        cfg = vars(parser().parse_args(cmd[3:]))
        validate_args(parser().parse_args(cmd[3:]))
        cfg['train_sha256'] = digest(pool)
        write(run/'train/run_config.json',cfg)
        write(run/'train/runtime_plan.json', {'startup_seconds':1})
        cp = run/'train/checkpoints/round_000004'
        write(cp/'lulu_state.json', {'completed_rounds':4,'completed_updates':4})
        (cp/'model.safetensors').write_bytes(b'fixture')
        gpu = {str(i):{'name':'A100','memory_total_mib':81920} for i in range(8)}
        evcmd = eval_command(root/'code', assets, run/'evaluation', cp, list('01234567'))
        commands[label] = {'train':cmd,'evaluation':evcmd}
        for folder, scope, command in [('telemetry','train',cmd),('evaluation_telemetry','evaluation',evcmd)]:
            write(run/folder/'summary.json', {'status':'complete','exit_code':0,'monitoring_complete':True,'scope':scope,
                'wall_seconds':100,'allocated_gpu_count':8,'gpu_hours':800/3600,'gpus':gpu,'command':command,
                'peak_device_memory_used_mib':2000})
        for index in range(4):
            phase = {'sum_worker_seconds':2.,'max_worker_seconds':1.}
            metrics = {'round':index,'skipped_update':False,'response_tokens':256,'round_seconds':10,
                'pipeline_seconds':5,'update_seconds':2,'weight_scoring_seconds':1,'snapshot_sync_seconds':.1,
                'student_peak_memory_allocated_mib':100,'student_peak_memory_reserved_mib':200,
                'cost_phases':{'collection':{k:phase for k in ('generation_seconds','generation_request_seconds','causal_scoring_seconds')},
                 'update':{'update_seconds':phase,'weight_scoring_seconds':phase},
                 'services':{k:0. for k in ('hindsight_forward_seconds','reference_forward_seconds','teacher_forward_seconds')}}}
            write(run/f'train/metrics/round_{index:04d}.json',[metrics])
            rollout = run/f'train/rollouts/round_{index:04d}/shard-000.jsonl'
            rollout.parent.mkdir(parents=True)
            rollout.write_text(''.join(json.dumps({'index':i,'response_ids':[1]})+'\n' for i in range(256)))
        plan = {'models':[{'name':'final','model':str(cp)}],'decoding':'qwen-thinking','max_examples':512,
                'max_response_tokens':32768,'benchmarks':[{'name':n,'path':p} for n,p in assets['benchmarks'].items()]}
        write(run/'evaluation/eval_plan.json',plan)
        accuracies = {}
        for name, count in COUNTS.items():
            shard = run/'evaluation/final'/name/'shard-000.jsonl'
            shard.parent.mkdir(parents=True)
            shard.write_text(''.join(json.dumps({'prompt_index':i,'ground_truth':str(i),'reward':i%2})+'\n' for i in range(count)))
            accuracies[name] = {'accuracy':.5}
        write(run/'evaluation/summary.json',{'models':{'final':{'benchmarks':accuracies,'macro_accuracy':.5}}})
    write(root/'experiment_plan.json',{'sha256':hashes,'commands':commands})
    return root


def test_complete_three_arm_report(tmp_path):
    fixture(tmp_path)
    rows = [read_run(label,tmp_path/'runs'/label) for label in ARMS]
    compare_runs(rows)
    assert rows[0]['generated_tokens'] == 1024
    assert rows[1]['relative_opd_cost'] == 1
    assert rows[1]['accuracy_gain_pp'] == 0
    script = Path(__file__).resolve().parents[1]/'scripts/summarize_arcq_exp3_cost.py'
    command = [sys.executable,str(script),'--output-dir',str(tmp_path/'analysis')]
    for label in ARMS:
        command += ['--run',f'{label}={tmp_path}/runs/{label}']
    subprocess.run(command,check=True,capture_output=True)
    assert (tmp_path/'analysis/REPORT.md').exists()
    assert len((tmp_path/'analysis/benchmark_accuracy.tsv').read_text().splitlines()) == 16


@pytest.mark.parametrize('fault', ['missing_round','nan','wrong_checkpoint','wrong_answer','duplicate','token_count'])
def test_reject_corrupt_run(tmp_path,fault):
    fixture(tmp_path)
    run = tmp_path/'runs/vanilla_opd'
    if fault == 'missing_round':
        (run/'train/metrics/round_0003.json').unlink()
    elif fault == 'nan':
        write(run/'evaluation/summary.json',{'models':{'final':{'benchmarks':{n:{'accuracy':float('nan')} for n in COUNTS},'macro_accuracy':float('nan')}}})
    elif fault == 'wrong_checkpoint':
        p = run/'evaluation/eval_plan.json'; v=json.loads(p.read_text()); v['models'][0]['model']='wrong'; write(p,v)
    elif fault in ('wrong_answer','duplicate'):
        p=run/'evaluation/final/math500/shard-000.jsonl'
        if fault == 'duplicate': p.write_text(p.read_text()+p.read_text().splitlines()[0]+'\n')
        else: p.write_text(p.read_text().replace('"ground_truth": "0"','"ground_truth": "wrong"'))
    else:
        p=run/'train/metrics/round_0000.json'; v=json.loads(p.read_text()); v[0]['response_tokens']=999; write(p,v)
    with pytest.raises(ValueError):
        read_run('vanilla_opd',run)


def test_telemetry_real_child_and_sampler_failure(tmp_path):
    # Fake the hardware interface, but run the real wrapper and child lifecycle.
    executable=tmp_path/'nvidia-smi'
    executable.write_text(f'#!{sys.executable}\nimport os\nfrom pathlib import Path\np=Path(os.environ["COUNT_FILE"])\nn=int(p.read_text()) if p.exists() else 0\np.write_text(str(n+1))\nif n==1 and os.environ.get("FAIL_SAMPLE"): raise SystemExit(1)\nprint("0, uuid, A100, 81920, 10, 0, 50")\n')
    executable.chmod(0o755)
    wrapper=Path(__file__).resolve().parents[1]/'scripts/run_with_cost_telemetry.py'
    for fail in (False,True):
        out=tmp_path/str(fail)
        env=dict(os.environ,PATH=str(tmp_path)+os.pathsep+os.environ['PATH'],COUNT_FILE=str(tmp_path/f'count{fail}'))
        if fail: env['FAIL_SAMPLE']='1'
        result=subprocess.run([sys.executable,str(wrapper),'--output-dir',str(out),'--label','test','--gpus','0','--poll-seconds','.02','--',sys.executable,'-c','import time; time.sleep(.15)'],env=env,capture_output=True,text=True)
        summary=json.loads((out/'summary.json').read_text())
        assert result.returncode == (2 if fail else 0),result.stderr
        assert summary['monitoring_complete'] is (not fail)
        assert summary['wall_seconds'] > 0


def test_nan_and_boolean_are_not_numbers():
    for value in (float('nan'),float('inf'),True,None,-1):
        with pytest.raises(ValueError): number(value,'test')
