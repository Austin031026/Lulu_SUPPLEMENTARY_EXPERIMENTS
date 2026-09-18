"""Regression checks for resource starvation and variable evaluation shard counts."""
from pathlib import Path
import json
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import run_gate_diagnostics as workflow
from summarize_gate_diagnostics import load_evaluation_rows


def plan():
    return {'resource_policy':{'gpus':[str(i) for i in range(8)],'idle_memory_mib':1024,
            'idle_utilization':5,'idle_checks':4,'poll_seconds':15}}


def snapshot(free):
    return {str(i):{'memory_mib':0 if str(i) in free else 13000,
                   'utilization':0 if str(i) in free else 85} for i in range(8)}


def test_training_starts_with_seven_idle_gpus_and_one_busy(monkeypatch,tmp_path):
    free=['0','1','2','3','4','6','7'];messages=[];sleeps=[]
    monkeypatch.setattr(workflow,'gpu_snapshot',lambda:snapshot(free))
    monkeypatch.setattr(workflow,'status',lambda *a,**k:messages.append(k))
    monkeypatch.setattr(workflow.time,'sleep',sleeps.append)
    got=workflow.available_devices(tmp_path,plan(),'training_A_control_ref',7,7)
    assert got==free and len(sleeps)==3
    assert messages[-1]['required_gpus']==7
    assert '5' not in messages[-1]['ready_gpus']


@pytest.mark.parametrize('free',[['3'],['0','1','2','3','4','6','7'],[str(i) for i in range(8)]])
def test_evaluation_uses_available_gpus_without_waiting_for_all_eight(monkeypatch,tmp_path,free):
    monkeypatch.setattr(workflow,'gpu_snapshot',lambda:snapshot(free))
    monkeypatch.setattr(workflow,'status',lambda *a,**k:None)
    monkeypatch.setattr(workflow.time,'sleep',lambda t:None)
    assert workflow.available_devices(tmp_path,plan(),'evaluation_arms',1)==free


def test_device_remap_preserves_global_batch_and_scientific_options():
    cmd=['python','train.py','--model','base','--gpus','0,1,2,3,4,5,6,7',
         '--student-gpus','0,1,2,3,4','--hindsight-gpus','5','--teacher-gpus','6,7',
         '--global-batch-prompts','64','--max-new-tokens','8192','--rounds','4',
         '--learning-rate','1e-6','--reasoning-ablation','ren']
    free=['0','1','2','3','4','6','7']
    actual=workflow.command_for_devices(cmd,free,student_workers=4)
    assert actual[actual.index('--student-gpus')+1]=='0,1,2,3'
    assert actual[actual.index('--hindsight-gpus')+1]=='4'
    assert actual[actual.index('--teacher-gpus')+1]=='6,7'
    for key in ['--model','--global-batch-prompts','--max-new-tokens','--rounds','--learning-rate','--reasoning-ablation']:
        assert actual[actual.index(key)+1]==cmd[cmd.index(key)+1]
    assert cmd[cmd.index('--student-gpus')+1]=='0,1,2,3,4'
    with pytest.raises(ValueError):workflow.command_for_devices(cmd,free,student_workers=5)


def test_report_handles_different_base_and_arm_shard_counts(tmp_path):
    for folder,name,count in [('base_dev','base',7),('evaluation','A_control_ref',8)]:
        suite=tmp_path/folder;suite.mkdir()
        (suite/'eval_plan.json').write_text(json.dumps({'devices':[str(i) for i in range(count)]}))
        dest=suite/name/'dapo_dev128';dest.mkdir(parents=True)
        for sid in range(count):
            rows=[{'prompt_index':i,'generation_seed':42+i} for i in range(sid,23,count)]
            (dest/f'shard-{sid:03d}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    base=load_evaluation_rows(tmp_path,'base','dapo_dev128',23)
    trained=load_evaluation_rows(tmp_path,'A_control_ref','dapo_dev128',23)
    assert base==trained and len(base)==23

@pytest.mark.parametrize('message,expected',[
    ('NCCL ncclSystemError: Call to pthread_join failed: No such process',True),
    ('NCCL ncclRemoteError connection reset by peer',True),
    ("ProcessGroupNCCL's watchdog got stuck for 480 seconds without making progress",True),
    ('ProcessGroupNCCL watchdog caught collective operation timeout',True),
    ("non-finite gradients; ProcessGroupNCCL's watchdog got stuck",False),
    ('Persistent worker closed its connection unexpectedly',False),
    ('CUDA out of memory; NCCL ncclSystemError',False),
    ('non-finite gradients; NCCL ncclSystemError',False),
    ('Resume configuration/data differs; NCCL ncclSystemError',False),
])
def test_retry_requires_explicit_transport_failure(message,expected):
    assert workflow.retryable_training_error(message) is expected


def checkpoint(dest,completed):
    path=dest/'checkpoints'/f'round_{completed:06d}';path.mkdir(parents=True)
    (path/'lulu_state.json').write_text(json.dumps({'completed_rounds':completed,
        'checkpoint_manager':{'owner':'lulu.persistent.v1'}}))
    (path/'optimizer.pt').write_bytes(b'fixture')
    (dest/'checkpoints/latest').symlink_to(path.name)
    return path


def test_resume_archives_only_uncommitted_rounds(tmp_path):
    dest=tmp_path/'train';checkpoint(dest,1)
    (dest/'metrics').mkdir();(dest/'metrics/round_0000.json').write_text('[]')
    for i in range(2):
        path=dest/'rollouts'/f'round_{i:04d}';path.mkdir(parents=True)
        (path/'shard-000.jsonl').write_text('saved data')
    result=workflow.prepare_training_resume(dest,tmp_path/'archive')
    assert result['completed_rounds']==1
    assert (dest/'rollouts/round_0000/shard-000.jsonl').is_file()
    assert not (dest/'rollouts/round_0001').exists()
    assert (tmp_path/'archive/rollouts/round_0001/shard-000.jsonl').read_text()=='saved data'
    assert (dest/'checkpoints/latest/optimizer.pt').is_file()


def test_missing_optimizer_refuses_to_replay_checkpoint(tmp_path):
    dest=tmp_path/'train';path=checkpoint(dest,1);(path/'optimizer.pt').unlink()
    with pytest.raises(ValueError,match='optimizer'):
        workflow.prepare_training_resume(dest,tmp_path/'archive')


def test_completed_arm_is_never_launched(monkeypatch,tmp_path):
    done=tmp_path/'done';done.touch()
    monkeypatch.setattr(workflow,'available_devices',lambda *a:pytest.fail('Completed arm acquired GPUs'))
    workflow.run_gpu_stage(tmp_path,{},[],'training_A_control_ref',done,{})


def fixture_stage(monkeypatch,tmp_path):
    p=plan();p['training']={'student_workers':4};p['recovery_policy']={'training_max_retries':2}
    (tmp_path/'logs').mkdir()
    monkeypatch.setattr(workflow,'available_devices',lambda *a:list('0123467'))
    monkeypatch.setattr(workflow,'status',lambda *a,**kw:None)
    cmd=['python','train.py','--output-dir',str(tmp_path/'train')]
    return p,cmd,tmp_path/'done'


def test_retry_resumes_then_completes(monkeypatch,tmp_path):
    p,cmd,done=fixture_stage(monkeypatch,tmp_path);calls=[]
    def run(root,plan,actual,phase,env):
        calls.append(actual)
        if len(calls)==1:
            dest=tmp_path/'train';checkpoint(dest,0);(dest/'run_config.json').write_text('{}')
            partial=dest/'rollouts/round_0000';partial.mkdir(parents=True)
            (partial/'shard-000.jsonl').write_text('partial')
            (root/'logs'/f'{phase}.log').write_text('NCCL ncclSystemError pthread_join failed\n')
            raise RuntimeError('worker aborted')
        assert '--resume' in actual
        assert not (tmp_path/'train/rollouts/round_0000').exists()
        done.touch()
    monkeypatch.setattr(workflow,'command',run)
    workflow.run_gpu_stage(tmp_path,p,cmd,'training_B_vanilla',done,{})
    assert len(calls)==2
    assert json.loads((tmp_path/'launches/training_B_vanilla.json').read_text())['status']=='complete'


def test_retry_budget_persists_across_controller_restart(monkeypatch,tmp_path):
    p,cmd,done=fixture_stage(monkeypatch,tmp_path);calls=[]
    def run(root,plan,actual,phase,env):
        calls.append(actual)
        with (root/'logs'/f'{phase}.log').open('a') as f:f.write('NCCL ncclSystemError\n')
        raise RuntimeError('worker aborted')
    monkeypatch.setattr(workflow,'command',run)
    with pytest.raises(RuntimeError,match='worker aborted'):
        workflow.run_gpu_stage(tmp_path,p,cmd,'training_B_vanilla',done,{})
    assert len(calls)==3
    with pytest.raises(RuntimeError,match='exhausted'):
        workflow.run_gpu_stage(tmp_path,p,cmd,'training_B_vanilla',done,{})
    assert len(calls)==3


def test_previous_nccl_log_does_not_make_new_unknown_error_retryable(monkeypatch,tmp_path):
    p,cmd,done=fixture_stage(monkeypatch,tmp_path);calls=[]
    (tmp_path/'logs/training_B_vanilla.log').write_text('NCCL ncclSystemError\n')
    def run(root,plan,actual,phase,env):
        calls.append(actual)
        with (root/'logs'/f'{phase}.log').open('a') as f:f.write('Unrecognized dataset format\n')
        raise RuntimeError('invalid data')
    monkeypatch.setattr(workflow,'command',run)
    with pytest.raises(RuntimeError,match='invalid data'):
        workflow.run_gpu_stage(tmp_path,p,cmd,'training_B_vanilla',done,{})
    assert len(calls)==1


def test_resume_allows_placement_only_not_role_size_or_science():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from lulu.persistent import resume_scientific_config
    config=dict(cpu=False,method='ren_balanced',gpus='0,1,2,3,4,5,6',student_gpus='0,1,2,3',
        hindsight_gpus='4',teacher_gpus='5,6',teacher_gpus_per_worker=2,
        train_sha256='frozen_data',learning_rate=1e-6,global_batch_prompts=64)
    remap=dict(config,gpus='0,1,2,3,4,6,7',teacher_gpus='6,7')
    original=resume_scientific_config(config)
    assert resume_scientific_config(remap)==original
    assert resume_scientific_config(dict(remap,student_gpus='0,1,2'))!=original
    assert resume_scientific_config(dict(remap,learning_rate=2e-6))!=original
    assert resume_scientific_config(dict(remap,train_sha256='different'))!=original
    with pytest.raises(ValueError,match='disjoint'):
        resume_scientific_config(dict(remap,teacher_gpus='4,7'))
