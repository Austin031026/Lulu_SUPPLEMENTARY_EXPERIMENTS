"""Freeze two cached audits, fixed-batch gradient angles, then three matched 8k x 4 arms."""
from pathlib import Path
import argparse,contextlib,fcntl,json,os,signal,subprocess,sys,time,hashlib
from run_decisive import WORKSPACE,freeze_code,digest,write_json,gpu_snapshot,status,launch_detached
OLD=WORKSPACE/'LuLu_outputs/experiments/ren_balanced_abs_full_qwen1p7_teacher32b_pool2048_dev256_r12_s42'
LONG=WORKSPACE/'LuLu_outputs/experiments/ren_balanced_abs_full_qwen1p7_teacher32b_pool2048_h16384_r12_s42'
DEFAULT=WORKSPACE/'LuLu_outputs/experiments/ren_gate_diagnostics_8k_r4_s42_20260917'
ARMS={'A_control_ref':'none','B_vanilla':'vanilla','C_ren':'ren'}

def setarg(cmd,key,value):
    if key in cmd:cmd[cmd.index(key)+1]=str(value)
    else:cmd.extend([key,str(value)])

def remove(cmd,key):
    while key in cmd:
        i=cmd.index(key);del cmd[i:i+2]

def prepare(root):
    import pyarrow as pa
    import pyarrow.parquet as pq
    source=json.loads((LONG/'experiment_plan.json').read_text());base=OLD/'train/checkpoints/round_000000';frozen=root/'code'
    root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True);(root/'data').mkdir(exist_ok=True)
    train=Path(source['train_command'][source['train_command'].index('--train-data')+1]);pool=[json.loads(s) for s in train.read_text().splitlines()];assert len(pool)==2048
    dev_source=WORKSPACE/'LuLu_outputs/data/dapo_dev256_balanced_s42/dev.jsonl';dev=[json.loads(s) for s in dev_source.read_text().splitlines()]
    assert not ({r['id'] for r in dev}&{r['id'] for r in pool})
    dev=sorted(dev,key=lambda r:hashlib.sha256(('sampled-dev:20260917:'+r['id']).encode()).hexdigest())[:128]
    (root/'data/dev128.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in dev))
    rows=[dict(prompt=r['messages'],data_source='dapo',reward_model=dict(ground_truth=r['gold_answer'],style='rule'),extra_info=dict(source_id=r['id'])) for r in dev]
    pq.write_table(pa.Table.from_pylist(rows),root/'data/dev128.parquet')
    original=Path(source['eval_command'][source['eval_command'].index('--data-manifest')+1]);manifest=json.loads(original.read_text())
    for name,m in manifest['benchmarks'].items():
        for key in ['full','probe']:
            if key in m:m[key]=str((original.parent/m[key]).resolve())
    manifest['benchmarks']['dapo_dev128']=dict(full=str(root/'data/dev128.parquet'),full_examples=128,scorer='math')
    write_json(root/'data/evaluation_manifest.json',manifest)
    from audit_component_cosines import prepare as cosine_prepare
    cosine_prepare(LONG,root/'gradient_audit')
    commands={}
    for name,variant in ARMS.items():
        cmd=list(source['train_command']);cmd[0]=sys.executable;cmd[2]=str(frozen/'scripts/train_lulu.py')
        for key,value in {'--model':base,'--output-dir':root/'arms'/name/'train','--rounds':4,'--max-new-tokens':8192,'--max-sequence-tokens':16384,
            '--gpus':'0,1,2,3,4,6,7','--student-gpus':'0,1,2,3','--hindsight-gpus':'4','--teacher-gpus':'6,7',
            '--rollout-vllm-memory':.42,'--rollout-vllm-max-seqs':32,'--retain-checkpoints':'4','--reasoning-diagnostic-split':0,'--reasoning-ablation':variant}.items():setarg(cmd,key,value)
        cmd.append('--gradient-cosines');commands[name]=cmd
    evaluate=list(source['eval_command']);evaluate[0]=sys.executable;evaluate[2]=str(frozen/'scripts/evaluate_lulu.py');remove(evaluate,'--checkpoint')
    for name in ARMS:evaluate.extend(['--checkpoint',f'{name}={root}/arms/{name}/train/checkpoints/round_000004'])
    for key,value in {'--data-manifest':root/'data/evaluation_manifest.json','--benchmarks':'math500,aime25,olympiadbench,mmlu_pro,gpqa_diamond,dapo_dev128','--parser-path':frozen/'lulu/thinking_final_parser.py','--output-dir':root/'evaluation'}.items():setarg(evaluate,key,value)
    dev_eval=list(evaluate);remove(dev_eval,'--checkpoint');setarg(dev_eval,'--benchmarks','dapo_dev128');setarg(dev_eval,'--output-dir',root/'base_dev')
    inputs=[train,dev_source,original,LONG/'experiment_plan.json',root/'data/dev128.jsonl',root/'data/dev128.parquet',root/'data/evaluation_manifest.json',root/'gradient_audit/records.jsonl',root/'gradient_audit/manifest.json']
    sem=OLD/'semantic_utility_audit_20260916';inputs += [sem/'manifest.json',sem/'records.jsonl',sem/'state_utility.csv',sem/'pair_results.jsonl',sem/'matched_states.jsonl']
    inputs+=sorted((sem/'scores').glob('*.npz'))+sorted((sem/'wrong_hidden').glob('*.pt'))+sorted((OLD/'certified_correction_audit_20260916/hidden').glob('*.pt'))
    for model in [base,*[LONG/f'train/checkpoints/round_{n:06d}' for n in [0,4,8,12]]]:inputs+=sorted(model.glob('*.safetensors'))+[model/'config.json',model/'lulu_state.json']
    for b in ['math500','aime25','olympiadbench','mmlu_pro','gpqa_diamond']:
        inputs.append(Path(manifest['benchmarks'][b]['full']));inputs.append(LONG/f'reference_32k/base/{b}/rows.jsonl')
    plan=dict(schema_version=1,arms=ARMS,train_commands=commands,eval_command=evaluate,dev_eval_command=dev_eval,
        source_semantic_audit=str(sem),source_16k=str(LONG),base=str(base),rounds=4,
        scope='Cached specificity/utility audits; fixed16-rollout checkpoint gradient audit; exactly three fresh4-round trainings; final-only common32k sampled evaluation',
        training=dict(student_workers=4,horizon=8192,pool=2048,prompts_per_round=64,rounds=4,learning_rate=1e-6,reference_coefficient=.1,control_coefficient=1,reasoning_reduction='reasoning-token mean then rollout then prompt mean',
            vanilla='w=1 on reasoning only, same control/reference and reduction; not legacy vanilla_opd backend',none='w=0 on reasoning only',ren='g/(1+g), no weight renormalization',gradient_norm_and_cosine_updates=[1,4]),
        dev=dict(examples=128,selection='SHA256 sampled-dev:20260917:source_id over preexisting disjoint dev256; frozen before these models/results',ids=[r['id'] for r in dev],sampling=dict(temperature=.6,top_p=.95,top_k=20,seed=42),horizon=32768),
        evaluation=dict(models=list(ARMS),benchmarks=['math500','aime25','olympiadbench','mmlu_pro','gpqa_diamond','dapo_dev128'],examples_per_model=953,checkpoint='round4 fixed a priori; no selection',base='reuse already rescored common32k external predictions; only independent dev128 is newly generated'),
        resource_policy=dict(gpus=[str(i) for i in range(8)],idle_memory_mib=1024,idle_utilization=5,idle_checks=4,poll_seconds=15,training_required_gpus=7,evaluation_min_gpus=1),
        input_sha256={str(p):digest(p) for p in dict.fromkeys(inputs)})
    plan['code_sha256']=freeze_code(root);write_json(root/'experiment_plan.json',plan);return plan

def verify(root,plan):
    for path,sha in plan['input_sha256'].items():
        if digest(path)!=sha:raise ValueError(f'Frozen input changed: {path}')
    for rel,sha in plan['code_sha256'].items():
        if digest(root/'code'/rel)!=sha:raise ValueError(f'Frozen source changed: {rel}')

def command(root,plan,cmd,phase,env):
    with (root/'logs'/f'{phase}.log').open('a') as log:
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:
            while child.poll() is None:
                detail=dict(child_pid=child.pid,log=str(root/'logs'/f'{phase}.log'),arms={name:dict(completed_rounds=0,total_rounds=4) for name in plan.get('arms',ARMS)})
                if '--gpu' in cmd:detail['assigned_gpu']=cmd[cmd.index('--gpu')+1]
                if '--gpus' in cmd:detail['assigned_gpus']=cmd[cmd.index('--gpus')+1]
                for name in plan.get('arms',ARMS):
                    p=root/'arms'/name/'train/latest.json';q=root/'arms'/name/'train/phase_progress.json'
                    if p.exists():
                        value=json.loads(p.read_text());detail['arms'][name]=dict(completed_rounds=value['completed_rounds'],total_rounds=4)
                        if q.exists():detail['arms'][name]['phase']=json.loads(q.read_text())
                audit=root/'gradient_audit/live_progress.json'
                if phase=='gradient_audit' and audit.exists():detail['audit']=json.loads(audit.read_text())
                if phase=='head_audit' and (root/'gap_scores/live_progress.json').exists():detail['audit']=json.loads((root/'gap_scores/live_progress.json').read_text())
                if phase.startswith('evaluation'):
                    folder=Path(cmd[cmd.index('--output-dir')+1])
                    detail['completed_shards']=len(list(folder.glob('timing-*.json')))
                status(root,phase,**detail);time.sleep(15)
            if child.returncode:raise RuntimeError(f'{phase} exited {child.returncode}; see {root}/logs/{phase}.log')
        finally:
            if child.poll() is None or child.returncode:
                with contextlib.suppress(ProcessLookupError):os.killpg(child.pid,signal.SIGTERM)
                # The parent may already be dead while its GPU workers remain.
                # This process group belongs only to our start_new_session child.
                deadline=time.monotonic()+20
                while time.monotonic()<deadline:
                    child.poll()
                    try:os.killpg(child.pid,0)
                    except ProcessLookupError:break
                    time.sleep(.2)
                with contextlib.suppress(ProcessLookupError):os.killpg(child.pid,signal.SIGKILL)
                child.wait()

def available_devices(root,plan,next_stage,minimum,maximum=None):
    """Admit a stage using idle devices, without requiring unrelated GPUs to idle."""
    policy=plan['resource_policy'];maximum=maximum or len(policy['gpus'])
    if not 1 <= minimum <= maximum <= len(policy['gpus']):
        raise ValueError('Invalid stage GPU requirement')
    streak={device:0 for device in policy['gpus']};started=time.monotonic()
    while True:
        snapshot=gpu_snapshot()
        for device in streak:
            free=(device in snapshot and snapshot[device]['memory_mib']<=policy['idle_memory_mib']
                  and snapshot[device]['utilization']<=policy['idle_utilization'])
            streak[device]=streak[device]+1 if free else 0
        ready=[device for device in streak if streak[device]>=policy['idle_checks']]
        status(root,'waiting_for_resources',next_stage=next_stage,gpus=snapshot,
               required_gpus=minimum,ready_gpus=ready,idle_streaks=streak,
               waiting_seconds=time.monotonic()-started)
        if len(ready)>=minimum:return ready[:maximum]
        time.sleep(policy['poll_seconds'])


def command_for_devices(cmd,devices,*,student_workers=None):
    """Change physical placement only; keep global batch/objective/sampling fixed."""
    cmd=list(cmd);setarg(cmd,'--gpus',','.join(devices))
    if student_workers is not None:
        if len(devices)!=student_workers+3 or student_workers<1:
            raise ValueError('Need Student workers + one H/reference + TP2 Teacher')
        setarg(cmd,'--student-gpus',','.join(devices[:student_workers]))
        setarg(cmd,'--hindsight-gpus',devices[student_workers])
        setarg(cmd,'--teacher-gpus',','.join(devices[student_workers+1:]))
    return cmd


def retryable_training_error(log_text):
    """Retry explicit NCCL transport/system failures, never arbitrary worker exits."""
    text = log_text.lower()
    fatal = ('out of memory', 'cuda error: device-side assert', 'illegal memory access',
             'non-finite', 'nonfinite', 'nan loss', 'resume configuration/data differs',
             'stale student rollout', 'missing teacher target', 'no space left on device')
    if any(message in text for message in fatal):
        return False
    return ('nccl' in text and any(message in text for message in
            ('ncclsystemerror', 'call to pthread_join failed', 'ncclremoteerror',
             'connection reset by peer', 'socketstartconnect',
             "processgroupnccl's watchdog got stuck", 'watchdog caught collective operation timeout')))


def prepare_training_resume(destination, archive):
    """Preserve partial round diagnostics; only a committed checkpoint advances training."""
    link = destination/'checkpoints/latest'
    completed = 0
    checkpoint = None
    if link.is_symlink():
        checkpoint = link.resolve(strict=True)
        state = json.loads((checkpoint/'lulu_state.json').read_text())
        completed = state['completed_rounds']
        if (checkpoint.parent != (destination/'checkpoints').resolve()
                or checkpoint.name != f'round_{completed:06d}'
                or state.get('checkpoint_manager',{}).get('owner') != 'lulu.persistent.v1'
                or not (checkpoint/'optimizer.pt').is_file()):
            raise ValueError('Invalid resume checkpoint or missing optimizer state')
        # A committed update must never be replayed merely to regenerate a report.
        if completed and not (destination/'metrics'/f'round_{completed-1:04d}.json').exists():
            raise RuntimeError('Committed checkpoint has unfinished round reporting; inspect before resume')
    elif link.exists():
        raise ValueError('Expected managed latest checkpoint symlink')
    moved = []
    for folder in ('rollouts','diagnostics','metrics'):
        for path in sorted((destination/folder).glob('round_*')):
            suffix = path.stem.removeprefix('round_')
            if suffix.isdigit() and int(suffix) >= completed:
                target = archive/folder/path.name
                target.parent.mkdir(parents=True,exist_ok=True)
                path.rename(target)
                moved.append(str(target))
    phase = destination/'phase_progress.json'
    if phase.exists():
        archive.mkdir(parents=True,exist_ok=True)
        phase.rename(archive/'phase_progress.json')
    return dict(completed_rounds=completed,checkpoint=str(checkpoint) if checkpoint else None,
                archived_partial_outputs=moved)


def run_gpu_stage(root,plan,cmd,phase,done,env):
    if done.exists():
        return
    training = phase.startswith('training_')
    student_workers = plan['training'].get('student_workers',4) if training else None
    required = student_workers+3 if training else plan['resource_policy'].get('evaluation_min_gpus',1)
    launches = root/'launches';launches.mkdir(exist_ok=True)
    max_attempts = 1 + (plan.get('recovery_policy',{}).get('training_max_retries',2) if training else 0)
    prior = sorted(launches.glob(f'{phase}.attempt-*.json'))
    # Persist the bound across controller restarts, including manually adopted failures.
    attempt = max([json.loads(p.read_text())['attempt'] for p in prior],default=0)
    while attempt < max_attempts:
        attempt += 1
        actual = list(cmd)
        recovery = None
        if training:
            destination = Path(actual[actual.index('--output-dir')+1])
            if (destination/'run_config.json').exists():
                recovery = prepare_training_resume(destination,root/'recovery'/phase/f'attempt-{attempt:03d}')
                if '--resume' not in actual:actual.append('--resume')
        devices = available_devices(root,plan,phase,required,required if training else None)
        actual = command_for_devices(actual,devices,student_workers=student_workers)
        logfile = root/'logs'/f'{phase}.log'
        offset = logfile.stat().st_size if logfile.exists() else 0
        record = dict(phase=phase,attempt=attempt,max_attempts=max_attempts,devices=devices,
                      command=actual,student_workers=student_workers,started_unix=time.time(),
                      status='running',log_offset=offset,resume=recovery,scientific_settings_unchanged=True)
        record_path = launches/f'{phase}.attempt-{attempt:03d}.json'
        def save():
            write_json(record_path,record);write_json(launches/f'{phase}.json',record)
        save()
        try:
            command(root,plan,actual,phase,env)
            if not done.exists():raise RuntimeError(f'Missing completed output for {phase}: {done}')
        except RuntimeError as exc:
            with logfile.open('rb') as stream:
                stream.seek(offset);error_text=stream.read().decode(errors='replace')
            retry = training and attempt < max_attempts and retryable_training_error(error_text)
            record.update(status='retry_pending' if retry else 'failed',error=str(exc),
                          finished_unix=time.time(),retryable=retryable_training_error(error_text))
            save()
            if not retry:raise
            status(root,'recovering_training',next_stage=phase,attempt=attempt,
                   max_attempts=max_attempts,error=str(exc),resume='latest committed checkpoint')
            continue
        record.update(status='complete',finished_unix=time.time());save()
        return
    raise RuntimeError(f'{phase} exhausted its {max_attempts} permitted attempts; inspect failure logs')


def execute(root,plan):
    status(root,'verifying_frozen_inputs',next_stage='resume_pending_stages')
    verify(root,plan);env=dict(os.environ,HF_HUB_OFFLINE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',TOKENIZERS_PARALLELISM='false',VLLM_DISABLE_COMPILE_CACHE='1',HF_HOME=str(WORKSPACE.parent/'huggingface_cache'),HF_HUB_CACHE=str(WORKSPACE.parent/'huggingface_cache/hub'),HUGGINGFACE_HUB_CACHE=str(WORKSPACE.parent/'huggingface_cache/hub'));env.pop('TRANSFORMERS_CACHE',None)
    py=sys.executable;scripts=root/'code/scripts';sem=plan['source_semantic_audit']
    env.setdefault('NCCL_DEBUG','INFO')
    def gpu(cmd,phase,done):
        run_gpu_stage(root,plan,cmd,phase,done,env)
    if not (root/'gap_scores/summary.json').exists():
        policy=plan['resource_policy'];streak={device:0 for device in policy['gpus']}
        while True:
            snapshot=gpu_snapshot()
            for device in streak:
                free=device in snapshot and snapshot[device]['memory_mib']<=policy['idle_memory_mib'] and snapshot[device]['utilization']<=policy['idle_utilization']
                streak[device]=streak[device]+1 if free else 0
            ready=[device for device in streak if streak[device]>=policy['idle_checks']]
            status(root,'waiting_for_head_gpu',gpus=snapshot,idle_streaks=streak,required_gpus=1)
            if ready:break
            time.sleep(policy['poll_seconds'])
        command(root,plan,[py,str(scripts/'rescore_gate_heads.py'),'--source',sem,'--output-dir',str(root/'gap_scores'),'--gpu',ready[0]],'head_audit',env)
        if not (root/'gap_scores/summary.json').exists():raise RuntimeError('Head audit did not complete')
    for precision in ['bf16','fp32']:
        dest=root/f'audit_{precision}_signed'
        if not (dest/'results.json').exists():command(root,plan,[py,str(scripts/'audit_current_gate.py'),'--source',sem,'--scores',str(root/'gap_scores'/precision),'--precision',precision+' head, signed gaps on existing hidden cache','--output-dir',str(dest)],'analysis_'+precision,env)
    if not (root/'gradient_audit/summary.json').exists():
        # The fixed-batch audit needs four independent students, one reference,
        # and TP2 Teacher. It can proceed while the eighth GPU finishes other work.
        policy=plan['resource_policy'];streak={device:0 for device in policy['gpus']}
        while True:
            snapshot=gpu_snapshot()
            for device in streak:
                free=device in snapshot and snapshot[device]['memory_mib']<=policy['idle_memory_mib'] and snapshot[device]['utilization']<=policy['idle_utilization']
                streak[device]=streak[device]+1 if free else 0
            ready=[device for device in streak if streak[device]>=policy['idle_checks']]
            status(root,'waiting_for_gradient_gpus',next_stage='gradient_audit',gpus=snapshot,idle_streaks=streak,required_gpus=7)
            if len(ready)>=7:break
            time.sleep(policy['poll_seconds'])
        command(root,plan,[py,str(scripts/'audit_component_cosines.py'),'--output-dir',str(root/'gradient_audit'),'--gpus',','.join(ready[:7]),'--run'],'gradient_audit',env)
        if not (root/'gradient_audit/summary.json').exists():raise RuntimeError('Gradient audit did not complete')
    for name in ARMS:
        dest=root/'arms'/name/'train';done=dest/'checkpoints/round_000004/lulu_state.json';cmd=list(plan['train_commands'][name])
        if (dest/'run_config.json').exists() and not done.exists():cmd.append('--resume')
        gpu(cmd,'training_'+name,done)
        if json.loads(done.read_text())['completed_rounds']!=4:raise RuntimeError('Incomplete arm')
    verify(root,plan)
    for folder in ['base_dev','evaluation']:
        if (root/folder).exists() and not (root/folder/'summary.json').exists():
            raise RuntimeError(f'Partial {folder} preserved; inspect before resuming generation')
    gpu(plan['dev_eval_command'],'evaluation_base_dev',root/'base_dev/summary.json')
    gpu(plan['eval_command'],'evaluation_arms',root/'evaluation/summary.json')
    command(root,plan,[py,str(scripts/'summarize_gate_diagnostics.py'),'--experiment-dir',str(root)],'reporting',env)
    status(root,'complete',report=str(root/'REPORT.md'),arms={n:4 for n in ARMS},summary=str(root/'comparison.json'))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',default=str(DEFAULT));p.add_argument('--run',action='store_true');p.add_argument('--detach',action='store_true');a=p.parse_args();root=Path(a.output_dir).resolve();root.mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
    if a.detach:
        if not a.run:raise ValueError('--detach needs --run')
        launch_detached([sys.executable,'-u',str(Path(__file__).resolve()),'--output-dir',str(root),'--run'],root);return
    def interrupted(signum,frame):raise KeyboardInterrupt(f'Controller signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    with (root/'.controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=json.loads((root/'experiment_plan.json').read_text()) if (root/'experiment_plan.json').exists() else prepare(root)
        if not a.run:status(root,'prepared',gpu_jobs_launched=False);return
        try:execute(root,plan)
        except BaseException as exc:status(root,'failed',error=str(exc));raise
if __name__=='__main__':main()
