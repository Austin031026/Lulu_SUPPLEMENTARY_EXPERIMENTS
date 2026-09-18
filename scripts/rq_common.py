"""Shared plan helpers for the paper RQ experiment suites.

The helpers are intentionally side-effect free except ``freeze_current_code``;
unit tests can construct commands from a tiny synthetic source plan without
models or GPUs.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]


def setarg(command, flag, value):
    command=list(command)
    if flag in command:
        command[command.index(flag)+1]=str(value)
    else:
        command.extend([flag,str(value)])
    return command


def remove_flag(command, flag, *, takes_value=True):
    command=list(command)
    while flag in command:
        i=command.index(flag)
        del command[i:i+(2 if takes_value else 1)]
    return command


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    tmp.replace(path)


def freeze_current_code(destination):
    destination=Path(destination)
    hashes={}
    for folder in ('lulu','scripts'):
        for source in sorted((ROOT/folder).rglob('*.py')):
            relative=source.relative_to(ROOT);target=destination/relative
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
            hashes[str(relative)]=digest(source)
    return hashes


def load_source_plan(source_experiment):
    source=Path(source_experiment).expanduser().resolve()
    path=source/'experiment_plan.json'
    if not path.is_file(): raise FileNotFoundError(path)
    return source,json.loads(path.read_text())


def train_command(source_plan, *, code_root, output_dir, arm, batch=None, rounds=4,
                  teacher_model=None, model=None, retain='4', control=None):
    cmd=list(source_plan['train_command']);cmd[0]=sys.executable
    cmd[2]=str(Path(code_root)/'scripts/train_lulu.py')
    for flag,value in {'--output-dir':output_dir,'--reasoning-ablation':arm,'--rounds':rounds,
                       '--retain-checkpoints':retain}.items():
        cmd=setarg(cmd,flag,value)
    if batch is not None:cmd=setarg(cmd,'--global-batch-prompts',batch)
    if teacher_model is not None:cmd=setarg(cmd,'--teacher-model',teacher_model)
    if model is not None:cmd=setarg(cmd,'--model',model)
    if control is not None:cmd=setarg(cmd,'--control-loss-coef',control)
    # Every paper arm uses the corrected score/live path and the current stable backbone.
    if '--match-causal-update' not in cmd:cmd.append('--match-causal-update')
    return cmd


def eval_command(source_plan, *, code_root, output_dir, checkpoints, include_base=True,
                 benchmarks=None, max_examples=None, model=None):
    cmd=list(source_plan['eval_command']);cmd[0]=sys.executable
    cmd[2]=str(Path(code_root)/'scripts/evaluate_lulu.py')
    cmd=remove_flag(cmd,'--checkpoint')
    cmd=remove_flag(cmd,'--include-base',takes_value=False)
    if include_base:cmd.append('--include-base')
    for name,path in checkpoints:
        cmd += ['--checkpoint',f'{name}={path}']
    cmd=setarg(cmd,'--output-dir',output_dir)
    cmd=setarg(cmd,'--parser-path',Path(code_root)/'lulu/thinking_final_parser.py')
    if benchmarks is not None:cmd=setarg(cmd,'--benchmarks',benchmarks)
    if max_examples is not None:cmd=setarg(cmd,'--max-examples',max_examples)
    if model is not None:cmd=setarg(cmd,'--model',model)
    # Paper analyses rely on exact prefix rescoring later.
    if '--store-token-ids' not in cmd:cmd.append('--store-token-ids')
    if '--store-text' not in cmd:cmd.append('--store-text')
    return cmd



def student_worker_count(command):
    """Infer the number of DDP Student workers from a frozen train command."""
    command=list(command)
    if '--student-gpus' not in command:
        raise ValueError('source train command must specify --student-gpus')
    value=str(command[command.index('--student-gpus')+1])
    devices=[x for x in value.split(',') if x.strip()]
    if not devices:
        raise ValueError('--student-gpus must contain at least one device')
    return len(devices)

def checkpoint(root, arm, round_index=4):
    return Path(root)/'arms'/arm/'train'/'checkpoints'/f'round_{round_index:06d}'


def validate_matrix(matrix):
    names=[row['name'] for row in matrix]
    if not names or len(names)!=len(set(names)):raise ValueError('Experiment arm names must be nonempty and unique')
    for row in matrix:
        if int(row.get('rounds',4))<1 or int(row.get('batch',1))<1:raise ValueError('rounds/batch must be positive')
    return matrix


def lightweight_provenance(source_experiment, source_plan, extra_models=()):
    """Hash small scientific inputs without re-reading multi-GB weight shards.

    The source experiment already freezes the successful recipe's large inputs;
    new model paths contribute config/tokenizer/index hashes so a later run can
    detect accidental model-directory changes cheaply.
    """
    files=[Path(source_experiment)/'experiment_plan.json']
    for cmd,flag in ((source_plan.get('train_command',[]),'--train-data'),(source_plan.get('eval_command',[]),'--data-manifest')):
        if flag in cmd:
            files.append(Path(cmd[cmd.index(flag)+1]))
    models=[]
    train=source_plan.get('train_command',[])
    for flag in ('--model','--teacher-model'):
        if flag in train:models.append(Path(train[train.index(flag)+1]))
    models.extend(Path(x) for x in extra_models)
    for model in models:
        if model.is_dir():
            for name in ('config.json','generation_config.json','tokenizer_config.json','tokenizer.json','model.safetensors.index.json'):
                path=model/name
                if path.is_file():files.append(path)
    result={}
    for path in dict.fromkeys(files):
        if Path(path).is_file():result[str(Path(path).resolve())]=digest(path)
    return result
