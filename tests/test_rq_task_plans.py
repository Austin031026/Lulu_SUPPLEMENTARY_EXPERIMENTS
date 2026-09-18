import json
from pathlib import Path
import sys

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
if str(SCRIPTS) not in sys.path:sys.path.insert(0,str(SCRIPTS))

from run_rq1_baselines import build_plan as build_baselines
from run_rq2_allocation import build_plan as build_allocation
from run_rq2_teacher_sweep import build_plan as build_teachers
from run_rq3_scaling import build_plan as build_scaling


def fake_source(tmp_path):
    source=tmp_path/'source';source.mkdir()
    train=[sys.executable,'-u','/old/scripts/train_lulu.py','--train-data','/data/train.jsonl','--output-dir',str(source/'arms/balanced_recipe/train'),
           '--model','/models/student','--teacher-model','/models/teacher32','--method','ren_balanced','--backend','persistent','--gpus','0,1,2,3,4,5,6,7',
           '--student-gpus','0,1,2,3,4','--hindsight-gpus','5','--teacher-gpus','6,7','--rounds','4','--global-batch-prompts','256',
           '--retain-checkpoints','2,4','--control-loss-coef','0.5','--train-micro-batch-size','1','--update-passes','1','--match-causal-update']
    ev=[sys.executable,'-u','/old/scripts/evaluate_lulu.py','--model','/models/student','--include-base','--checkpoint','round2=/old/r2','--checkpoint','round4=/old/r4',
        '--data-manifest','/data/manifest.json','--benchmarks','math500,aime25,olympiadbench,mmlu_pro,gpqa_diamond,dapo_dev128','--gpus','0,1,2,3,4,5,6,7',
        '--max-examples','199','--max-response-tokens','32768','--max-model-len','40960','--context-safety-margin','128','--thinking','--store-text','--store-token-ids','--output-dir','/old/eval']
    (source/'arms/balanced_recipe/train/checkpoints/round_000004').mkdir(parents=True)
    (source/'experiment_plan.json').write_text(json.dumps({'train_command':train,'eval_command':ev}))
    return source


def flag(cmd,name):return cmd[cmd.index(name)+1]


def test_rq1_baseline_plan_contains_all_matched_baselines(tmp_path):
    source=fake_source(tmp_path);plan=build_baselines(source,tmp_path/'rq1')
    assert set(plan['arms'])=={'control_ref','vanilla_opd','opsd'}
    assert flag(plan['train_commands']['opsd'],'--reasoning-ablation')=='opsd'
    assert flag(plan['train_commands']['vanilla_opd'],'--global-batch-prompts')=='256'
    assert '--store-token-ids' in plan['eval_command']


def test_rq2_plan_has_budget_shape_and_causal_controls(tmp_path):
    source=fake_source(tmp_path);plan=build_allocation(source,tmp_path/'rq2')
    assert plan['arms']=={'uniform_matched':'uniform','shuffled_ren':'shuffled','causal_matched':'causal_matched'}
    for name,arm in plan['arms'].items():assert flag(plan['train_commands'][name],'--reasoning-ablation')==arm


def test_teacher_sweep_retains_every_round_and_two_arms_per_teacher(tmp_path):
    source=fake_source(tmp_path);plan=build_teachers(source,tmp_path/'teachers',[('t4','/m/t4'),('t32','/m/t32')])
    assert len(plan['train_commands'])==4
    for cmd in plan['train_commands'].values():assert flag(cmd,'--retain-checkpoints')=='1,2,3,4'
    assert set(plan['teachers'])=={'t4','t32'}


def test_scaling_plan_fixes_four_updates_and_nested_budget_sizes(tmp_path):
    source=fake_source(tmp_path);plan=build_scaling(source,tmp_path/'scaling',(64,128,256,512),False)
    assert [x['unique_prompts'] for x in plan['matrix']]==[256,512,1024,2048]
    assert len(plan['matrix'])==4
    for row in plan['matrix']:
        cmd=plan['train_commands'][row['name']]
        assert flag(cmd,'--rounds')=='4'
        assert flag(cmd,'--retain-checkpoints')=='1,2,3,4'
