import importlib.util,json,sys
from pathlib import Path
import numpy as np

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
if str(SCRIPTS) not in sys.path:sys.path.insert(0,str(SCRIPTS))


def load(name):
    spec=importlib.util.spec_from_file_location(name,SCRIPTS/f'{name}.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def bench(acc,n=10):return {'accuracy':acc,'examples':n,'scored_examples':n}

def model(offset=0.):
    return {'benchmarks':{
        'math500':bench(.8+offset,20),'aime25':bench(.4+offset,5),'olympiadbench':bench(.6+offset,20),
        'mmlu_pro':bench(.5+offset,20),'gpqa_diamond':bench(.3+offset,20),'dapo_dev128':bench(.55+offset,8)}}

def evaluation(names):return {'models':{name:model(i*.01) for i,name in enumerate(names)}}


def diag(path,round_=0,applied=.1):
    path.mkdir(parents=True,exist_ok=True)
    d={'round':round_,'causal_kl_mean':.3,'hindsight_kl_mean':.28,'resolved_mean':.02,'resolved_positive_fraction':.4,
       'rho_mean':.01,'applied_weight_mean':applied,'prompt_reasoning_loss_ess':7.,'prompt_reasoning_loss_top10_share':.2,
       'teacher_supervision_retained_fraction':.2,'teacher_supervision_retained_fraction_prompt_balanced':.21,
       'teacher_supervision_retained_fraction_token_global':.19,'prompt_count':8,
       'concentration':{'bounded_weight':{'positions':4,'effective_positions':2.,'positive_fraction':.5,'top_fraction_mass':{'0.01':.3,'0.1':.8}},
                        'applied_weight':{'positions':4,'effective_positions':2.5,'positive_fraction':.75,'top_fraction_mass':{'0.01':.25,'0.1':.7}}}}
    (path/'summary.json').write_text(json.dumps(d))
    np.savez_compressed(path/'position_scores.npz',resolved_mismatch=np.array([-.1,0,.1,.2]),causal_kl=np.array([.1,.2,.3,.4]),bounded_weight=np.array([0,0,.09,.16]),trajectory_index=np.array([0,0,1,1]))


def metric(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps([{'round_seconds':10.,'response_tokens':100,'causal_prompt_tokens':20,'hindsight_prompt_tokens':25,
                                 'teacher_sequence_tokens':120,'hindsight_sequence_tokens':125,'teacher_positions':100,'hindsight_positions':100,
                                 'teacher_seconds':4.,'hindsight_seconds':2.}]))


def test_rq1_summary_emits_grouped_table(tmp_path):
    mod=load('summarize_rq1_baselines');root=tmp_path/'rq1';(root/'evaluation').mkdir(parents=True)
    (root/'evaluation/summary.json').write_text(json.dumps(evaluation(['base','control_ref','vanilla_opd','opsd','ren'])))
    s=mod.summarize(root)
    assert s['rows']==5 and (root/'analysis/main_results.csv').is_file()
    text=(root/'analysis/main_results.csv').read_text();assert 'math_avg' in text and 'external_micro' in text


def test_rq2_allocation_summary_includes_source_ren_diagnostics(tmp_path):
    mod=load('summarize_rq2_allocation');root=tmp_path/'rq2';(root/'evaluation').mkdir(parents=True)
    names=['base','uniform_matched','shuffled_ren','causal_matched','ren'];(root/'evaluation/summary.json').write_text(json.dumps(evaluation(names)))
    source=tmp_path/'source/train';(source/'checkpoints/round_000004').mkdir(parents=True)
    plan={'arms':{'uniform_matched':'uniform','shuffled_ren':'shuffled','causal_matched':'causal_matched'},'source_ren_checkpoint':str(source/'checkpoints/round_000004')}
    (root/'experiment_plan.json').write_text(json.dumps(plan))
    for arm in plan['arms']:diag(root/f'arms/{arm}/train/diagnostics/round_0000')
    diag(source/'diagnostics/round_0000')
    s=mod.summarize(root);assert s['allocation_rows']==4
    assert 'retained_teacher_kl_prompt_balanced' in (root/'analysis/allocation_dynamics.csv').read_text()


def test_teacher_sweep_summary_emits_distribution_and_capability(tmp_path):
    mod=load('summarize_rq2_teacher_sweep');root=tmp_path/'t';(root/'evaluation').mkdir(parents=True);(root/'teacher_evaluation').mkdir()
    arms={'t4_vanilla':{'teacher':'t4','reasoning':'vanilla'},'t4_ren':{'teacher':'t4','reasoning':'ren'},
          't32_vanilla':{'teacher':'t32','reasoning':'vanilla'},'t32_ren':{'teacher':'t32','reasoning':'ren'}}
    (root/'experiment_plan.json').write_text(json.dumps({'arms':arms,'teachers':{'t4':'/m4','t32':'/m32'}}))
    (root/'evaluation/summary.json').write_text(json.dumps(evaluation(['base',*arms])))
    (root/'teacher_evaluation/summary.json').write_text(json.dumps(evaluation(['teacher_t4','teacher_t32'])))
    for arm in arms:
        for r in range(4):diag(root/f'arms/{arm}/train/diagnostics/round_{r:04d}',r);metric(root/f'arms/{arm}/train/metrics/round_{r:04d}.json')
    s=mod.summarize(root);assert s['round_rows']==16 and s['distribution_rows']==16 and s['teacher_capability_rows']==2
    assert 'teacher_math_gap' in (root/'analysis/performance.csv').read_text()


def test_scaling_summary_emits_eight_heatmap_metrics_and_performance_compute(tmp_path):
    mod=load('summarize_rq3_scaling');root=tmp_path/'s';(root/'evaluation_dev').mkdir(parents=True);(root/'evaluation_final').mkdir()
    matrix=[{'name':'ren_n256','arm':'ren','batch':64,'rounds':4,'unique_prompts':256},{'name':'ren_n512','arm':'ren','batch':128,'rounds':4,'unique_prompts':512}]
    (root/'experiment_plan.json').write_text(json.dumps({'matrix':matrix}))
    dev_names=['base']+[f"{row['name']}_r{r}" for row in matrix for r in range(1,5)]
    (root/'evaluation_dev/summary.json').write_text(json.dumps(evaluation(dev_names)))
    (root/'evaluation_final/summary.json').write_text(json.dumps(evaluation(['base']+[r['name'] for r in matrix])))
    for row in matrix:
        for r in range(4):diag(root/f"arms/{row['name']}/train/diagnostics/round_{r:04d}",r);metric(root/f"arms/{row['name']}/train/metrics/round_{r:04d}.json")
    s=mod.summarize(root);assert len(s['heatmap_metrics'])==8 and s['heatmap_rows']==16
    text=(root/'analysis/compute_frontier.csv').read_text();assert 'math_delta' in text and 'teacher_sequence_tokens' in text


def test_policy_drift_runner_builds_one_fixed_prefix_command(tmp_path):
    mod=load('run_rq2_policy_drift');root=tmp_path/'exp';root.mkdir();arms={'t4_ren':{'teacher':'t4','reasoning':'ren'},'t4_vanilla':{'teacher':'t4','reasoning':'vanilla'}}
    (root/'experiment_plan.json').write_text(json.dumps({'arms':arms}))
    train=root/'arms/t4_ren/train';(train/'checkpoints/round_000000').mkdir(parents=True)
    (train/'rollouts/round_0000').mkdir(parents=True);(train/'rollouts/round_0000/shard-000.jsonl').write_text('{}\n')
    for arm in arms:
        for r in range(1,5):(root/f'arms/{arm}/train/checkpoints/round_{r:06d}').mkdir(parents=True)
    cmd,anchor=mod.build_command(root);assert anchor=='t4_ren' and cmd.count('--checkpoint')==8 and cmd.count('--rollout')==1
