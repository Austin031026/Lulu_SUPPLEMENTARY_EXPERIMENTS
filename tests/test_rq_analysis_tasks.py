import importlib.util
import json
from pathlib import Path
import numpy as np

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'

def load(name):
    spec=importlib.util.spec_from_file_location(name,SCRIPTS/f'{name}.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod


def test_rq2_geometry_analysis_emits_quantiles_and_overlap(tmp_path):
    mod=load('analyze_rq2_geometry')
    root=tmp_path/'train/diagnostics/round_0000';root.mkdir(parents=True)
    np.savez_compressed(root/'position_scores.npz',
        causal_kl=np.array([.1,.4,.2,.9],dtype=np.float32),
        resolved_mismatch=np.array([.05,-.1,.2,.3],dtype=np.float32),
        bounded_weight=np.array([.0476,0,.1667,.2308],dtype=np.float32),
        trajectory_index=np.array([0,0,1,1],dtype=np.int32))
    out=tmp_path/'out';summary=mod.analyze([root/'position_scores.npz'],out)
    assert summary['positions']==4 and (out/'dc_delta_quantiles.csv').is_file() and (out/'topk_overlap.csv').is_file()


def test_inference_budget_analysis_uses_saved_tokens_without_generation():
    mod=load('analyze_inference_budget')
    class Parser:
        @staticmethod
        def extract_answer(text,source):
            return '42' if '</think>' in text and '42' in text else None
        @staticmethod
        def math_equal(a,b):return a==b
    # Decoder exposes a final answer only once the third token is present.
    decoder=lambda ids:'<think>x</think> Answer: 42' if len(ids)>=3 else '<think>x'
    rows=[dict(prompt_index=0,data_source='math',ground_truth='42',response_token_ids=[1,2,3,4])]
    scored,first=mod.analyze_rows(rows,(2,3,4),decoder,Parser)
    assert [x['reward'] for x in scored]==[0.,1.,1.]
    assert first[0]['first_correct_budget']==3
    assert first[0]['first_valid_budget']==3

def test_policy_drift_sampler_uses_saved_prefixes_only(tmp_path):
    mod=load('audit_policy_drift')
    path=tmp_path/'rollouts.jsonl'
    rows=[dict(index=0,source_id='a',causal_prompt_ids=[10,11],response_ids=list(range(20)),reasoning_mask=[True]*20),
          dict(index=1,source_id='b',causal_prompt_ids=[12],response_ids=list(range(10)),reasoning_mask=[i%2==0 for i in range(10)])]
    path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    sampled=mod.sample_records([path],max_trajectories=2,positions_per_bin=3,seed=1,bin_size=8)
    assert len(sampled)==2
    assert all('causal_prompt_ids' in x and 'response_ids' in x for x in sampled)
    assert all(len(x['positions'])<=9 for x in sampled)
