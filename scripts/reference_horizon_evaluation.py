"""CPU-only, provenance-checked 32k prefix rescoring of existing generations."""
from __future__ import annotations
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lulu import thinking_final_parser as final_parser
from lulu import benchmark_parser as legacy
from lulu.vllm_evaluation import build_jobs
from evaluate_lulu import summarize_rows
from run_decisive import write_json,digest


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment-dir',required=True);x=p.parse_args();root=Path(x.experiment_dir)
    plan=json.loads((root/'experiment_plan.json').read_text());source=Path(plan['reference_generations'])
    original=json.loads((source/'eval_plan.json').read_text())
    assert original['sampling']==dict(temperature=.6,top_p=.95,top_k=20,seed=42)
    assert original['max_response_tokens']>=32768 and original['thinking'] and original['store_token_ids']
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(plan['model'],local_files_only=True)
    evaluation=dict(original,max_response_tokens=32768,max_model_len=40960,context_safety_margin=128)
    expected={}
    for shard in range(len(evaluation['devices'])):
        for job in build_jobs(evaluation,shard,tok):
            key=(job['benchmark'],job['index']);expected[key]=job
    assert len(expected)==825
    mapping={'base':'base','round8':'train8k_round8','final':'train8k_final12'};summary={};corrections={}
    dest=root/'reference_32k'
    if dest.exists() and any(dest.iterdir()):raise ValueError('Refusing to overwrite partially rescored references')
    for old,label in mapping.items():
        summary[label]={};corrections[label]={}
        for benchmark in evaluation['benchmarks']:
            name=benchmark['name'];raw=[]
            for file in sorted((source/old/name).glob('shard-*.jsonl')):raw.extend(json.loads(s) for s in file.read_text().splitlines() if s)
            raw.sort(key=lambda r:r['prompt_index'])
            assert [r['prompt_index'] for r in raw]==list(range(benchmark['expected_examples']))
            results=[];changed=0
            for row in raw:
                key=(name,row['prompt_index']);job=expected[key]
                assert row['prompt_sha256']==hashlib.sha256(json.dumps(job['prompt_token_ids'],separators=(',',':')).encode()).hexdigest()
                seed=int.from_bytes(hashlib.sha256(f"42:{name}:{row['prompt_index']}".encode()).digest()[:4],'big')
                assert row['generation_seed']==seed
                assert row['ground_truth']==job['row']['reward_model']['ground_truth']
                budget=job['max_response_tokens'];ids=row['response_token_ids'][:budget]
                assert row['max_response_tokens']>=budget
                clipped=len(row['response_token_ids'])>budget
                text=tok.decode(ids,skip_special_tokens=True)
                pred=final_parser.extract_answer(text,row['data_source']);old_pred=legacy.extract_answer(text,row['data_source'])
                reward=float(final_parser.math_equal(pred,row['ground_truth']));old_reward=float(legacy.math_equal(old_pred,row['ground_truth']))
                changed+=reward!=old_reward
                result=dict(row,response_token_ids=ids,response=text,response_tokens=len(ids),max_response_tokens=budget,
                    hit_cap=bool(clipped or row['hit_cap']),finish_reason='length' if clipped else row['finish_reason'],
                    prediction=pred,reward=reward,legacy_prediction=old_pred,legacy_reward=old_reward,
                    source_model=old,cached_generation=True,original_response_tokens=row['response_tokens'],original_budget=row['max_response_tokens'])
                results.append(result)
            path=dest/label/name/'rows.jsonl';path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in results))
            summary[label][name]=summarize_rows(results);corrections[label][name]=changed
    write_json(dest/'summary.json',summary)
    write_json(dest/'provenance.json',dict(source=str(source),source_models=mapping,source_plan_sha256=digest(source/'eval_plan.json'),
        source_shard_sha256={str(f.relative_to(source)):digest(f) for f in sorted(source.glob('*/*/shard-*.jsonl'))},
        prompt_hash_seed_and_gold_verified=True,horizon=32768,parser='thinking_final_parser',
        parser_changes_by_model=corrections,gpu_generations=0,
        interpretation='Cached sampled-policy prefixes, rescored with one frozen parser; not newly sampled replicates, not claimed bitwise equivalent to rerunning vLLM with a different maximum length'))
    print(json.dumps({'status':'complete','models':list(summary),'examples_per_model':825,'score_changes':corrections},indent=2))
if __name__=='__main__':main()
