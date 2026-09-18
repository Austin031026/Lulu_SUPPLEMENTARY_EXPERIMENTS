"""Report fixed Round2/4 at 32k plus CPU-only 8k prefixes under one final parser."""
from pathlib import Path
import argparse,json,sys,multiprocessing as mp
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluate_lulu import load_complete_rows,summarize_rows,paired_comparison
from lulu import thinking_final_parser as parser
from lulu.training import atomic_json


def init_tokenizer(model):
    global TOKENIZER
    from transformers import AutoTokenizer
    TOKENIZER=AutoTokenizer.from_pretrained(model,local_files_only=True)


def prefix_rows(job):
    model,benchmark,rows=job;result=[]
    for row in rows:
        clipped=len(row['response_token_ids'])>8192
        text=TOKENIZER.decode(row['response_token_ids'][:8192],skip_special_tokens=True)
        pred=parser.extract_answer(text,row['data_source']);reward=float(parser.math_equal(pred,row['ground_truth']))
        value={k:row[k] for k in ('prompt_index','data_source','prompt_sha256','generation_seed','ground_truth','prompt_tokens')}
        value.update(prediction=pred,reward=reward,response_tokens=min(len(row['response_token_ids']),8192),
            hit_cap=bool(clipped or row['hit_cap']),finish_reason='length' if clipped else row['finish_reason'],
            max_response_tokens=8192,prefix_rescored=True)
        result.append(value)
    return model,benchmark,result


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment-dir',required=True);a=p.parse_args();root=Path(a.experiment_dir)
    plan=json.loads((root/'experiment_plan.json').read_text());ep=json.loads((root/'evaluation/eval_plan.json').read_text())
    models=['base','round2','round4'];rows={32768:{},8192:{}};metrics=[]
    for model in models:
        rows[32768][model]={};rows[8192][model]={}
        for b in ep['benchmarks']:
            rr=load_complete_rows(root/'evaluation'/model/b['name'],len(ep['devices']),b['expected_examples'])
            if model!='base':
                for x,y in zip(rows[32768]['base'][b['name']],rr):
                    if any(x[k]!=y[k] for k in ('prompt_index','prompt_sha256','generation_seed','ground_truth')):
                        raise ValueError('Fresh Base/checkpoint evaluation pairing mismatch')
            rows[32768][model][b['name']]=rr
    jobs=[(m,b,rr) for m,sets in rows[32768].items() for b,rr in sets.items()]
    base=plan['train_command'][plan['train_command'].index('--model')+1]
    with mp.get_context('spawn').Pool(4,initializer=init_tokenizer,initargs=(base,)) as pool:
        for m,b,rr in pool.imap_unordered(prefix_rows,jobs):
            rows[8192][m][b]=rr;dest=root/'prefix_8192'/m/b;dest.mkdir(parents=True,exist_ok=True)
            (dest/'rows.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rr))
    external=[b['name'] for b in ep['benchmarks'] if b['name']!='dapo_dev128'];summaries={};paired={};rng=np.random.default_rng(20260918)
    for horizon,models_rows in rows.items():
        summaries[horizon]={m:{b:summarize_rows(rr) for b,rr in sets.items()} for m,sets in models_rows.items()}
        paired[horizon]={}
        for model in models[1:]:
            individual={};draws=[]
            for b in external:
                value=paired_comparison(models_rows['base'][b],models_rows[model][b]);individual[b]=value
                n=value['paired_examples'];up,down=value['rescues'],value['degradations']
                z=rng.multinomial(n,[up/n,down/n,1-(up+down)/n],size=10000);draws.append((z[:,0]-z[:,1])/n)
            paired[horizon][model]=dict(benchmarks=individual,macro_delta=float(np.mean([v['accuracy_delta'] for v in individual.values()])),
                macro_ci95=np.quantile(np.mean(draws,axis=0),[.025,.975]).tolist())
    train=root/'arms/balanced_recipe/train'
    for rnd in range(4):
        m=json.loads((train/'metrics'/f'round_{rnd:04d}.json').read_text())[0]
        if m['completed_updates']!=rnd+1 or m['skipped_update'] or not m['causal_update_matched']:
            raise ValueError('Missing matched full-parameter update')
        metrics.append(m)
    atomic_json(root/'comparison.json',dict(scores=summaries,paired_vs_base=paired,training=metrics,
        evaluation_scope='Fresh paired 32k generation; 8k same-trajectory prefix, final-only parser; single seed, no checkpoint selection'))
    lines=['# Matched causal path / larger-batch bounded absolute ReN','',
        f'训练完成：{plan["training"]["prompts_per_round"]} prompts × 4 rounds，{plan["training"]["unique_prompt_exposures"]} 个不同训练题，8k rollout；control={plan["training"]["control_coefficient"]}，reference=0.1。',
        'Base/Round2/Round4 均重新生成，使用相同题目/seed/thinking设置；32k为实际生成，8k为同轨迹CPU截断评分。所有结果只评分结束thinking后的最终回答。','']
    for horizon in (8192,32768):
        lines += [f'## {horizon} tokens','', '| Dataset | N | Base | Round2 | Round4 |','|---|---:|---:|---:|---:|']
        for b in [x['name'] for x in ep['benchmarks']]:
            lines.append(f'| {b} | {summaries[horizon]["base"][b]["examples"]} | '+' | '.join(f'{summaries[horizon][m][b]["accuracy"]:.2%}' for m in models)+' |')
        lines.append('| External macro | — | '+' | '.join(f'{np.mean([summaries[horizon][m][b]["accuracy"] for b in external]):.2%}' for m in models)+' |')
        for m in models[1:]:
            v=paired[horizon][m];lines.append(f'\n{m} 对 Base：{v["macro_delta"]*100:+.2f} pp，配对题目 bootstrap 95% CI [{v["macro_ci95"][0]*100:+.2f}, {v["macro_ci95"][1]*100:+.2f}] pp。')
    lines += ['','## Training','', '| Round | Reason loss | Weighted control | Ref penalty | Score/live relative error | R/C/ref gradient norms |', '|---|---:|---:|---:|---:|---|']
    for m in metrics:
        lines.append(f'| {m["completed_updates"]} | {m["ren_loss"]:.7f} | {m["control_penalty"]:.7f} | {m["reference_penalty"]:.7f} | {m["reasoning_score_live_relative_error"]:.3g} | {m.get("component_gradient_norms",{})} |')
    lines += ['', '单seed、小样本探索性实验；同时改变 batch、control 系数和计算一致性，不能单独归因于任何一项。Round2/4 都预先固定评估，不按外部测试集挑选赢家。',
        '8k prefix 不冒充另一次独立8k生成；与历史完整文本解析的8k数据比较时必须区分评分口径。','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status='complete',paired=paired),indent=2))
if __name__=='__main__':main()
