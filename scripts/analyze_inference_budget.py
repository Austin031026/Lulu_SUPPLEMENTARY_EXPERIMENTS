#!/usr/bin/env python3
"""CPU-only inference-budget analysis from stored vLLM token IDs.

No new text is generated.  The same saved 32k response is decoded at shorter
prefix budgets, which supports RQ3.2 accuracy/efficiency curves while keeping
sampling noise fixed within a response.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
from rq_metrics import MATH,OOD,EXTERNAL,mean


def analyze_rows(rows, budgets, decode, parser):
    out=[];first=[]
    for row in rows:
        tokens=row.get('response_token_ids')
        if tokens is None:raise ValueError('evaluation rows must store response_token_ids')
        first_correct=first_valid=first_closed=None
        for budget in budgets:
            prefix=tokens[:budget];text=decode(prefix)
            pred=parser.extract_answer(text,str(row.get('data_source','')))
            gold=row.get('ground_truth');reward=float(parser.math_equal(pred,gold)) if gold is not None else None
            closed='</think>' in text;valid=pred is not None
            out.append({'prompt_index':int(row['prompt_index']),'budget':int(budget),'prediction':pred,'reward':reward,
                        'valid_answer':valid,'closed_thinking':closed,'prefix_tokens':len(prefix),'full_response_tokens':len(tokens),
                        'hit_prefix_cap':len(tokens)>=budget})
            if closed and first_closed is None:first_closed=budget
            if valid and first_valid is None:first_valid=budget
            if reward==1 and first_correct is None:first_correct=budget
        first.append({'prompt_index':int(row['prompt_index']),'first_closed_budget':first_closed,
                      'first_valid_budget':first_valid,'first_correct_budget':first_correct,
                      'full_response_tokens':len(tokens)})
    return out,first


def load_rows(directory):
    rows=[]
    for path in sorted(Path(directory).glob('shard-*.jsonl')):
        rows.extend(json.loads(line) for line in path.read_text().splitlines() if line.strip())
    return sorted(rows,key=lambda x:int(x['prompt_index']))


def _write(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)


def _aggregates(records,budgets):
    rows=[];models=sorted({r['model'] for r in records});benches=sorted({r['benchmark'] for r in records})
    for model in models:
        for bench in benches:
            for budget in budgets:
                x=[r for r in records if r['model']==model and r['benchmark']==bench and r['budget']==budget]
                if not x:continue
                rows.append({'model':model,'benchmark':bench,'budget':budget,'examples':len(x),
                             'accuracy':sum(r['reward'] for r in x)/len(x),
                             'valid_answer_fraction':sum(r['valid_answer'] for r in x)/len(x),
                             'closed_thinking_fraction':sum(r['closed_thinking'] for r in x)/len(x),
                             'mean_prefix_tokens':sum(r['prefix_tokens'] for r in x)/len(x),
                             'mean_full_response_tokens':sum(r['full_response_tokens'] for r in x)/len(x)})
    return rows


def _grouped(aggregate,budgets):
    rows=[];models=sorted({r['model'] for r in aggregate})
    index={(r['model'],r['benchmark'],r['budget']):r for r in aggregate}
    for model in models:
        for budget in budgets:
            def accs(names):return [index[(model,b,budget)]['accuracy'] for b in names if (model,b,budget) in index]
            external=[index[(model,b,budget)] for b in EXTERNAL if (model,b,budget) in index]
            micro_num=sum(r['accuracy']*r['examples'] for r in external);micro_den=sum(r['examples'] for r in external)
            rows.append({'model':model,'budget':budget,'math_avg':mean(accs(MATH)),'ood_avg':mean(accs(OOD)),
                         'external_macro':mean(accs(EXTERNAL)),'external_micro':micro_num/micro_den if micro_den else None,
                         'external_examples':micro_den})
    return rows


def _solve_curves(first_rows,budgets):
    rows=[]
    groups=sorted({(r['model'],r['benchmark']) for r in first_rows})
    for model,bench in groups:
        x=[r for r in first_rows if r['model']==model and r['benchmark']==bench]
        for budget in budgets:
            rows.append({'model':model,'benchmark':bench,'budget':budget,'examples':len(x),
                         'correct_by_budget_fraction':sum(r['first_correct_budget'] is not None and r['first_correct_budget']<=budget for r in x)/len(x),
                         'valid_by_budget_fraction':sum(r['first_valid_budget'] is not None and r['first_valid_budget']<=budget for r in x)/len(x),
                         'closed_by_budget_fraction':sum(r['first_closed_budget'] is not None and r['first_closed_budget']<=budget for r in x)/len(x)})
    return rows


def summarize(evaluation_dir,output_dir,budgets):
    root=Path(evaluation_dir);outdir=Path(output_dir);outdir.mkdir(parents=True,exist_ok=True)
    plan=json.loads((root/'eval_plan.json').read_text())
    from transformers import AutoTokenizer
    from lulu.evaluation_runner import load_parser
    parser=load_parser(plan['parser_path']);records=[];first=[]
    for model in plan['models']:
        tok=AutoTokenizer.from_pretrained(model['model'],trust_remote_code=plan.get('trust_remote_code',False))
        decode=lambda ids,tok=tok:tok.decode(ids,skip_special_tokens=True)
        for bench in plan['benchmarks']:
            rows=load_rows(root/model['name']/bench['name']);scored,earliest=analyze_rows(rows,budgets,decode,parser)
            for item in scored:item.update(model=model['name'],benchmark=bench['name'])
            records.extend(scored)
            for item in earliest:item.update(model=model['name'],benchmark=bench['name'])
            first.extend(earliest)
    _write(outdir/'prefix_scores.csv',records);_write(outdir/'first_budget.csv',first)
    aggregate=_aggregates(records,budgets);grouped=_grouped(aggregate,budgets);solve=_solve_curves(first,budgets)
    _write(outdir/'budget_summary.csv',aggregate);_write(outdir/'budget_grouped_summary.csv',grouped);_write(outdir/'solve_by_budget.csv',solve)
    summary={'budgets':list(budgets),'rows':len(records),'models':sorted({r['model'] for r in records}),
             'benchmarks':sorted({r['benchmark'] for r in records}),'files':['prefix_scores.csv','first_budget.csv','budget_summary.csv','budget_grouped_summary.csv','solve_by_budget.csv']}
    (outdir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--evaluation-dir',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--budgets',default='2048,4096,8192,16384,32768');a=p.parse_args();budgets=tuple(sorted(set(int(x) for x in a.budgets.split(',') if x.strip())))
    print(json.dumps(summarize(a.evaluation_dir,a.output_dir,budgets),indent=2))
if __name__=='__main__':main()
