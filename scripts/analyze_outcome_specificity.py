"""Paired correct-vs-wrong outcome specificity; rollout-cluster intervals."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lulu.training import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);args=p.parse_args();out=Path(args.output_dir)
    manifest=json.loads((out/'manifest.json').read_text());source=Path(manifest['source_audit']);rows=[];score_parts=[];maximum_reconstruction_error=0.
    for path in sorted((out/'scores').glob('*.npz')):
        with np.load(path) as z:a={k:z[k].copy() for k in z.files}
        with np.load(source/'scores'/path.name) as previous:
            maximum_reconstruction_error=max(maximum_reconstruction_error,float(np.abs(a['gold_lambda_']-previous['fp32head_lambda_']).max()))
        w=a['position_expansion']/a['reasoning_tokens'];assert np.isclose(w.sum(),1)
        row={'uid':int(a['uid'][0]),'round':int(a['round'][0]),'capped':bool(a['capped'][0])}
        for label in ['gold','wrong']:
            lam=a[label+'_lambda_']
            for name,values in {'mean_lambda':lam,'positive_fraction':lam>0,'above_025_fraction':lam>.25,'above_05_fraction':lam>.5,'gradient_norm':a[label+'_gradient_norm']}.items():row[label+'_'+name]=float(np.dot(values,w))
        rows.append(row);score_parts.append(a)
    assert len(rows)==48 and maximum_reconstruction_error<1e-5
    measures=['mean_lambda','positive_fraction','above_025_fraction','above_05_fraction','gradient_norm'];rng=np.random.default_rng(20260916)
    rounds=np.array([r['round'] for r in rows]);indices=np.stack([np.concatenate([rng.choice(np.flatnonzero(rounds==ri),16,replace=True) for ri in [0,4,8]]) for _ in range(4000)])
    summary={}
    for measure in measures:
        gold=np.array([r['gold_'+measure] for r in rows]);wrong=np.array([r['wrong_'+measure] for r in rows]);delta=gold-wrong
        summary[measure]={'gold':float(gold.mean()),'wrong':float(wrong.mean()),'gold_minus_wrong':float(delta.mean()),'paired_rollout_cluster_ci95':np.quantile(delta[indices].mean(1),[.025,.975]).tolist(),'rollouts_gold_greater_fraction':float(np.mean(delta>0))}
    rcounts={str(r):{k:float(np.mean([x['gold_'+k]-x['wrong_'+k] for x in rows if x['round']==r])) for k in measures} for r in [0,4,8]}
    round_details={}
    for ri in [0,4,8]:
        group=np.flatnonzero(rounds==ri);samples=rng.choice(group,size=(4000,len(group)),replace=True)
        round_details[str(ri)]={}
        for measure in measures:
            gold=np.array([r['gold_'+measure] for r in rows]);wrong=np.array([r['wrong_'+measure] for r in rows]);delta=gold-wrong
            round_details[str(ri)][measure]={'gold':float(gold[group].mean()),'wrong':float(wrong[group].mean()),'difference':float(delta[group].mean()),'ci95':np.quantile(delta[samples].mean(1),[.025,.975]).tolist()}
    result={'by_round_exploratory':round_details,'status':'complete','rollouts':48,'positions':sum(len(a['uid']) for a in score_parts),'summary':summary,'by_round_paired_differences':rcounts,'maximum_cached_gold_lambda_reconstruction_error':maximum_reconstruction_error,
        'confidence_intervals':'4000 paired rollout-cluster bootstrap, stratified by snapshot round; no token-independence assumption; conditional on one wrong-answer draw per rollout',
        'wrong_answer_rule':manifest['wrong_answer_rule']}
    atomic_json(out/'outcome_specificity.json',result);atomic_json(out/'outcome_rollout_metrics.json',rows)
    lines=['# Correct-outcome specificity audit','',f"48 paired rollouts / {result['positions']} positions; original C/H/T states reused; FP32 output heads.",'',
           '| Metric | Gold | Wrong | Gold − Wrong | Paired rollout CI95 |','|---|---:|---:|---:|---|']
    for name,s in summary.items():lines.append(f"| {name} | {s['gold']:.6f} | {s['wrong']:.6f} | {s['gold_minus_wrong']:+.6f} | {s['paired_rollout_cluster_ci95']} |")
    lines+=['',manifest['wrong_answer_rule'],'',result['confidence_intervals'],'',f"Gold reconstruction max |lambda_new−lambda_saved_FP32|={maximum_reconstruction_error:.3g}. No formula, state selection or numerical threshold changed after observing wrong-answer results."]
    (out/'OUTCOME_SPECIFICITY.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
