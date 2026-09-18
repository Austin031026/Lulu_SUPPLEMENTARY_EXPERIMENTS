"""Cluster-aware analysis of the frozen matched continuation experiment."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lulu import training as tr

GROUPS=['high','medium','zero']

def rows(path):return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]

def outcome_overlap(out):
    per=[];gold=[];wrong=[];weights=[]
    for file in sorted((out/'scores').glob('*.npz')):
        with np.load(file) as z:
            g=z['gold_lambda_'].astype(float);w=z['wrong_lambda_'].astype(float)
            weight=z['position_expansion']/z['reasoning_tokens']
            assert abs(weight.sum()-1)<1e-7
            gold.extend(g);wrong.extend(w);weights.extend(weight/48)
            per.append(dict(uid=int(z['uid'][0]),round=int(z['round'][0]),mean_absolute_difference=float(weight@np.abs(g-w)),
                active_agreement=float(weight@((g>0)==(w>0))),gold_high_and_wrong_high=float(weight@((g>.25)&(w>.25))),
                gold_high=float(weight@(g>.25))))
    g,w,p=map(np.asarray,[gold,wrong,weights]);gc=g-p@g;wc=w-p@w
    return dict(weighted_pearson=float(np.sum(p*gc*wc)/np.sqrt(np.sum(p*gc*gc)*np.sum(p*wc*wc))),
        mean_absolute_lambda_difference=float(p@np.abs(g-w)),active_agreement=float(p@((g>0)==(w>0))),
        wrong_high_given_gold_high=float(np.sum(p*((g>.25)&(w>.25)))/np.sum(p*(g>.25))),rollout_metrics=per)

def analyze(out):
    states=rows(out/'matched_states.jsonl');events=rows(out/'pair_events.jsonl');jobs=rows(out/'continuation_jobs.jsonl')
    protocol=json.loads((out/'continuation_protocol.json').read_text());assert hashlib.sha256((out/'continuation_jobs.jsonl').read_bytes()).hexdigest()==protocol['jobs_sha256']
    assert hashlib.sha256((out/'matched_states.jsonl').read_bytes()).hexdigest()==protocol['matching_sha256']
    results=[]
    for f in sorted((out/'continuations').glob('task_*.jsonl')):results.extend(rows(f))
    expected={r['job_id'] for r in jobs};actual=[r['job_id'] for r in results]
    assert len(actual)==len(set(actual)), 'Duplicate job results'
    assert set(actual)==expected,f'Incomplete results: {len(actual)}/{len(expected)}'
    byjob={r['job_id']:r for r in results};byevent=defaultdict(dict)
    errors=[]
    for j in jobs:
        r=byjob[j['job_id']]
        for key in ['state_id','replicate','arm','seed','first_token','prompt_sha256','max_new_tokens']:assert r[key]==j[key],key
        assert r['full_response_tokens']==j['position']+1+len(r['continuation_token_ids'])<=32768
        byevent[(j['state_id'],j['replicate'])][j['arm']]=r
        if any(k.endswith('_error') for k in r['verification']):errors.append({'job_id':r['job_id'],'verification':r['verification']})
    if errors:raise ValueError(f'Verifier errors require resolution before analysis: {errors[:5]}')
    pairs=[]
    for e in events:
        r=dict(e)
        if e['generated_pair']:
            arm=byevent[(e['state_id'],e['replicate'])];assert set(arm)=={'control','intervention'}
            for metric in ['legacy','strict_final']:
                c=int(arm['control']['verification'][metric+'_success']);i=int(arm['intervention']['verification'][metric+'_success'])
                r[metric+'_control']=c;r[metric+'_intervention']=i;r[metric+'_effect']=e['effect_weight']*(i-c)
            r['cap_difference']=int(arm['intervention']['hit_cap'])-int(arm['control']['hit_cap'])
        else:
            for metric in ['legacy','strict_final']:r[metric+'_effect']=0.
        pairs.append(r)
    features=np.array([s['matching_features'] for s in states]);feature_sd=features.std(0)
    names=json.loads((out/'matching.json').read_text())['feature_names'];balance={}
    for ga,gb in [('high','medium'),('high','zero'),('medium','zero')]:
        a=np.array([s['matching_features'] for s in states if s['group']==ga]);b=np.array([s['matching_features'] for s in states if s['group']==gb])
        balance[ga+'_vs_'+gb]=dict(zip(names,((a.mean(0)-b.mean(0))/feature_sd).tolist()))
    tr.atomic_json(out/'matching_all_pair_balance.json',balance)
    state_table=[]
    for s in states:
        p=[v for v in pairs if v['state_id']==s['state_id']];assert len(p)==protocol['replicates_per_state']
        with np.load(out/'scores'/f"rollout_{s['uid']:03d}.npz") as z:
            old=float(z['gold_old_weight'][s['cache_offset']]);wrong=float(z['wrong_lambda_'][s['cache_offset']])
        row={k:v for k,v in s.items() if k not in ['matching_features','checkpoint']}
        row.update(old_weight=old,lambda_wrong=wrong,changed_mass=float(np.mean([r['changed_mass'] for r in p])))
        for metric in ['legacy','strict_final']:
            row[metric+'_effect']=float(np.mean([r[metric+'_effect'] for r in p]))
            if protocol['mode']=='residual':
                for arm in ['control','intervention']:row[metric+'_residual_'+arm]=float(np.mean([r[metric+'_'+arm] for r in p]))
        state_table.append(row)
    # Unit of resampling is the original rollout; all states, matched triplets,
    # repetitions and treatment/control outcomes within a rollout stay together.
    uid=np.array([s['uid'] for s in states]);group=np.array([s['group'] for s in states]);rounds=np.array([s['snapshot_round'] for s in states])
    clusters=np.unique(uid);rng=np.random.default_rng(2026091603)
    counts=np.empty((4000,len(clusters)),dtype=np.int16)
    for b in range(len(counts)):
        sampled=[]
        for round_ in np.unique(rounds):
            pool=np.unique(uid[rounds==round_]);sampled.extend(rng.choice(pool,len(pool),replace=True).tolist())
        counts[b]=[sampled.count(int(u)) for u in clusters]
    state_weights=counts[:,np.searchsorted(clusters,uid)]
    output={'status':'complete','states':len(states),'original_rollouts':len(clusters),'paired_replicates':len(events),'generation_jobs':len(jobs),
        'mode':protocol['mode'],'eta':protocol['eta'],'group_metrics':{},'contrasts':{},
        'ci_method':'4000 original-rollout cluster bootstraps, stratified by snapshot round; states/triplets/repetitions/arms remain paired',
        'interpretation':'Effects estimate the original fixed-eta mixture first-token intervention. Residual arm success rates are conditional, not original policy accuracies.',
        'verifier_errors':errors,'exploratory_associations':{}}
    boot_by_metric={}
    for metric in ['legacy','strict_final']:
        y=np.array([s[metric+'_effect'] for s in state_table]);boot={}
        for g in GROUPS:
            mask=group==g;wg=state_weights[:,mask];boot[g]=wg@y[mask]/wg.sum(1)
            gm=output['group_metrics'].setdefault(g,{'states':int(mask.sum()),'mean_lambda':float(np.mean([s['lambda_gold'] for s in state_table if s['group']==g])),
                'mean_changed_mass':float(np.mean([s['changed_mass'] for s in state_table if s['group']==g]))})
            gm[metric]={'effect':float(y[mask].mean()),'ci95':np.quantile(boot[g],[.025,.975]).tolist()}
            if protocol['mode']=='residual':
                for arm in ['control','intervention']:gm[metric]['conditional_residual_'+arm+'_success']=float(np.mean([s[metric+'_residual_'+arm] for s in state_table if s['group']==g]))
        for g in ['high','medium']:
            output['contrasts'][metric+'_'+g+'_minus_zero']={'effect':float(y[group==g].mean()-y[group=='zero'].mean()),'ci95':np.quantile(boot[g]-boot['zero'],[.025,.975]).tolist()}
        # Exploratory within-triplet association, not a classification AUC:
        # two repetitions cannot label a state's true utility sign reliably.
        trip=np.array([s['triplet_id'] for s in state_table]);yc=y.copy()
        for tid in np.unique(trip):yc[trip==tid]-=yc[trip==tid].mean()
        for name in ['lambda_gold','old_weight']:
            x=np.array([s[name] for s in state_table]);xc=x.copy()
            for tid in np.unique(trip):xc[trip==tid]-=xc[trip==tid].mean()
            numerator=state_weights@(xc*yc);denominator=state_weights@(xc*xc)
            slopes=numerator/denominator
            output['exploratory_associations'][metric+'_'+name+'_within_triplet_slope']={'slope':float(xc@yc/(xc@xc)),'ci95':np.quantile(slopes,[.025,.975]).tolist()}
    output['generation']={'total_continuation_tokens':sum(r['continuation_tokens'] for r in results),
        'hit_cap':sum(r['hit_cap'] for r in results),'finish_reason_counts':dict(__import__('collections').Counter(r['finish_reason'] for r in results)),
        'legacy_strict_disagreements':sum(r['verification']['legacy_success']!=r['verification']['strict_final_success'] for r in results)}
    tr.atomic_json(out/'utility_results.json',output);tr.atomic_json(out/'outcome_overlap.json',outcome_overlap(out))
    (out/'pair_results.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in pairs))
    with (out/'state_utility.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(state_table[0]));writer.writeheader();writer.writerows(state_table)
    print(json.dumps(output,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);x=p.parse_args();analyze(Path(x.output_dir))
if __name__=='__main__':main()
