"""Outcome-blind matching and first-token intervention preparation."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lulu import training as tr
from lulu.semantic_audit import load_output_weight,residual_coupling,sample_categorical,stable_seed,continuation_budget


def load_scores(out):
    files=sorted((out/'scores').glob('*.npz'));assert len(files)==48
    chunks=[]
    for path in files:
        with np.load(path) as z:chunks.append({k:z[k].copy() for k in z.files})
    return {k:np.concatenate([p[k] for p in chunks]) for k in chunks[0]}


def match(out,target=72):
    a=load_scores(out);lam=a['gold_lambda_']
    labels=np.where(lam>.25,2,np.where(lam>0,1,0))
    logit=lambda p:np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))
    names=['log1p_Teacher_KL_over_0.01','log1p_entropy_over_0.01','log1p_TV_over_0.01','logit_max_confidence','logit_actual_token_confidence','absolute_position']
    features=np.stack([np.log1p(a['gold_teacher_kl']/.01),np.log1p(a['gold_student_entropy']/.01),np.log1p(a['gold_teacher_tv']/.01),logit(a['gold_student_max_probability']),logit(a['student_actual_probability']),a['positions']],axis=1)
    scale=features.std(0);scale=np.maximum(scale,1e-6);z=(features-features.mean(0))/scale
    candidates=[]
    for hi in np.flatnonzero(labels==2):
        exact=(a['round']==a['round'][hi])&(a['capped']==a['capped'][hi])&(a['position_bin']==a['position_bin'][hi])
        close=exact&(np.abs(a['positions']-a['positions'][hi])<=768)&(np.abs(z-z[hi]).max(1)<=1.)
        options=[]
        for level in [1,0]:
            ids=np.flatnonzero(close&(labels==level));dist=np.square(z[ids]-z[hi]).sum(1)
            # Keep nearest candidates separately within/across original rollout.
            selected=[]
            for same in [True,False]:
                keep=(a['uid'][ids]==a['uid'][hi])==same
                ix=np.flatnonzero(keep);selected.extend(ids[ix[np.argsort(dist[ix])[:6]]].tolist())
            options.append(selected)
        for md in options[0]:
            for ze in options[1]:
                # Medium vs zero also satisfy the same continuous caliper.
                if np.abs(z[md]-z[ze]).max()>1. or abs(a['positions'][md]-a['positions'][ze])>768:continue
                same=bool(a['uid'][hi]==a['uid'][md]==a['uid'][ze])
                cost=float(np.square(z[hi]-z[md]).sum()+np.square(z[hi]-z[ze]).sum()+np.square(z[md]-z[ze]).sum())
                candidates.append((not same,cost,int(hi),int(md),int(ze)))
    candidates.sort();used=set();uid_counts=Counter();triplets=[]
    for cross,cost,*ids in candidates:
        if any(i in used for i in ids):continue
        extra=Counter(int(a['uid'][i]) for i in ids)
        if any(uid_counts[u]+n>6 for u,n in extra.items()):continue
        triplets.append(dict(indices=ids,cost=cost,same_rollout=not cross));used.update(ids);uid_counts.update(extra)
        if len(triplets)>=target:break
    if len(triplets)<40:raise ValueError(f'Insufficient common support: {len(triplets)} triplets; do not silently relax calipers')
    records={r['uid']:r for r in map(json.loads,(out/'records.jsonl').read_text().splitlines())}
    states=[]
    for triplet_id,triplet in enumerate(triplets):
        for group,i in zip(['high','medium','zero'],triplet['indices']):
            uid=int(a['uid'][i]);r=records[uid];pos=int(a['positions'][i]);offset=r['positions'].index(pos)
            states.append(dict(state_id=len(states),triplet_id=triplet_id,group=group,uid=uid,source_id=r['source_id'],
                snapshot_round=r['snapshot_round'],position=pos,cache_offset=offset,checkpoint=r['checkpoint'],
                lambda_gold=float(lam[i]),teacher_kl=float(a['gold_teacher_kl'][i]),student_entropy=float(a['gold_student_entropy'][i]),
                teacher_tv=float(a['gold_teacher_tv'][i]),student_max_probability=float(a['gold_student_max_probability'][i]),
                student_actual_probability=float(a['student_actual_probability'][i]),capped=bool(a['capped'][i]),
                matching_features=features[i].tolist()))
    balance={}
    for j,name in enumerate(names):
        values={g:np.array([s['matching_features'][j] for s in states if s['group']==g]) for g in ['high','medium','zero']}
        pooled_sd=np.std(np.concatenate(list(values.values())))
        balance[name]={'group_means':{g:float(v.mean()) for g,v in values.items()},
            'high_vs_zero_smd_global_sd':float((values['high'].mean()-values['zero'].mean())/scale[j]),
            'high_vs_zero_smd_matched_sd':float((values['high'].mean()-values['zero'].mean())/max(pooled_sd,1e-12))}
    output=dict(target_triplets=target,triplets=len(triplets),states=len(states),original_rollouts=len(uid_counts),
        same_rollout_triplets=sum(t['same_rollout'] for t in triplets),feature_names=names,global_feature_std=scale.tolist(),
        rules={'exact':['snapshot_round','capped','absolute_position_bin_2048'],
               'continuous_caliper':'All pairwise transformed feature differences <= 1 global SD; absolute position <=768 tokens',
               'max_states_per_original_rollout':6,'prefer_same_rollout':True,'no_state_reuse':True,
               'excluded_from_matching':['wrong_answer_probe','original_correctness','continuation_outcomes']},
        balance=balance,lambda_group_means={g:float(np.mean([s['lambda_gold'] for s in states if s['group']==g])) for g in ['high','medium','zero']})
    tr.atomic_json(out/'matching.json',output)
    (out/'matched_states.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in states))
    print(json.dumps(output,indent=2))


def prepare_tokens(out,mode):
    if (out/'continuation_jobs.jsonl').exists():raise ValueError('Intervention assignment already frozen')
    m=json.loads((out/'manifest.json').read_text());eta=m['eta'];reps=m['continuation_replicates'] if mode=='residual' else 32
    records={r['uid']:r for r in map(json.loads,(out/'records.jsonl').read_text().splitlines())}
    states=list(map(json.loads,(out/'matched_states.jsonl').read_text().splitlines()))
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    device='cuda';tw=load_output_weight(m['teacher_model'],device)
    jobs=[];pair_events=[]
    for round_ in [0,4,8]:
        subset=[s for s in states if s['snapshot_round']==round_]
        if not subset:continue
        cw=load_output_weight(subset[0]['checkpoint'],device)
        for uid in sorted({s['uid'] for s in subset}):
            r=records[uid];cache=tr.load_tensor_file(r['hidden_path']);selected=[s for s in subset if s['uid']==uid]
            for s in selected:
                k=s['cache_offset']
                with torch.inference_mode():
                    cl=torch.nn.functional.linear(cache['causal'][k:k+1].to(device).float(),cw).squeeze(0)
                    tl=torch.nn.functional.linear(cache['teacher'][k:k+1].to(device).float(),tw).squeeze(0)
                    pc=cl.double().softmax(-1).cpu().numpy();qt=tl.double().softmax(-1).cpu().numpy()
                mass,control,intervention=residual_coupling(pc,qt,eta)
                # Softmax is FP64 here solely to construct normalized categorical
                # sampling probabilities; FP32 heads/selection lambda stay fixed.
                for rep in range(reps):
                    seed=stable_seed(20260916,'continuation',s['state_id'],rep)
                    rng=np.random.default_rng(stable_seed(20260916,'first_token',s['state_id'],rep))
                    changed=bool(mass>0)
                    if mode=='direct':changed=bool(rng.random()<mass)
                    event=dict(state_id=s['state_id'],triplet_id=s['triplet_id'],group=s['group'],uid=uid,
                        replicate=rep,pair_seed=seed,changed_mass=mass,eta=eta,mode=mode,
                        effect_weight=mass if mode=='residual' else 1.,generated_pair=changed)
                    if not changed:
                        event['known_effect']=0.;pair_events.append(event);continue
                    u=float(rng.random());first={'control':sample_categorical(control,u),'intervention':sample_categorical(intervention,u)}
                    assert first['control']!=first['intervention']
                    event['first_tokens']=first;pair_events.append(event)
                    for arm,token in first.items():
                        prefix=r['response_ids'][:s['position']]+[token]
                        prompt=r['causal_prompt_ids']+prefix
                        job=dict(job_id=len(jobs),state_id=s['state_id'],triplet_id=s['triplet_id'],uid=uid,
                            group=s['group'],snapshot_round=round_,checkpoint=s['checkpoint'],position=s['position'],
                            arm=arm,replicate=rep,seed=seed,first_token=token,changed_mass=mass,eta=eta,
                            effect_weight=event['effect_weight'],prompt_token_ids=prompt,response_prefix_ids=prefix,
                            causal_prompt_length=len(r['causal_prompt_ids']),gold_answer=r['gold_answer'],
                            max_new_tokens=continuation_budget(len(r['causal_prompt_ids']),s['position']),
                            first_token_is_eos=token in [151645,151643])
                        job['prompt_sha256']=hashlib.sha256(json.dumps(prompt,separators=(',',':')).encode()).hexdigest()
                        jobs.append(job)
        del cw
    (out/'continuation_jobs.jsonl').write_text(''.join(json.dumps(j)+'\n' for j in jobs))
    (out/'pair_events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in pair_events))
    protocol=dict(mode=mode,states=len(states),replicates_per_state=reps,generated_jobs=len(jobs),eta=eta,
        estimand='Success((1-eta)pC+eta qT first token, then ordinary Student) - Success(pC first token, then ordinary Student)',
        estimator='changed_mass * mean(Y_intervention_residual - Y_control_residual)' if mode=='residual' else 'mean paired difference; identical coupled first tokens analytically cancel',
        equality='For m=eta*TV(pC,qT), common mass cancels exactly; A=m*(E_success_positive_residual-E_success_negative_residual)',
        first_token_policy='Full vocabulary, temperature 1; FP32 LM-head logits normalized in FP64 for categorical sampling',
        continuation_policy={'temperature':.6,'top_p':.95,'top_k':20,'repetition_penalty':1.},
        paired_seeds=True,shared_uniform_for_first_token=True,full_response_horizon=32768,max_model_len=40960,safety_margin=128,
        prefix_unchanged=True,no_prompt_truncation=True,gold_or_wrong_answer_never_in_generation_prompt=True,
        min_changed_mass=float(min(e['changed_mass'] for e in pair_events)),max_changed_mass=float(max(e['changed_mass'] for e in pair_events)),
        jobs_sha256=hashlib.sha256((out/'continuation_jobs.jsonl').read_bytes()).hexdigest(),
        matching_sha256=hashlib.sha256((out/'matched_states.jsonl').read_bytes()).hexdigest())
    tr.atomic_json(out/'continuation_protocol.json',protocol);print(json.dumps(protocol,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);p.add_argument('--tokens',action='store_true');p.add_argument('--mode',choices=['residual','direct'],default='residual');x=p.parse_args();out=Path(x.output_dir)
    prepare_tokens(out,x.mode) if x.tokens else match(out)
if __name__=='__main__':main()
