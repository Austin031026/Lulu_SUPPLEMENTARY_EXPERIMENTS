"""CPU-only checks on saved projection moments; no extra model execution."""
import argparse
from pathlib import Path
import numpy as np
import json

p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);x=p.parse_args()
out=Path(x.output_dir);parts=[]
for f in sorted((out/'scores').glob('*.npz')):
    with np.load(f) as z:parts.append({k:z[k].copy() for k in z.files})
a={k:np.concatenate([b[k] for b in parts]) for k in parts[0]}
w=a['position_expansion']/a['reasoning_tokens']
def m(value):return float(np.average(value,weights=w))
def sub(value,mask):return float(np.average(value[mask],weights=w[mask])) if mask.any() else None
result={}
for label,mask in {'degenerate':~a['alignment_valid'],'small_H':a['hindsight_variance']<=1e-10,
    'small_T':a['teacher_variance']<=1e-10,'exact_zero_H':a['hindsight_variance']==0,
    'high_cos':a['cosine']>.5,'high_beta':a['beta']>.5}.items():
    result[label]={'fraction':m(mask),'beta':sub(a['beta'],mask),'entropy_C':sub(a['causal_entropy'],mask),
        'KL_T':sub(a['causal_kl'],mask),'KL_star':sub(a['projected_kl'],mask),
        'movement_mass':float(np.sum(w[mask]*a['projected_kl'][mask])/np.sum(w*a['projected_kl']))}
result['epsilon_beta_sensitivity']={}
for e in [1e-10,1e-8,1e-6,1e-4]:
    beta=np.where(a['teacher_variance']>1e-10,np.clip(a['covariance']/(a['teacher_variance']+e),0,1),0)
    result['epsilon_beta_sensitivity'][str(e)]={'beta_mean':m(beta),'fraction_over_05':m(beta>.5)}
result['gradient_norm_ratio_new_over_old']=m(a['projected_logit_gradient_norm'])/m(a['old_weighted_logit_gradient_norm'])
result['capped_projected_kl_share']=float(np.sum(w*a['projected_kl']*a['capped'])/np.sum(w*a['projected_kl']))
result['new_mass_on_old_zero']=float(np.sum(w*a['projected_kl']*(a['old_weight']==0))/np.sum(w*a['projected_kl']))
result['weighted_gap_sign_agreement_away_from_zero']={str(e):sub(((a['resolved_mismatch']>0)==(a['saved_resolved_mismatch']>0)).astype(float),np.abs(a['saved_resolved_mismatch'])>e) for e in [.001,.01,.05,.1]}
result['variance_floor_sensitivity']={}
for floor in [1e-12,1e-10,1e-8,1e-6]:
    valid=(a['teacher_variance']>floor)&(a['hindsight_variance']>floor)
    cosine=a['covariance']/np.maximum(np.sqrt(a['teacher_variance']*a['hindsight_variance']),floor)
    result['variance_floor_sensitivity'][str(floor)]={'degenerate_fraction':m(~valid),'positive_over_005':m(valid&(cosine>.05))}
(out/'extra_checks.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
