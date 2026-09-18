#!/usr/bin/env python3
"""CPU-only RQ2.2 geometry extraction from saved position_scores.npz files.

All aggregate statistics use equal trajectory weight, matching the paper's
reasoning reduction more closely than a raw token average.  Trajectory IDs are
scoped by round so local record indices reused in later rounds never collide.
"""
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np


def composite_groups(rounds, ids):
    rounds=np.asarray(rounds,dtype=np.int64);ids=np.asarray(ids,dtype=np.int64)
    pairs=np.stack((rounds,ids),axis=1)
    _,inverse=np.unique(pairs,axis=0,return_inverse=True)
    return inverse


def trajectory_weights(groups):
    groups=np.asarray(groups)
    unique,counts=np.unique(groups,return_counts=True)
    mapping={u:1.0/c for u,c in zip(unique,counts)}
    w=np.asarray([mapping[x] for x in groups],dtype=np.float64)
    return w/w.sum()


def weighted_mean(x,w):return float(np.sum(np.asarray(x,dtype=np.float64)*w))


def weighted_quantile(values, quantiles, weights):
    values=np.asarray(values,dtype=np.float64);weights=np.asarray(weights,dtype=np.float64);q=np.asarray(quantiles,dtype=np.float64)
    if values.size==0:return np.full_like(q,np.nan,dtype=np.float64)
    order=np.argsort(values,kind='stable');v=values[order];w=weights[order]
    total=w.sum()
    if not total>0:raise ValueError('weighted quantiles require positive total weight')
    # Midpoint CDF avoids making a long trajectory's first token a special edge.
    cdf=(np.cumsum(w)-.5*w)/total
    return np.interp(q,cdf,v,left=v[0],right=v[-1])


def analyze(npz_files,output_dir):
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    parts=[]
    for path in npz_files:
        z=np.load(path)
        required=('causal_kl','resolved_mismatch','bounded_weight','trajectory_index')
        if any(k not in z for k in required):raise ValueError(f'{path} lacks one of {required}')
        round_index=int(Path(path).parent.name.split('_')[-1])
        parts.append({k:z[k] for k in required}|{'round':np.full(len(z['causal_kl']),round_index,dtype=np.int32)})
    if not parts:raise ValueError('no diagnostic files')
    full={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    dc=full['causal_kl'].astype(np.float64);delta=full['resolved_mismatch'].astype(np.float64);ren=full['bounded_weight'].astype(np.float64)
    groups=composite_groups(full['round'],full['trajectory_index']);tw=trajectory_weights(groups)
    # Equal-trajectory weighted D_C quantiles for the conditional geometry plot.
    edges=weighted_quantile(dc,np.linspace(0,1,21),tw);rows=[]
    for i in range(20):
        keep=(dc>=edges[i]) & ((dc<=edges[i+1]) if i==19 else (dc<edges[i+1]))
        if not keep.any():continue
        ww=tw[keep];ww/=ww.sum();vals=delta[keep]
        dq=weighted_quantile(vals,[.25,.5,.75],ww)
        rows.append(dict(bin=i,q_low=i/20,q_high=(i+1)/20,dc_low=float(edges[i]),dc_high=float(edges[i+1]),
                         dc_mean=weighted_mean(dc[keep],ww),delta_mean=weighted_mean(vals,ww),
                         delta_median=float(dq[1]),delta_q25=float(dq[0]),delta_q75=float(dq[2]),
                         positive_fraction=weighted_mean(vals>0,ww),positions=int(keep.sum()),trajectories=int(len(np.unique(groups[keep])))))
    with (output/'dc_delta_quantiles.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
    # Ranking overlap is computed trajectory-by-trajectory, then averaged, so
    # trajectories with 8k reasoning tokens do not dominate short trajectories.
    overlaps=[]
    for frac in (.01,.05,.1,.2,.5):
        vals=[]
        for gid in np.unique(groups):
            idx=np.flatnonzero(groups==gid);k=max(1,int(np.ceil(frac*len(idx))))
            a=set(idx[np.argsort(-ren[idx],kind='stable')[:k]].tolist());b=set(idx[np.argsort(-dc[idx],kind='stable')[:k]].tolist())
            vals.append(len(a&b)/k)
        overlaps.append(dict(top_fraction=frac,mean_overlap=float(np.mean(vals)),median_overlap=float(np.median(vals)),
                             q25_overlap=float(np.quantile(vals,.25)),q75_overlap=float(np.quantile(vals,.75)),
                             random_overlap=frac,trajectories=len(vals)))
    with (output/'topk_overlap.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=overlaps[0].keys());w.writeheader();w.writerows(overlaps)
    summary=dict(positions=len(dc),trajectories=int(len(np.unique(groups))),rounds=int(len(np.unique(full['round']))),
                 causal_kl_mean=weighted_mean(dc,tw),resolved_mean=weighted_mean(delta,tw),
                 resolved_positive_fraction=weighted_mean(delta>0,tw),ren_weight_mean=weighted_mean(ren,tw),
                 pearson_dc_delta=float(np.corrcoef(dc,delta)[0,1]))
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--train-dir',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args()
    files=sorted(Path(a.train_dir).glob('diagnostics/round_*/position_scores.npz'))
    if not files:raise FileNotFoundError('no position_scores.npz files')
    print(json.dumps(analyze(files,a.output_dir),indent=2))
if __name__=='__main__':main()
