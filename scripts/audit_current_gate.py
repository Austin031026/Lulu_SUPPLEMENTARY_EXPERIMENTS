"""CPU audit of actual bounded absolute ReN, reusing fixed wrong-answer/utility caches."""
from pathlib import Path
import argparse,json,csv,hashlib
import numpy as np

def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def corr(x,y,p):
    x=x-p@x;y=y-p@y;d=np.sqrt((p@(x*x))*(p@(y*y)))
    return float(p@(x*y)/d) if d else None

def cluster_counts(uid,rounds,draws=4000,seed=20260917):
    clusters=np.unique(uid);rng=np.random.default_rng(seed);counts=np.zeros((draws,len(clusters)))
    for ri in np.unique(rounds):
        ids=np.unique(uid[rounds==ri]);cols=np.searchsorted(clusters,ids)
        counts[:,cols]=rng.multinomial(len(ids),np.ones(len(ids))/len(ids),size=draws)
    return counts,clusters

def adjusted(x,y,controls,fixed,uid,rounds):
    """Within-stratum OLS; full rollout clusters resampled with their paired outcomes."""
    x=np.asarray(x,float)
    if x.ndim==1:x=x[:,None]
    X=np.column_stack((x,controls));sd=X.std(0);sd[sd<1e-12]=1
    X=(X-X.mean(0))/sd;Y=y.copy()
    for f in np.unique(fixed):
        mask=fixed==f;X[mask]-=X[mask].mean(0);Y[mask]-=Y[mask].mean()
    counts,clusters=cluster_counts(uid,rounds)
    xx=np.array([X[uid==u].T@X[uid==u] for u in clusters]);xy=np.array([X[uid==u].T@Y[uid==u] for u in clusters])
    mat=xx.sum(0);v=xy.sum(0);coef=np.linalg.pinv(mat,rcond=1e-10)@v
    bm=np.einsum('bi,ijk->bjk',counts,xx);bv=counts@xy
    bc=np.einsum('bij,bj->bi',np.linalg.pinv(bm,rcond=1e-10),bv)
    # Return coefficients in raw x units; controls only standardize conditioning.
    return {'coefficients':(coef[:x.shape[1]]/sd[:x.shape[1]]).tolist(),
        'ci95':np.quantile(bc[:,:x.shape[1]]/sd[:x.shape[1]],[.025,.975],axis=0).T.tolist(),
        'design_rank':int(np.linalg.matrix_rank(X)),'columns':X.shape[1],
        'bootstrap_rank_deficient_fraction':float(np.mean(np.linalg.matrix_rank(bm)<X.shape[1]))}

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--scores');p.add_argument('--precision',default='cached FP32 head; BF16 backbone');a=p.parse_args()
    source=Path(a.source);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);score_dir=Path(a.scores) if a.scores else source/'scores'
    files=sorted(score_dir.glob('rollout_*.npz'));assert len(files)==48
    provenance={};parts=[];per=[];offsets={}
    for path in files:
        provenance[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as z:
            d={k:z[k].copy() for k in z.files}
        uid=int(d['uid'][0]);round_=int(d['round'][0]);n=int(d['reasoning_tokens'][0]);mass=d['position_expansion'].astype(float)/n
        assert abs(mass.sum()-1)<1e-7
        wg,ww=d['gold_old_weight'].astype(float),d['wrong_old_weight'].astype(float)
        assert np.all((wg>=0)&(wg<1)) and np.all((ww>=0)&(ww<1))
        gg=wg/(1-wg);gw=ww/(1-ww)
        part=dict(uid=np.full(len(wg),uid),round=np.full(len(wg),round_),p=mass/48,wg=wg,ww=ww,gg=gg,gw=gw,positions=d['positions'])
        if 'gold_gap' in d:
            assert np.max(np.abs(np.maximum(d['gold_gap'],0)-gg))<2e-5
            s=np.maximum(d['gold_gap'].astype(float)-d['wrong_gap'].astype(float),0)
            signed=d['gold_gap'].astype(float)-d['wrong_gap'].astype(float);reverse=np.maximum(-signed,0)
            part.update(s=s,ws=s/(1+s),signed_specificity=signed,ws_reverse=reverse/(1+reverse))
        parts.append(part);offsets[uid]=(d,part)
        per.append(dict(uid=uid,round=round_,gold_g=float(mass@gg),wrong_g=float(mass@gw),gold_w=float(mass@wg),wrong_w=float(mass@ww),gold_active=float(mass@(wg>0)),wrong_active=float(mass@(ww>0))))
    keys=parts[0].keys();v={k:np.concatenate([x[k] for x in parts]) for k in keys};assert len(v['wg'])==10433
    pp=v['p'];counts,clusters=cluster_counts(np.array([x['uid'] for x in per]),np.array([x['round'] for x in per]));summary={}
    for key in ['g','w','active']:
        gold=np.array([r['gold_'+key] for r in per]);wrong=np.array([r['wrong_'+key] for r in per]);delta=gold-wrong
        summary[key]=dict(gold=float(gold.mean()),wrong=float(wrong.mean()),difference=float(delta.mean()),ci95=np.quantile(counts@delta/48,[.025,.975]).tolist(),positive_rollout_fraction=float(np.mean(delta>0)))
    def quantile(x,q):
        order=np.argsort(x,kind='stable');return float(x[order][np.searchsorted(pp[order].cumsum(),q)])
    threshold=quantile(v['wg'],.9);wrong_threshold=quantile(v['ww'],.9)
    hg=v['wg']>threshold;hw=v['ww']>wrong_threshold
    overlap=dict(weighted_correlation=corr(v['wg'],v['ww'],pp),gold_high_threshold=threshold,wrong_high_threshold=wrong_threshold,
        high_overlap_given_gold=float(pp@(hg&hw)/(pp@hg)),high_jaccard=float(pp@(hg&hw)/(pp@(hg|hw))),
        same_threshold_wrong_high_given_gold=float(pp@(hg&(v['ww']>threshold))/(pp@hg)),
        shared_minimum_weight_mass_fraction_of_gold=float(pp@np.minimum(v['wg'],v['ww'])/(pp@v['wg'])),
        wrong_to_gold_weight_mass=float((pp@v['ww'])/(pp@v['wg'])))
    group_protocol={'high':f'w > {threshold:.12g} (90th percentile of all cached gold positions, prompt balanced)','medium':f'0 < w <= {threshold:.12g}','zero':'w == 0',
        'no_outcome_based_selection':True,'primary':'strict-final mixture treatment effect, not residual-arm accuracy','controls':['log1p Teacher KL','Student entropy','absolute response position','Teacher TV','Student maximum probability','Student actual-token probability'],
        'primary_adjustment':'original matched-triplet fixed effects + continuous controls; rollout-cluster bootstrap stratified by snapshot',
        'selection_limit':'states were originally selected/matched on lambda; regrouping does not create a representative sample or remove all confounding'}
    save(out/'utility_group_protocol.json',group_protocol)
    # Only now read the existing continuation outcomes; never generate new text.
    path=source/'state_utility.csv';provenance[str(path.resolve())]=hashlib.sha256(path.read_bytes()).hexdigest()
    states=list(csv.DictReader(path.open()));assert len(states)==216
    uid=np.array([int(s['uid']) for s in states]);rounds=np.array([int(s['snapshot_round']) for s in states]);trip=np.array([int(s['triplet_id']) for s in states])
    weights=np.array([offsets[int(s['uid'])][1]['wg'][int(s['cache_offset'])] for s in states]);wrongw=np.array([offsets[int(s['uid'])][1]['ww'][int(s['cache_offset'])] for s in states])
    for s,w in zip(states,weights):
        d,_=offsets[int(s['uid'])];assert int(d['positions'][int(s['cache_offset'])])==int(s['position'])
        if not a.scores:assert abs(w-float(s['old_weight']))<1e-9
    groups=np.where(weights>threshold,'high',np.where(weights>0,'medium','zero'));y=np.array([float(s['strict_final_effect']) for s in states])
    controls=np.array([[np.log1p(float(s['teacher_kl'])),float(s['student_entropy']),int(s['position'])/8192,float(s['teacher_tv']),float(s['student_max_probability']),float(s['student_actual_probability'])] for s in states]);sd=controls.std(0);sd[sd<1e-12]=1
    bc,uc=cluster_counts(uid,rounds);sw=bc[:,np.searchsorted(uc,uid)];unadjusted={};boots={}
    for label in ['zero','medium','high']:
        mask=groups==label;den=sw[:,mask].sum(1);good=den>0;boots[label]=(sw[good][:,mask]@y[mask])/den[good]
        unadjusted[label]=dict(states=int(mask.sum()),rollouts=int(len(np.unique(uid[mask]))),mean_w=float(weights[mask].mean()) if mask.any() else None,
            effect=float(y[mask].mean()) if mask.any() else None,ci95=np.quantile(boots[label],[.025,.975]).tolist() if mask.any() else None,
            mean_changed_mass=float(np.mean([float(s['changed_mass']) for s,m in zip(states,mask) if m])) if mask.any() else None)
    maskh=groups=='high';maskz=groups=='zero';balance=((controls[maskh].mean(0)-controls[maskz].mean(0))/sd).tolist()
    continuous=adjusted(weights,y,controls,trip,uid,rounds);group_fit=adjusted(np.column_stack((groups=='high',groups=='medium')),y,controls,trip,uid,rounds)
    sensitivity=adjusted(weights,y,controls,uid,uid,rounds)
    utility=dict(states=216,rollouts=len(uc),groups=unadjusted,unadjusted_high_minus_zero=float(y[maskh].mean()-y[maskz].mean()),
        unadjusted_high_vs_zero_standardized_differences=dict(zip(group_protocol['controls'],balance)),
        adjusted_w_slope=continuous,adjusted_high_minus_zero=group_fit,rollout_fixed_effect_sensitivity=sensitivity,
        w_sd=float(weights.std()),adjusted_effect_per_w_sd=continuous['coefficients'][0]*float(weights.std()),
        adjusted_ci95_per_w_sd=(np.array(continuous['ci95'][0])*weights.std()).tolist())
    if 's' in v:
        spec=np.array([offsets[int(s['uid'])][1]['ws'][int(s['cache_offset'])] for s in states]);specfit=adjusted(spec,y,controls,trip,uid,rounds)
        signed_rollout=np.array([float((part['p']*48)@part['signed_specificity']) for part in parts])
        ws_rollout=np.array([float((part['p']*48)@part['ws']) for part in parts])
        reverse_rollout=np.array([float((part['p']*48)@part['ws_reverse']) for part in parts])
        summary['specificity_score']=dict(mean_signed_gap=float(signed_rollout.mean()),signed_gap_ci95=np.quantile(counts@signed_rollout/48,[.025,.975]).tolist(),
            mean_reverse_wspec=float(reverse_rollout.mean()),wspec_minus_reverse_ci95=np.quantile(counts@(ws_rollout-reverse_rollout)/48,[.025,.975]).tolist(),
            mean_s=float(pp@v['s']),mean_wspec=float(pp@v['ws']),mean_wspec_ci95=np.quantile(counts@ws_rollout/48,[.025,.975]).tolist(),positive_fraction=float(pp@(v['s']>0)),utility_slope=specfit,
            utility_wspec_sd=float(spec.std()),utility_effect_per_sd=specfit['coefficients'][0]*float(spec.std()),utility_ci95_per_sd=(np.array(specfit['ci95'][0])*spec.std()).tolist(),
            limitation='one wrong-answer donor per rollout; between-donor robustness not identified')
    result=dict(precision=a.precision,positions=len(pp),rollouts=48,specificity=summary,overlap=overlap,utility=utility,
        per_rollout=per,confidence='4000 paired original-rollout cluster bootstrap, stratified by snapshot; conditional on sampled positions and one wrong donor',
        s_available='s' in v)
    save(out/'results.json',result);save(out/'provenance.json',dict(inputs=provenance,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    rows=[]
    for i,s in enumerate(states):rows.append(dict(state_id=int(s['state_id']),uid=int(uid[i]),round=int(rounds[i]),position=int(s['position']),triplet=int(trip[i]),old_lambda_group=s['group'],group=groups[i],w_gold=float(weights[i]),w_wrong=float(wrongw[i]),effect=float(y[i])))
    with (out/'regrouped_states.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps({k:result[k] for k in ['precision','specificity','overlap','utility','s_available']},indent=2))
if __name__=='__main__':main()
