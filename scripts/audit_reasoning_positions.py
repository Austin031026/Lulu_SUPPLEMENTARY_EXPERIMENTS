#!/usr/bin/env python3
"""CPU-only positional audit of saved ReN diagnostics; no model inference."""
from pathlib import Path
from collections import Counter,defaultdict
import argparse,csv,json,hashlib
import numpy as np

EDGES=np.array([0,2048,4096,6144,8192])
LABELS=['0–2k','2–4k','4–6k','6–8k']

def audit(train,output):
    output.mkdir(parents=True,exist_ok=True)
    trajectory_rows=[];source_hashes={};round_validation=[];available_keys=set();all_stats=[]
    for folder in sorted((train/'diagnostics').glob('round_*')):
        index=int(folder.name.split('_')[-1]);path=folder/'position_scores.npz'
        source_hashes[str(path.relative_to(train))]=hashlib.sha256(path.read_bytes()).hexdigest()
        records={}
        for rp in sorted((train/'rollouts'/folder.name).glob('shard-*.jsonl')):
            source_hashes[str(rp.relative_to(train))]=hashlib.sha256(rp.read_bytes()).hexdigest()
            for line in rp.open():
                r=json.loads(line)
                assert r['index'] not in records and r['snapshot_round']==index
                records[r['index']]=r
        counts=Counter(r['source_id'] for r in records.values());B=len(counts)
        with np.load(path) as z:
            available_keys.update(z.files)
            ids=z['trajectory_index'];positions=z['position']
            w=z['bounded_weight'].astype(float);dc=z['causal_kl'].astype(float)
            gap=z['resolved_mismatch'].astype(float);loss=w*dc
            assert np.isfinite(w).all() and np.isfinite(loss).all()
            assert np.allclose(w,np.maximum(gap,0)/(1+np.maximum(gap,0)),atol=2e-7)
            reconstructed=0.;N=sum(r['response_tokens'] for r in records.values())
            for rid,r in records.items():
                mask=ids==rid;pos=positions[mask];weights=w[mask];kl=dc[mask];gaps=gap[mask];losses=loss[mask]
                expected=np.flatnonzero(r['reasoning_mask'])
                assert np.array_equal(pos,expected),f'Position coverage mismatch {index}/{rid}'
                R=len(pos);scale=1/(B*counts[r['source_id']]*R) if R else 0.
                reconstructed+=losses.sum()*scale
                for b,(lo,hi) in enumerate(zip(EDGES[:-1],EDGES[1:])):
                    m=(pos>=lo)&(pos<hi);n=int(m.sum())
                    trajectory_rows.append(dict(round=index+1,trajectory_index=rid,source_id=r['source_id'],
                        capped=bool(r['truncated']),correct=bool(r['correct']),reasoning_tokens=R,
                        bin=b,start=int(lo),end=int(hi),tokens=n,weight_sum=float(weights[m].sum()),
                        weighted_kl_sum=float(losses[m].sum()),signed_gap_sum=float(gaps[m].sum()),
                        dc_sum=float(kl[m].sum()),positive_gap_tokens=int((gaps[m]>0).sum()),
                        balanced_weight_mass=float(weights[m].sum()*scale),
                        balanced_loss_mass=float(losses[m].sum()*scale)))
            metrics=json.loads((train/'metrics'/f'round_{index:04d}.json').read_text())
            if isinstance(metrics,list):metrics=metrics[0]
            logged=metrics['reasoning_loss_rollout_balanced']
            round_validation.append(dict(round=index+1,rollouts=len(records),prompts=B,
                reconstructed_preupdate_reasoning_loss=reconstructed,logged_training_reasoning_loss=logged,
                relative_difference=(reconstructed-logged)/logged if logged else None,
                reconstructed_global_token_loss=float(loss.sum()/N),logged_global_token_loss=metrics['reasoning_loss_token_global']))
    cohorts={'all':lambda r:True,'capped_same_cohort':lambda r:r['capped'],
             'completed':lambda r:not r['capped'],'correct':lambda r:r['correct'],
             'incorrect':lambda r:not r['correct']}
    for cohort,keep in cohorts.items():
        subset=[r for r in trajectory_rows if keep(r)]
        for scope in ['pooled',*sorted({r['round'] for r in subset})]:
            scope_rows=[r for r in subset if scope=='pooled' or r['round']==scope]
            totals={key:sum(r[key] for r in scope_rows) for key in ['weight_sum','weighted_kl_sum','balanced_weight_mass','balanced_loss_mass']}
            for b in range(4):
                rows=[r for r in scope_rows if r['bin']==b];n=sum(r['tokens'] for r in rows)
                if not n:continue
                row=dict(cohort=cohort,round=scope,bin=LABELS[b],start=int(EDGES[b]),end=int(EDGES[b+1]),
                    reasoning_tokens=n,contributing_rollouts=sum(r['tokens']>0 for r in rows),
                    mean_w=sum(r['weight_sum'] for r in rows)/n,
                    mean_w_kl_preupdate=sum(r['weighted_kl_sum'] for r in rows)/n,
                    mean_dc_minus_dh=sum(r['signed_gap_sum'] for r in rows)/n,
                    mean_dc=sum(r['dc_sum'] for r in rows)/n,
                    positive_gap_fraction=sum(r['positive_gap_tokens'] for r in rows)/n,
                    teacher_actual_token_logprob=None)
                for key,total in totals.items():
                    row[key]=sum(r[key] for r in rows)
                    row[key+'_share']=row[key]/total if total else None
                    row[key+'_cumulative_share']=sum(r[key] for r in scope_rows if r['bin']<=b)/total if total else None
                all_stats.append(row)
    def write_csv(name,rows):
        with (output/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_csv('bins.csv',all_stats);write_csv('trajectory_bins.csv',trajectory_rows);write_csv('round_reconstruction.csv',round_validation)
    # Matched capped trajectories avoid mixing different surviving populations by position.
    capped=[r for r in trajectory_rows if r['capped']]
    paired=[];rng=np.random.default_rng(20260916)
    for key in ['weight_sum','weighted_kl_sum','signed_gap_sum']:
        byid=defaultdict(dict)
        for r in capped:
            if r['tokens']:byid[(r['round'],r['trajectory_index'])][r['bin']]=r[key]/r['tokens']
        values=np.array([[v[0],v[3]] for v in byid.values() if 0 in v and 3 in v]);delta=values[:,1]-values[:,0]
        boot=delta[rng.integers(0,len(delta),size=(5000,len(delta)))].mean(1)
        paired.append(dict(metric=key,rollouts=len(delta),early_mean=float(values[:,0].mean()),late_mean=float(values[:,1].mean()),
                           late_minus_early=float(delta.mean()),paired_rollout_bootstrap_95ci=np.quantile(boot,[.025,.975]).tolist()))
    result={'scope':'Saved reasoning positions, absolute zero-based generated-token offsets; k=1024; no GPU/model calls',
        'training_run':str(train),'bins':EDGES.tolist(),'rollouts':len(trajectory_rows)//4,
        'weighted_kl_definition':'w * D_C at frozen pre-update Student. One optimizer update/round, one pass; reconstructed target loss is checked against training metrics. Not per-position post-update KL or gradient mass.',
        'teacher_actual_token_logprob':{'available':False,'reason':'Not stored in position_scores.npz or rollout metadata; aggregate full-vocabulary KL cannot recover log q_T(actual_token). No extra Teacher inference performed.'},
        'reconstruction_relative_error_range':[min(x['relative_difference'] for x in round_validation),max(x['relative_difference'] for x in round_validation)],
        'npz_keys':sorted(available_keys),'pooled_bins':[r for r in all_stats if r['round']=='pooled'],
        'capped_paired_early_vs_late':paired,'round_validation':round_validation,'source_sha256':source_hashes,
        'limitations':['ReN score/loss mass is not causal usefulness or gradient contribution.',
         'All-rollout later bins condition on surviving trajectories; matched capped cohort controls this composition difference.',
         'Nothing beyond token 8191 was observed; these data do not determine whether new useful signal appears after 8k.',
         'Bootstrap covers rollout sampling only, not training-seed uncertainty.']}
    (output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    pooled={c:[r for r in result['pooled_bins'] if r['cohort']==c] for c in ['all','capped_same_cohort']}
    ar,cr=pooled['all'],pooled['capped_same_cohort']
    capped_count=len({(r['round'],r['trajectory_index']) for r in capped})
    capped_rounds={i:[r for r in all_stats if r['cohort']=='capped_same_cohort' and r['round']==i] for i in range(1,len(round_validation)+1)}
    lower=sum(r[-1]['mean_w_kl_preupdate']<r[0]['mean_w_kl_preupdate'] for r in capped_rounds.values())
    direction='衰减' if cr[-1]['mean_w_kl_preupdate']<cr[0]['mean_w_kl_preupdate'] else '增强'
    text=['# 8192 训练轨迹：ReN 绝对位置审计','',
        f'**观察到的末段信号密度相对首段{direction}；后段仍有信号。仅凭 hit-cap 不能决定扩大训练 horizon。**',
        f"全样本 prompt-balanced 加权 KL mass 的 {ar[1]['balanced_loss_mass_cumulative_share']:.2%} 在前4k、{ar[2]['balanced_loss_mass_cumulative_share']:.2%}在前6k；同一批{capped_count}条 capped 轨迹的前6k占{cr[2]['balanced_loss_mass_cumulative_share']:.2%}，末2k仍占{cr[3]['balanced_loss_mass_share']:.2%}。",
        f"同一 capped cohort 中，每token加权 KL 从0–2k的{cr[0]['mean_w_kl_preupdate']:.5f}变为6–8k的{cr[3]['mean_w_kl_preupdate']:.5f}；{len(round_validation)}轮分别计算时，{lower}轮末段低于首段。",
        '这里的 mass 是已观测 ReN 权重/加权 KL 的质量，不等同于能改善最终答案的因果信号，也不是参数梯度份额。','',
        '仅使用现有日志，未加载模型或使用 GPU。位置是生成序列的绝对下标，k=1024；只计入 reasoning positions。',
        'E[w KL] 使用更新前 w·D_C；本轮一批仅做一次 optimizer update，因此它对应更新起点的目标，而不是更新后的 KL。逐轮重建误差见 round_reconstruction.csv。',
        f"用保存的 w·D_C 重建的逐轮 reasoning loss 比实际训练日志高 {100*result['reconstruction_relative_error_range'][0]:.2f}%–{100*result['reconstruction_relative_error_range'][1]:.2f}%。因此这是更新前 snapshot 的近似审计，不是实际 backward 逐token loss 的精确恢复；当前日志不足以确定数值差异来源。",
        'Teacher 对实际 token 的 logprob 没有保存，无法由 KL 反推，本报告明确缺失；不声称已验证 Teacher-off-policy proxy。','']
    for cohort in ['all','capped_same_cohort']:
        text += ['## '+cohort,'','| 位置 | 轨迹数 | E[w] | E[w KL] | E[DC−DH] | 原始 wKL mass | prompt-balanced loss mass | balanced 累计 |',
                 '|---|---:|---:|---:|---:|---:|---:|---:|']
        for r in result['pooled_bins']:
            if r['cohort']!=cohort:continue
            text.append(f"| {r['bin']} | {r['contributing_rollouts']} | {r['mean_w']:.6f} | {r['mean_w_kl_preupdate']:.6f} | {r['mean_dc_minus_dh']:.6f} | {r['weighted_kl_sum_share']:.2%} | {r['balanced_loss_mass_share']:.2%} | {r['balanced_loss_mass_cumulative_share']:.2%} |")
        text.append('')
    text+=['同一批 capped 轨迹全部走到 8k，其分桶变化更适合判断沿轨迹的衰减；全样本表同时混合了长度、难度和完成状态变化。',
      '带符号的 E[DC−DH] 在各桶均为负；从更负变成较少负值，不代表正向 ReN correction 增强。实际训练使用正部再作 g/(1+g)，需与 E[w]、E[wKL] 一起读。',
      '权重质量、加权 KL 和实际梯度贡献并不相同；前 4–6k 占比用于描述已观察到的信号，不能证明 8k 之后不存在新的信号。',
      '', '[完整分桶与逐轮结果](bins.csv) · [每条轨迹分桶](trajectory_bins.csv) · [原始统计与缺失字段说明](audit.json)']
    (output/'REPORT.md').write_text('\n'.join(text)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7))
    for cohort in ['all','capped_same_cohort']:
        rows=[r for r in result['pooled_bins'] if r['cohort']==cohort]
        for ax,key,label in zip(axes.flat,['mean_w','mean_w_kl_preupdate','mean_dc_minus_dh','balanced_loss_mass_cumulative_share'],
                               ['Mean ReN weight','Mean weighted KL (pre-update)','Mean signed DC - DH','Cumulative prompt-balanced loss mass']):
            ax.plot(range(4),[r[key] for r in rows],marker='o',label=cohort);ax.set_xticks(range(4),['0-2k','2-4k','4-6k','6-8k']);ax.set_title(label);ax.grid(alpha=.2)
    axes[1,1].set_ylim(0,1.05);axes[0,0].legend();fig.tight_layout();fig.savefig(output/'position_audit.png',dpi=180);plt.close(fig)
    print(json.dumps({'pooled_bins':result['pooled_bins'][:8],'capped_paired':paired,'reconstruction':round_validation},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--train',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();audit(a.train.resolve(),a.output.resolve())
