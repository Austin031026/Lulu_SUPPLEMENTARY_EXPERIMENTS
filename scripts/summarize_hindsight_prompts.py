"""Paired prompt audit statistics, with prompt-cluster bootstrap intervals."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lulu.hindsight_prompt_audit import PROMPTS
from lulu.training import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);a=p.parse_args()
    out=Path(a.output_dir);manifest=json.loads((out/'manifest.json').read_text())
    records=[json.loads(s) for s in (out/'selected_records.jsonl').read_text().splitlines()]
    metrics=('kl_h_c','kl_c_h','tv_h_c','teacher_tv','shared_mass','shared_structural_mass',
             'shared_nonstructural_mass','structural_l1','teacher_positive_h_uplift','supported_teacher_positive_mass')
    tables=[];bins=[];saved=[]
    for r in records:
        with np.load(out/'scores'/f'rollout_{r["uid"]:03d}.npz') as d:
            if not np.array_equal(d['positions'],r['positions']):raise ValueError('Position mismatch')
            w=d['position_expansion'].astype(float)
            for region in (0,1):
                keep=d['region']==region;n=w[keep].sum()
                expected=r['reasoning_tokens'] if region else r['control_tokens']
                if not np.isclose(n,expected):raise ValueError('Expansion weights do not recover population')
                for name in PROMPTS:
                    row=dict(uid=r['uid'],snapshot=r['snapshot_round'],prompt=r['source_id'],truncated=r['truncated'],
                             view=name,region='reasoning' if region else 'control',tokens=int(round(n)),sampled=int(keep.sum()))
                    row.update({key:float(np.dot(w[keep],d[name+'__'+key][keep])/n) if n else 0. for key in metrics})
                    tables.append(row)
                    for b in range(4):
                        sel=keep & (d['position_bin']==b);nb=w[sel].sum()
                        if nb:
                            bins.append(dict(uid=r['uid'],snapshot=r['snapshot_round'],view=name,region=row['region'],bin=b,tokens=nb,
                                **{key:float(np.dot(w[sel],d[name+'__'+key][sel])/nb) for key in metrics}))
            keep=d['region']==1
            saved.append(dict(uid=r['uid'],saved=float(np.dot(w[keep],d['saved_shared_mass'][keep])/w[keep].sum()),
                replay=float(np.dot(w[keep],d['current__shared_mass'][keep])/w[keep].sum())))
    with (out/'rollout_metrics.csv').open('w') as f:
        wr=csv.DictWriter(f,fieldnames=list(tables[0]));wr.writeheader();wr.writerows(tables)
    with (out/'position_bins.csv').open('w') as f:
        wr=csv.DictWriter(f,fieldnames=list(bins[0]));wr.writeheader();wr.writerows(bins)
    def aggregate(rows,mode):
        rows=[r for r in rows if r['tokens']]
        w=np.array([r['tokens'] if mode=='token' else 1. for r in rows])
        result={key:float(np.average([r[key] for r in rows],weights=w)) for key in metrics}
        result.update(rollouts=len(rows),tokens=sum(r['tokens'] for r in rows))
        for key,denom in [('teacher_correction_coverage','teacher_tv'),('hindsight_overlap_efficiency','tv_h_c')]:
            result[key]=result['shared_mass']/result[denom] if result[denom] else 0.
        result['structural_shared_fraction']=result['shared_structural_mass']/result['shared_mass'] if result['shared_mass'] else 0.
        return result
    aggregate_results={}
    for snapshot in ('all',0,2):
        aggregate_results[str(snapshot)]={}
        for region in ('reasoning','control'):
            aggregate_results[str(snapshot)][region]={}
            for mode in ('prompt','token'):
                aggregate_results[str(snapshot)][region][mode]={name:aggregate([r for r in tables if r['view']==name and
                    r['region']==region and (snapshot=='all' or r['snapshot']==snapshot)],mode) for name in PROMPTS}
    # Same sampled states in each view; resample prompt clusters within snapshot.
    rng=np.random.default_rng(20260918);boots=5000
    bootstrap_indices=np.concatenate([rng.choice(np.flatnonzero(np.array([r['snapshot_round'] for r in records])==rnd),
        size=(boots,64),replace=True) for rnd in (0,2)],axis=1)
    paired={}
    for name in ('minimal','opsd'):
        paired[name]={}
        for region in ('reasoning','control'):
            cur=[r for r in tables if r['view']=='current' and r['region']==region]
            alt=[r for r in tables if r['view']==name and r['region']==region]
            for key in metrics:
                w=np.array([1. if region=='reasoning' else r['tokens'] for r in cur])
                c=np.array([r[key] for r in cur]);h=np.array([r[key] for r in alt])
                bw=w[bootstrap_indices];bc=(bw*c[bootstrap_indices]).sum(1)/bw.sum(1);bh=(bw*h[bootstrap_indices]).sum(1)/bw.sum(1)
                delta=bh-bc;ratio=bh/np.maximum(bc,1e-30)
                paired[name][region+'__'+key]=dict(delta=float(np.average(h-c,weights=w)),
                    delta_ci95=np.quantile(delta,[.025,.975]).tolist(),ratio=float(np.average(h,weights=w)/np.average(c,weights=w)),
                    ratio_ci95=np.quantile(ratio,[.025,.975]).tolist())
    reason=aggregate_results['all']['reasoning']['prompt'];control=aggregate_results['all']['control']['token']
    cur,alt=reason['current'],reason['minimal'];cc,ac=control['current'],control['minimal']
    rules=manifest['decision_rule']
    checks=dict(reasoning_shared_gain=alt['shared_mass']/cur['shared_mass']>=1+rules['minimum_reasoning_shared_gain'],
        paired_gain_positive=paired['minimal']['reasoning__shared_mass']['delta_ci95'][0]>0,
        both_snapshots_positive=all(aggregate_results[str(rnd)]['reasoning']['prompt']['minimal']['shared_mass']>
            aggregate_results[str(rnd)]['reasoning']['prompt']['current']['shared_mass'] for rnd in (0,2)),
        overlap_efficiency=alt['hindsight_overlap_efficiency']/cur['hindsight_overlap_efficiency']>=rules['minimum_overlap_efficiency_ratio'],
        control_kl=ac['kl_h_c']/cc['kl_h_c']<=rules['maximum_control_kl_ratio'] and ac['kl_h_c']-cc['kl_h_c']<=rules['maximum_control_kl_absolute_increase'],
        control_tv=ac['tv_h_c']/cc['tv_h_c']<=rules['maximum_control_tv_ratio'] and ac['tv_h_c']-cc['tv_h_c']<=rules['maximum_control_tv_absolute_increase'],
        structural_l1=alt['structural_l1']/cur['structural_l1']<=rules['maximum_reasoning_structural_l1_ratio'],
        nonstructural_shared_gain=alt['shared_nonstructural_mass']/cur['shared_nonstructural_mass']>=1+rules['minimum_nonstructural_shared_gain'])
    replay=dict(saved_mean=float(np.mean([r['saved'] for r in saved])),replayed_mean=float(np.mean([r['replay'] for r in saved])),
        mean_absolute_rollout_difference=float(np.mean([abs(r['saved']-r['replay']) for r in saved])))
    result=dict(manifest=manifest,aggregates=aggregate_results,paired=paired,replay=replay,decision_checks=checks,
        launch_minimal_pilot=all(checks.values()),bootstrap=dict(replicates=boots,unit='prompt',stratification='snapshot 0/2',
        limitations='Conditional on these checkpoints and sampled positions; no model/seed uncertainty; no additional position-level resampling.'))
    atomic_json(out/'summary.json',result)
    lines=['# Fixed-prefix hindsight prompt comparison','',
        f'配对结果：Minimal shared mass 相对 Current {(reason["minimal"]["shared_mass"]/reason["current"]["shared_mass"]-1)*100:+.2f}%；OPSD-style {(reason["opsd"]["shared_mass"]/reason["current"]["shared_mass"]-1)*100:+.2f}%。是否触发 pilot 见下方筛选；审计程序本身不启动训练。','',
        '没有生成新 rollout，也没有训练。使用原 shared-positive 实验中 Base / Round2 快照下全部 128 条轨迹；同题、同前缀、同位置配对。',
        f'共 {manifest["sampled_positions"]:,} 个位置，每个非空 2k bin 抽取至多 128 个 reasoning、32 个 control 位置。逆抽样概率加权；完整前缀不截断。',
        'reasoning 主口径为 token → rollout → prompt 平均；control 主口径为全 token 平均。以下概率质量单位均为百分点，KL 单位为 nat。','',
        '| Reasoning 指标 | Current | Minimal | OPSD |','|---|---:|---:|---:|']
    for label,key,scale in [('KL(H‖C)','kl_h_c',1),('KL(C‖H)','kl_c_h',1),('TV(H,C)','tv_h_c',100),
        ('Shared correction mass m','shared_mass',100),('m / Teacher TV','teacher_correction_coverage',100),
        ('m / Hindsight TV','hindsight_overlap_efficiency',100),('非结构 token shared mass','shared_nonstructural_mass',100),
        ('结构 token 占 shared mass','structural_shared_fraction',100),('结构 token L1','structural_l1',100)]:
        lines.append('| '+label+' | '+' | '.join(f'{reason[n][key]*scale:.6f}' for n in PROMPTS)+' |')
    lines+=['','| Control 指标 | Current | Minimal | OPSD |','|---|---:|---:|---:|']
    for label,key,scale in [('KL(H‖C)','kl_h_c',1),('TV(H,C)','tv_h_c',100),('结构 token L1','structural_l1',100)]:
        lines.append('| '+label+' | '+' | '.join(f'{control[n][key]*scale:.6f}' for n in PROMPTS)+' |')
    lines+=['','| Snapshot | Current m | Minimal m | OPSD m |','|---|---:|---:|---:|']
    for rnd in (0,2):
        rr=aggregate_results[str(rnd)]['reasoning']['prompt']
        lines.append(f'| {rnd} | '+' | '.join(f'{rr[n]["shared_mass"]*100:.6f}%' for n in PROMPTS)+' |')
    lines+=['','配对 95% bootstrap 区间（5,000 次，按 snapshot 分层重采样 prompt）：','']
    for name in ('minimal','opsd'):
        r=paired[name]['reasoning__shared_mass']
        lines.append(f'- {name}: Δm={r["delta"]*100:+.6f} pp，95% CI [{r["delta_ci95"][0]*100:+.6f}, {r["delta_ci95"][1]*100:+.6f}]；比例={r["ratio"]:.4f}。')
    lines+=['','## 运行前写入 manifest 的 pilot 筛选','',
        '这些是节省计算的操作阈值，并非经验证的科学显著性标准：m 至少增加 20%、配对增量 CI 下界 >0、两个快照均增加；m/TV(H,C) 至少保留 90%；control KL/TV 不超过 1.5 倍且绝对增加分别 ≤0.05 nat / 0.02；结构 token L1 ≤1.5 倍；非结构 shared mass 至少增加 20%。','']
    lines.extend(f'- {k}: {v}' for k,v in checks.items())
    lines+=['',f'筛选结论：{"满足 Minimal pilot 条件" if all(checks.values()) else "不满足 Minimal pilot 条件，不自动启动新训练"}。','',
        '## 解释边界','',
        '- Shared mass 是方向重叠，不等价于正确推理、下游收益或实际参数更新大小。',
        '- 结构代理只包含特殊 token、空白及纯标点/符号 token，也会包含数学符号；不能据此判断所有 style 变化。',
        '- control 与 reasoning 是位置类别；结构/非结构是词表 action 类别，两者分别统计。',
        '- 区间只覆盖固定轨迹上的 prompt 间差异，未包含新训练 seed 或额外位置抽样不确定性。',
        '- 本次只比较 prompt 文本，角色仍为 user 末尾追加，thinking template 不变，Teacher 完全不见答案。',
        f'- Current 回放检查：旧缓存同位置 prompt-balanced m={replay["saved_mean"]:.8f}，本次={replay["replayed_mean"]:.8f}；单题平均绝对差={replay["mean_absolute_rollout_difference"]:.8f}。总体均值接近不表示逐题完全一致，单题平均绝对差为旧均值的 {replay["mean_absolute_rollout_difference"]/replay["saved_mean"]:.2%}。原训练 batch=2、本次 Student 单条前向，BF16 批形状等数值差异仍需保留为限制；三个 prompt 的主比较均使用本次相同 C/T，不把旧缓存当比较组。','',
        '三个 prompt 原文见 `manifest.json`；逐题数据见 `rollout_metrics.csv`；分位置数据见 `position_bins.csv`。','']
    (out/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps(dict(reasoning=reason,control=control,checks=checks,launch_minimal_pilot=all(checks.values())),indent=2))

if __name__=='__main__':main()
