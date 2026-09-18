"""Design-weighted summaries for the bounded offline directional audit."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lulu.training import atomic_json


def mean(x,w):return float(np.dot(np.asarray(x,dtype=float),w)/w.sum()) if w.sum() else None


def quantiles(x,w):
    order=np.argsort(x);xx=x[order];ww=w[order]
    if not len(x) or ww.sum()==0:return {}
    return {str(q):float(np.interp(q,np.cumsum(ww)/ww.sum(),xx)) for q in [.01,.1,.25,.5,.75,.9,.99]}


def summarize(a,mask=None,measure='prompt_balanced'):
    if mask is None:mask=np.ones(len(a['uid']),dtype=bool)
    b={k:v[mask] for k,v in a.items()}
    w=b['position_expansion'].astype(float)
    if measure=='prompt_balanced':w=w/b['reasoning_tokens']
    if not len(w):return {'sampled_positions':0}
    valid=b['alignment_valid'].astype(bool);cos=b['cosine'];beta=b['beta']
    scalar=['cosine','beta','teacher_variance','hindsight_variance','causal_kl','hindsight_kl',
        'resolved_mismatch','old_weight','old_weighted_kl','projected_kl','movement_ratio',
        'teacher_actual_logprob','causal_actual_logprob','hindsight_actual_logprob',
        'causal_entropy','teacher_entropy','projected_entropy','projected_logit_gradient_norm',
        'vanilla_logit_gradient_norm','old_weighted_logit_gradient_norm']
    summary=dict(sampled_positions=len(w),rollouts=len(np.unique(b['uid'])),measure=measure,
        weight_sum=float(w.sum()),means={k:mean(b[k],w) for k in scalar},
        quantiles={k:quantiles(b[k],w) for k in ['cosine','beta','movement_ratio','teacher_variance','hindsight_variance']},
        ratio_of_mean_kl=mean(b['projected_kl'],w)/max(mean(b['causal_kl'],w),1e-30),
        beta_fractions={'zero':mean(beta==0,w),'0_to_0.25_inclusive':mean((beta>0)&(beta<=.25),w),
            '0.25_to_0.5_inclusive':mean((beta>.25)&(beta<=.5),w),'above_0.5':mean(beta>.5,w),'exact_one':mean(beta==1,w)},
        alignment={},old_signal_cross={
            'both':mean((beta>0)&(b['old_weight']>0),w),
            'projection_only':mean((beta>0)&(b['old_weight']==0),w),
            'old_only':mean((beta==0)&(b['old_weight']>0),w),
            'neither':mean((beta==0)&(b['old_weight']==0),w)})
    for threshold in [.01,.05,.1]:
        summary['alignment'][str(threshold)]={
            'negative':mean(valid&(cos < -threshold),w),
            'near_zero':mean(valid&(np.abs(cos)<=threshold),w),
            'positive':mean(valid&(cos>threshold),w),'degenerate':mean(~valid,w)}
    valid_ratio=b['movement_ratio_valid'].astype(bool)
    summary['mean_valid_movement_ratio']=mean(b['movement_ratio'][valid_ratio],w[valid_ratio])
    summary['mean_valid_cosine']=mean(cos[valid],w[valid])
    summary['reconstruction']={
        'mean_old_causal_kl':mean(b['saved_causal_kl'],w),
        'mean_causal_kl':mean(b['causal_kl'],w),
        'causal_kl_mae':mean(np.abs(b['causal_kl']-b['saved_causal_kl']),w),
        'gap_mae':mean(np.abs(b['resolved_mismatch']-b['saved_resolved_mismatch']),w),
        'gap_positive_sign_agreement':mean((b['resolved_mismatch']>0)==(b['saved_resolved_mismatch']>0),w)}
    return summary


def bootstrap(a,draws=2000):
    # Rollout cluster bootstrap stratified by snapshot round. It conditions on
    # sampled positions and does not quantify training-seed uncertainty.
    row=[]
    for uid in np.unique(a['uid']):
        m=a['uid']==uid;w=a['position_expansion'][m]/a['reasoning_tokens'][m]
        valid=a['alignment_valid'][m];cos=a['cosine'][m];beta=a['beta'][m]
        row.append([a['round'][m][0],mean(valid&(cos>.05),w),mean(beta,w),
            mean(beta==0,w),mean(a['projected_kl'][m],w),mean(a['causal_kl'][m],w)])
    rows=np.asarray(row);rng=np.random.default_rng(20260916);stats=[]
    for _ in range(draws):
        selected=np.concatenate([rng.choice(np.flatnonzero(rows[:,0]==r),sum(rows[:,0]==r),replace=True) for r in np.unique(rows[:,0])])
        b=rows[selected].mean(0);stats.append([*b[1:4],b[4]/b[5]])
    return {k:np.quantile(np.asarray(stats)[:,i],[.025,.975]).tolist() for i,k in enumerate(['positive_cosine_fraction','mean_beta','beta_zero_fraction','ratio_of_mean_kl'])}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);x=p.parse_args();out=Path(x.output_dir)
    manifest=json.loads((out/'manifest.json').read_text());records=[json.loads(s) for s in (out/'selected_records.jsonl').read_text().splitlines()]
    files=sorted((out/'scores').glob('rollout_*.npz'))
    if len(files)!=manifest['rollouts']:raise ValueError('Incomplete audit')
    parts=[]
    for path in files:
        with np.load(path) as f:parts.append({k:f[k].copy() for k in f.files})
    a={k:np.concatenate([v[k] for v in parts]) for k in parts[0]}
    assert len(a['uid'])==manifest['positions']
    for r in records:
        m=a['uid']==r['uid']
        assert a['positions'][m].tolist()==r['positions'] and np.all(a['round'][m]==r['snapshot_round'])
        assert np.isclose(a['position_expansion'][m].sum(),r['reasoning_tokens'])
    assert all(np.isfinite(v).all() for v in a.values())
    assert np.all((a['beta']>=0)&(a['beta']<=1))
    assert np.all(a['projected_kl']<=a['causal_kl']+1e-5)
    masks={'all':np.ones(len(a['uid']),bool),'capped':a['capped'],'completed':~a['capped'],
        'correct':a['correct'],'incorrect':~a['correct'],
        'high_alignment':a['alignment_valid']&(a['cosine']>.5),
        'positive_alignment':a['alignment_valid']&(a['cosine']>.05),
        'old_positive':a['old_weight']>0,'old_zero':a['old_weight']==0}
    masks.update({f'round_{r}':a['round']==r for r in manifest['rounds']})
    masks.update({f'bin_{b}':a['position_bin']==b for b in range(4)})
    masks.update({f'capped_bin_{b}':(a['position_bin']==b)&a['capped'] for b in range(4)})
    summaries={k:summarize(a,m) for k,m in masks.items()}
    output=dict(manifest=manifest,primary=summaries,token_global=summarize(a,measure='token_global'),
        bootstrap_95=bootstrap(a),bootstrap_note='2000 stratified rollout-cluster draws; conditional on sampled positions, not training seeds',
        sample_capped=sum(r['truncated'] for r in records),sample_correct=sum(r['correct'] for r in records),
        retained_reasoning_tokens=sum(r['reasoning_tokens'] for r in records))
    atomic_json(out/'summary.json',output)
    atomic_json(out/'validation.json',dict(status='passed',rollouts=len(records),positions=len(a['uid']),
        unique_positions=all(len(r['positions'])==len(set(r['positions'])) for r in records),
        finite=True,beta_bounds=True,geometric_movement_bounded_by_teacher=True,
        expansion_recovers_reasoning_denominator=True,exact_snapshot_and_full_prompt_checked_in_preparation=True,
        file_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files}))
    with (out/'position_metrics.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(a)
        writer.writerows(zip(*(a[k] for k in a)))
    columns=['group','rollouts','positions','negative','near_zero','positive','degenerate','mean_beta','beta_zero','beta_0_025','beta_025_05','beta_above_05','teacher_kl','projected_kl','ratio_of_mean_kl','teacher_actual_logprob']
    with (out/'group_summary.csv').open('w') as f:
        wr=csv.writer(f);wr.writerow(columns)
        for group,s in summaries.items():
            if s['sampled_positions']==0:continue
            z=s['alignment']['0.05'];b=s['beta_fractions'];v=s['means']
            wr.writerow([group,s['rollouts'],s['sampled_positions'],*[z[k] for k in ['negative','near_zero','positive','degenerate']],v['beta'],b['zero'],b['0_to_0.25_inclusive'],b['0.25_to_0.5_inclusive'],b['above_0.5'],v['causal_kl'],v['projected_kl'],s['ratio_of_mean_kl'],v['teacher_actual_logprob']])
    lines=['# Directional-alignment audit — 2026-09-16','',
        '只补做已有 Student 轨迹的前向计算；无新 rollout、无梯度更新。训练 horizon 仍为 8192；后续 evaluation 采用 32768。','',
        f"第 0/4/8 轮各随机抽取 16 条，共 {len(records)} 条 rollout（capped {output['sample_capped']} 条，correct {output['sample_correct']} 条）。每条每个绝对位置 bin 随机抽最多 64 个 reasoning token，共 {len(a['uid']):,} 个位置，代表所抽轨迹的 {output['retained_reasoning_tokens']:,} 个 reasoning token。",'',
        'Student causal/hindsight 使用同轮、同参数、同一条已生成轨迹；hindsight 仅额外得到 gold final answer。Teacher 完全 answer-blind。三个分布均为温度 1 的 full-vocabulary softmax，无 Top-K 概率近似。抽样不按正确性、截断或信号大小筛选。','',
        '主结果按每条 rollout 的 reasoning token 平均，再按 prompt 等权；各位置用 bin 内抽样概率倒数加权。每题一条轨迹。下表 negative/near-zero/positive 分别为 cosine <−0.05 / |cosine|≤0.05 / cosine>0.05。退化范数另计；阈值与抽样清单先于模型评分固定。','',
        '| Group | negative | near zero | positive | degenerate | mean β | β=0 | KL(q*∥C)/KL(T∥C) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for key in ['all','round_0','round_4','round_8','capped','completed','correct','incorrect','high_alignment']:
        s=summaries[key];z=s['alignment']['0.05'];v=s['means']
        lines.append(f"| {key} | {z['negative']:.2%} | {z['near_zero']:.2%} | {z['positive']:.2%} | {z['degenerate']:.2%} | {v['beta']:.4f} | {s['beta_fractions']['zero']:.2%} | {s['ratio_of_mean_kl']:.2%} |")
    overall=summaries['all'];b=overall['beta_fractions'];v=overall['means']
    lines+=['',f"β bins：0 = {b['zero']:.2%}；(0,0.25] = {b['0_to_0.25_inclusive']:.2%}；(0.25,0.5] = {b['0.25_to_0.5_inclusive']:.2%}；(0.5,1] = {b['above_0.5']:.2%}。",'',
        f"E[KL(T∥C)] = {v['causal_kl']:.6f} nats；E[KL(q*∥C)] = {v['projected_kl']:.6f} nats；两者均值的比值 = {overall['ratio_of_mean_kl']:.2%}；逐位置比值的加权均值 = {overall['mean_valid_movement_ratio']:.2%}。这两个 ratio 不是同一个统计量。",'',
        f"Bootstrap 95% CI（按 rollout、按轮次分层）：positive fraction {output['bootstrap_95']['positive_cosine_fraction']}，mean β {output['bootstrap_95']['mean_beta']}，ratio of mean KL {output['bootstrap_95']['ratio_of_mean_kl']}。条件于这次位置抽样，不覆盖不同训练 seed 的不确定性。",'',
        '## 位置与实际 continuation likelihood','',
        '| Positions | mean β | positive cosine | E log qT(actual token) | E log pC(actual token) | KL movement ratio |',
        '|---|---:|---:|---:|---:|---:|']
    for key in ['bin_0','bin_1','bin_2','bin_3','capped_bin_0','capped_bin_1','capped_bin_2','capped_bin_3']:
        s=summaries[key];v=s['means'];z=s['alignment']['0.05']
        lines.append(f"| {key} | {v['beta']:.4f} | {z['positive']:.2%} | {v['teacher_actual_logprob']:.4f} | {v['causal_actual_logprob']:.4f} | {s['ratio_of_mean_kl']:.2%} |")
    lines+=['','位置分桶是 generated-token 的绝对位置 [0,2048)、[2048,4096)、[4096,6144)、[6144,8192)，仅纳入 reasoning positions。all-bin 后段只剩长轨迹；capped-bin 仅比较同一批 capped 轨迹，减少完成时间带来的组成偏差。Teacher actual-token logprob 是本次新补算，不能从旧 scalar KL 恢复。','',
        '## 精度与可复现性','',
        'Student 保留 checkpoint 的 FP32 master weights，用 BF16 autocast；Teacher 使用原有 eager TP2 BF16。log-softmax、centered covariance 与 full-vocabulary reduction 为 FP32。当前复算 batch=1，原日志 batch=2；GPU kernel / padding 可带来数值差异，下面单独报告。', '',
        f"同位置旧日志复算：{json.dumps(overall['reconstruction'],ensure_ascii=False)}",'',
        'manifest.json 保存固定抽样规则与来源 hashes；selected_records.jsonl 保存精确 token IDs 和位置；scores/ 保存每位置全部统计；position_metrics.csv 可直接分析；group_summary.csv 和 summary.json 提供两种归一化与分组。code/ 为启动前冻结代码。验证见 validation.json。','',
        '## 解释边界','',
        '* 审计只检验局部几何假设，不是性能评估；48 条来自三个轮次，不代表全部 12 轮或多 seed。round 之间题目不同，不是同题纵向因果比较。',
        '* β 控制 logit 几何插值，不是概率、KL 或参数步长的线性百分比；β=0 仅让该 state 的 frozen target 等于 C，不能阻止其他 state/control/reference/Adam 通过共享参数改变它。',
        '* cosine 高不等于幅度大：β = cosine × sqrt(VarH/VarT)（忽略 epsilon、clip）。需同时看 β、KL movement 和梯度大小。',
        '* pC 加权度量会降低 Student 低概率、Teacher 高概率候选的影响；这能提高保守性，也可能抑制真正的新知识。正 alignment 也不证明它由正确答案识别能力产生：未做错答案/格式扰动因果对照。',
        '* 旧式 w·KL 已经缩小该 state 的梯度；不能因为 target 是 Teacher 就说一次更新必然全量走向 Teacher。新 target 会改变方向和幅度，仍需后续受控训练验证。',
        '* full-vocabulary forward KL 对 Student logits 的梯度是 p−q，不含必然发散的 1/p 项。已有训练没有梯度爆炸证据；policy rewrite 是待检验的解释。','',
        '## 附件文献核查','',
        '[Veto 原文](https://aclanthology.org/2026.findings-acl.2094.pdf) §4.2 / Algorithm 1 使用 Q∝qT·pS^β（乘积专家，Teacher 指数固定为 1），而此提案是 q*∝pC^(1−β)·qT^β（指数之和为 1）。两者不是同一个 target，不能直接移植该文稳定性/固定点结论。本审计完全按用户提出的 centered pC projection 与 convex logit interpolation 计算。','']
    interpretation=out/'INTERPRETATION.md'
    if interpretation.exists():lines[2:2]=[interpretation.read_text(),'']
    (out/'REPORT.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    w=a['position_expansion']/a['reasoning_tokens'];w=w/w.sum()
    fig,axes=plt.subplots(1,3,figsize=(14,3.6),layout='constrained')
    valid=a['alignment_valid']
    axes[0].hist(a['cosine'][valid],bins=np.linspace(-1,1,41),weights=w[valid],color='#3679a8')
    axes[0].set_title(f'Degenerate states omitted: {np.sum(w[~valid]):.1%}')
    axes[0].axvspan(-.05,.05,color='grey',alpha=.25);axes[0].set(xlabel='Centered pC-weighted cosine',ylabel='Prompt-balanced probability')
    axes[1].hist(a['beta'],bins=np.linspace(0,1,41),weights=w,color='#3b9876');axes[1].set(xlabel='Projected beta')
    axes[2].hist(a['movement_ratio'],bins=np.linspace(0,1,41),weights=w,color='#c38139');axes[2].set(xlabel='KL(projected || C) / KL(Teacher || C)')
    fig.savefig(out/'alignment_distributions.png',dpi=160);plt.close(fig)
    print(json.dumps({k:overall[k] for k in ['alignment','beta_fractions','means','ratio_of_mean_kl','reconstruction']},indent=2))

if __name__=='__main__':main()
