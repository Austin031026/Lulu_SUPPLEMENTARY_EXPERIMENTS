"""Summarize the fixed-position probability-projection audit, without GPU work."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from lulu.training import atomic_json


def weighted_mean(x,w):return float(np.average(np.asarray(x,dtype=float),weights=w)) if w.sum() else None


def describe(a,mask=None,prefix='',token_global=False):
    if mask is None:mask=np.ones(len(a['uid']),dtype=bool)
    w=a['position_expansion'][mask].astype(float)
    if not token_global:w=w/a['reasoning_tokens'][mask]
    def v(key):return a[prefix+key][mask]
    def m(x):return weighted_mean(x,w)
    if len(w)==0:return dict(positions=0)
    lam=v('lambda_');cos=v('probability_cosine');valid=v('alignment_valid')
    scalar=['lambda_','certified_kl','causal_kl','old_weight','old_weighted_kl','certified_gradient_norm',
            'vanilla_gradient_norm','old_weighted_gradient_norm','dot_product','teacher_delta_squared',
            'hindsight_delta_squared','causal_entropy']
    mean={k:m(v(k)) for k in scalar}
    return dict(positions=len(w),rollouts=len(np.unique(a['uid'][mask])),weight_sum=float(w.sum()),means=mean,
        lambda_zero=m(lam==0),lambda_above_025=m(lam>.25),lambda_above_05=m(lam>.5),lambda_one=m(lam==1),
        lambda_bins={'zero':m(lam==0),'0_to_025':m((lam>0)&(lam<=.25)),'025_to_05':m((lam>.25)&(lam<=.5)),'above_05':m(lam>.5)},
        ratio_mean_kl=mean['certified_kl']/max(mean['causal_kl'],1e-30),
        ratio_mean_gradient_to_vanilla=mean['certified_gradient_norm']/max(mean['vanilla_gradient_norm'],1e-30),
        ratio_mean_gradient_to_current=mean['certified_gradient_norm']/max(mean['old_weighted_gradient_norm'],1e-30),
        alignment={'negative':m(valid&(cos<-.05)),'near_zero':m(valid&(np.abs(cos)<=.05)),
            'positive':m(valid&(cos>.05)),'degenerate':m(~valid)},
        ratio_valid_fraction=m(v('gradient_ratio_valid')),
        projected_gt_current_fraction=m(v('certified_gradient_norm')>v('old_weighted_gradient_norm')+1e-12))


def numerical_validation(a,prefix=''):
    valid=a[prefix+'gradient_ratio_valid']
    v=lambda k:a[prefix+k]
    checks=dict(max_gradient_identity_error_fp64=float(v('gradient_identity_error_fp64').max()),
        max_gradient_bound_excess_fp64=float(v('gradient_bound_excess_fp64').max()),
        max_ratio_error_fp64_above_norm_floor=float(v('gradient_ratio_error_fp64')[valid].max()),
        max_gradient_identity_error_fp32=float(v('gradient_identity_error_fp32').max()),
        max_ratio_error_fp32_above_norm_floor=float(np.abs(v('gradient_ratio_fp32')-v('lambda_'))[valid].max()),
        max_target_sum_error=float(v('target_sum_error').max()),minimum_target_probability=float(v('target_min_probability').min()),
        positions_above_ratio_norm_floor=int(valid.sum()),positions_below_ratio_norm_floor=int((~valid).sum()),
        exact_zero_teacher_direction_positions=int((v('teacher_delta_squared')==0).sum()))
    assert checks['max_gradient_identity_error_fp64']<1e-12
    assert checks['max_gradient_bound_excess_fp64']<1e-12
    assert checks['max_ratio_error_fp64_above_norm_floor']<1e-8
    assert checks['max_gradient_identity_error_fp32']<5e-7
    assert checks['max_target_sum_error']<2e-6
    assert checks['minimum_target_probability']>=0
    assert np.all(v('certified_kl')<=v('lambda_')*v('causal_kl')+2e-6)
    assert np.all((v('lambda_')>=0)&(v('lambda_')<=1))
    return checks


def bootstrap(a):
    rows=[]
    for uid in np.unique(a['uid']):
        m=a['uid']==uid;s=describe(a,m);v=s['means']
        rows.append([a['round'][m][0],s['lambda_zero'],v['lambda_'],s['lambda_above_025'],s['lambda_above_05'],v['certified_kl'],v['certified_gradient_norm'],v['old_weighted_gradient_norm']])
    rows=np.asarray(rows);rng=np.random.default_rng(20260916);draws=[]
    for _ in range(2000):
        idx=np.concatenate([rng.choice(np.flatnonzero(rows[:,0]==r),sum(rows[:,0]==r),replace=True) for r in np.unique(rows[:,0])])
        s=rows[idx].mean(0);draws.append([*s[1:6],s[6]/s[7]])
    return {k:np.quantile(np.asarray(draws)[:,i],[.025,.975]).tolist() for i,k in enumerate(['lambda_zero','mean_lambda','lambda_above_025','lambda_above_05','mean_target_kl','gradient_ratio_to_current'])}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',required=True);args=parser.parse_args();out=Path(args.output_dir)
    manifest=json.loads((out/'manifest.json').read_text());parts=[];files=sorted((out/'scores').glob('rollout_*.npz'))
    assert len(files)==manifest['rollouts']
    records=[json.loads(s) for s in (out/'selected_records.jsonl').read_text().splitlines()]
    for f in files:
        with np.load(f) as z:parts.append({k:z[k].copy() for k in z.files})
    a={k:np.concatenate([p[k] for p in parts]) for k in parts[0]}
    assert len(a['uid'])==manifest['positions']
    assert all(np.isfinite(v).all() for v in a.values())
    for r in records:
        sel=a['uid']==r['uid'];assert a['positions'][sel].tolist()==r['positions']
        assert np.all(a['round'][sel]==r['snapshot_round'])
        assert np.isclose(a['position_expansion'][sel].sum(),r['reasoning_tokens'])
    validation={'status':'passed','rollouts':len(records),'positions':len(a['uid']),
        'convex_kl_bound_checked':True,
        'same_source_selection':hashlib.sha256((out/'selected_records.jsonl').read_bytes()).hexdigest()==manifest['selected_records_sha256'],
        'bf16_head':numerical_validation(a),'fp32_head':numerical_validation(a,'fp32head_')}
    atomic_json(out/'validation.json',validation)
    masks={'all':np.ones(len(a['uid']),bool),'capped':a['capped'],'completed':~a['capped'],'correct':a['correct'],'incorrect':~a['correct']}
    masks.update({f'round_{r}':a['round']==r for r in [0,4,8]})
    masks.update({f'bin_{b}':a['position_bin']==b for b in range(4)})
    masks.update({f'capped_bin_{b}':(a['position_bin']==b)&a['capped'] for b in range(4)})
    masks.update(high_lambda=a['lambda_']>.5,high_alignment=a['alignment_valid']&(a['probability_cosine']>.5))
    primary={k:describe(a,m) for k,m in masks.items()};secondary={k:describe(a,m,'fp32head_') for k,m in masks.items()}
    w=a['position_expansion']/a['reasoning_tokens'];m=lambda x:weighted_mean(x,w)
    gradient=a['certified_gradient_norm'];movement=a['certified_kl'];fpgrad=a['fp32head_certified_gradient_norm']
    lam=a['lambda_'];fplam=a['fp32head_lambda_']
    sensitivity=dict(mean_abs_lambda_change=m(np.abs(fplam-lam)),active_sign_agreement=m((fplam>0)==(lam>0)),
        mean_abs_gradient_norm_change=m(np.abs(fpgrad-gradient)),
        mean_abs_gradient_norm_change_relative_to_primary_mean=m(np.abs(fpgrad-gradient))/m(gradient),
        movement_relative_change=m(a['fp32head_certified_kl'])/m(movement)-1,
        gradient_mean_relative_change=m(fpgrad)/m(gradient)-1,
        epsilon={})
    for e in ['1e-10','1e-6','1e-4']:
        sensitivity['epsilon'][e]={'mean_lambda':m(a['lambda_eps_'+e]),'mean_gradient_norm':m(a['gradient_norm_eps_'+e]),
            'relative_gradient_mean_change':m(a['gradient_norm_eps_'+e])/m(gradient)-1}
    shares=dict(capped_rollout_fraction=sum(r['truncated'] for r in records)/len(records),
        capped_target_movement=float(np.sum(w*movement*a['capped'])/np.sum(w*movement)),
        capped_gradient_norm=float(np.sum(w*gradient*a['capped'])/np.sum(w*gradient)),
        high_lambda_target_movement=float(np.sum(w*movement*(lam>.5))/np.sum(w*movement)),
        high_lambda_gradient_norm=float(np.sum(w*gradient*(lam>.5))/np.sum(w*gradient)),
        correction_on_old_zero=float(np.sum(w*gradient*(a['old_weight']==0))/np.sum(w*gradient)))
    summary=dict(group_definitions={'high_lambda':'Defined by PRIMARY BF16 lambda > 0.5, held fixed for precision comparison','high_alignment':'Defined by PRIMARY BF16 probability cosine > 0.5, held fixed for precision comparison'},primary=primary,fp32_head=secondary,token_global=describe(a,token_global=True),
        sensitivity=sensitivity,shares=shares,bootstrap_95=bootstrap(a),validation=validation,
        interpretation_scope='Mean per-position logit norm, not accumulated parameter norm or optimizer step; bootstrap conditional on sampled positions and training run')
    atomic_json(out/'summary.json',summary)
    with (out/'position_metrics.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(a);writer.writerows(zip(*(a[k] for k in a)))
    with (out/'group_summary.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['group','precision','lambda_zero','mean_lambda','lambda_gt_025','lambda_gt_05','target_kl','teacher_kl','gradient_ratio_to_vanilla','gradient_ratio_to_current'])
        for precision,groups in [('bf16head',primary),('fp32head',secondary)]:
            for group,s in groups.items():
                if not s['positions']:continue
                writer.writerow([group,precision,s['lambda_zero'],s['means']['lambda_'],s['lambda_above_025'],s['lambda_above_05'],s['means']['certified_kl'],s['means']['causal_kl'],s['ratio_mean_gradient_to_vanilla'],s['ratio_mean_gradient_to_current']])
    lines=['# Certified-Correction probability-space audit — 2026-09-16','',
        '**本轮仅执行 offline audit；无新 rollout、无训练更新。** 精确复用上一轮 48 条轨迹、10,433 个 reasoning positions 与原来的抽样权重。Student snapshot 0/4/8 各 16 条。', '',
        '计算 uT=qT−pC、uH=pH−pC，λ=clip(〈uH,uT〉/(||uT||²+1e−8),0,1)，target=(1−λ)pC+λqT。不加入额外稀疏门槛，不做 weight renormalization；Teacher answer-blind，Student causal/hindsight 同参数、同已生成前缀。', '',
        '主结果：Student FP32 master weights + BF16 backbone/head compute，Teacher 为两份独立单卡 BF16 副本；softmax/概率运算 FP32。按 reasoning-token mean within rollout，再按 prompt 等权，校正每 bin 位置抽样概率。', '',
        '| Group / head precision | λ=0 | mean λ | λ>0.25 | λ>0.5 | E KL(target∥C) | mean norm / Vanilla | mean norm / current ReN |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for label,groups,keys in [('BF16',primary,['all','round_0','round_4','round_8','capped','completed']),('FP32',secondary,['all'])]:
        for key in keys:
            s=groups[key];v=s['means']
            lines.append(f"| {key} / {label} | {s['lambda_zero']:.2%} | {v['lambda_']:.5f} | {s['lambda_above_025']:.2%} | {s['lambda_above_05']:.2%} | {v['certified_kl']:.6f} | {s['ratio_mean_gradient_to_vanilla']:.2%} | {s['ratio_mean_gradient_to_current']:.3f}× |")
    v=validation['bf16_head'];s=primary['all']
    lines+=['','## 逐位置 implementation audit','',
        f"全部 {len(a['uid']):,} 个位置通过非负/归一化 target、λ∈[0,1]、gradient identity 与相对 Vanilla 的范数上界检查。构造 FP64 显式 mixture 后，max ||(pC−q*)−λ(pC−qT)||₂ = {v['max_gradient_identity_error_fp64']:.3g}；max 上界超出量 = {v['max_gradient_bound_excess_fp64']:.3g}。",'',
        f"||qT−pC||>1e−7 的 {v['positions_above_ratio_norm_floor']:,} 个位置，显式 norm ratio 与 λ 的最大绝对误差 {v['max_ratio_error_fp64_above_norm_floor']:.3g}。其余 {v['positions_below_ratio_norm_floor']:,} 个位置只报告 absolute gradient identity；0/0 或接近零的 ratio 不赋予意义。1e−7 仅用于报告，不参与 λ 或 target 构造。",'',
        f"直接 FP32 构造 target 后相减会有 cancellation：max absolute vector error = {v['max_gradient_identity_error_fp32']:.3g}；非退化位置 max ratio error = {v['max_ratio_error_fp32_above_norm_floor']:.3g}。因此不能声称浮点 ratio 在所有接近零分母的位置逐位相等；数学恒等式、绝对误差与独立 autograd 单元测试共同验证实现。",'',
        '## 相同 hidden states 的输出头精度检查','',
        '所有位置均额外使用 FP32 输出头矩阵乘法（TF32 关闭）复算；没有再跑 backbone、没有重新采样。Teacher FP32 head 是原 BF16 weights 转为 FP32，并不能恢复加载前精度。Student 使用已保存 FP32 head weights。此检查不覆盖 full-FP32 backbone、batch/padding 差异。', '',
        f"mean |Δλ|={sensitivity['mean_abs_lambda_change']:.6f}；λ>0 的一致率 {sensitivity['active_sign_agreement']:.2%}；平均梯度范数变化 {sensitivity['gradient_mean_relative_change']:+.2%}；平均 KL movement 变化 {sensitivity['movement_relative_change']:+.2%}。逐位置 gradient-norm 的平均绝对变化占原均值 {sensitivity['mean_abs_gradient_norm_change_relative_to_primary_mean']:.2%}。",'',
        'ε 敏感性（仅重算保存的 dot / squared norm，未据结果调参）：','',
        '| ε | mean λ | mean gradient norm | gradient mean change vs 1e−8 |','|---|---:|---:|---:|']
    for e,value in sensitivity['epsilon'].items():lines.append(f"| {e} | {value['mean_lambda']:.5f} | {value['mean_gradient_norm']:.6f} | {value['relative_gradient_mean_change']:+.2%} |")
    lines+=['','## Capped 与稀疏 correction','',
        f"capped rollout 占 {shares['capped_rollout_fraction']:.2%}，target movement share 为 {shares['capped_target_movement']:.2%}，mean-position gradient-norm share 为 {shares['capped_gradient_norm']:.2%}。λ>0.5 的位置承载 {shares['high_lambda_target_movement']:.2%} movement。旧 w=0 位置承载新 gradient-norm mass 的 {shares['correction_on_old_zero']:.2%}。这些是诊断量，不能当作参数梯度 share；未过滤 capped/失败样本。",'',
        '## 必须修正的理论解释','',
        '1. 初始每位置 g_new=λg_Vanilla 的保证成立；它不保证 g_new 小于 w·g_Vanilla，因为 λ 未必≤w。上一轮 1.48× 的比较对象是 current weighted ReN，不能拿它与 Vanilla 的上界混淆。不同位置缩放还会改变 batch 梯度的抵消，不能从逐位置上界推出累计参数梯度/Adam 步长上界。','',
        '2. 每轮 snapshot / target 固定时，独立 state 的最优点为 q*。跨轮 refresh 则并不固定这个端点：假设固定 state / Teacher 和 λ，若每轮拟合到 target，则 p_r=qT+(1−λ)^r(p_0−qT)，只要 λ>0 就仍趋向 Teacher。现实还有共享参数和 control/reference，因此也没有能力保留保证。','',
        '3. 固定 λ 和 pC 时，新 KL 与 λ·KL(qT∥pθ)+(1−λ)·KL(pC∥pθ) 只差一个与 θ 无关的常数。当前训练 update_passes=1，在 pθ=pC 起点 self-anchor 梯度为零，所以首次完整梯度与 λ-weighted Teacher KL 相同。相同 control/reference、normalization、optimizer state 下，一次参数更新也相同（数学精确、忽略数值差异）。这次若与 current ReN 比较，首先检验的是 λ 的定义；不能将结果直接归因于更受限的终点。','',
        '4. 之前 36.07% 是 centered log-prob / pC-weighted cosine；这里换了 probability-difference 空间，正向比例必须重新计算。即使方向对齐，也不是正确纠错的数学认证，仍需 task performance 验证。', '',
        '## 可复现与局限','',
        'manifest.json 固定输入 hashes、原选样文件 hash、ε、验证容差和 precision 分支；code/ 保存启动时源码。hidden/ 保留所选位置的 C/H/T states（不保存巨大 full-vocabulary logits）。scores/、position_metrics.csv、group_summary.csv、summary.json、validation.json 保存完整诊断。训练 horizon 8192、未来 sampled dev/eval 32768。', '',
        'Bootstrap 为按 round 分层的 2000 次 rollout cluster bootstrap，条件于本次位置抽样与同一训练 run，不代表多 seed 或新训练效果。summary.json 保存区间。', '']
    if (out/'INTERPRETATION.md').exists():lines[2:2]=[(out/'INTERPRETATION.md').read_text(),'']
    (out/'REPORT.md').write_text('\n'.join(lines))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,3.8),layout='constrained')
    for label,pre,color in [('BF16 head','', '#327aa0'),('FP32 head','fp32head_', '#c38a41')]:
        axes[0].hist(a[pre+'lambda_'],bins=np.linspace(0,1,41),weights=w/w.sum(),histtype='step',label=label,color=color)
    axes[0].set(xlabel='Probability projection lambda',ylabel='Prompt-balanced probability');axes[0].legend()
    valid=a['gradient_ratio_valid'];axes[1].scatter(lam[valid],a['gradient_ratio_fp64'][valid],s=2,alpha=.25)
    axes[1].plot([0,1],[0,1],color='black',ls='--');axes[1].set(xlabel='lambda',ylabel='Explicit initial gradient ratio / Vanilla')
    keys=['round_0','round_4','round_8'];axes[2].bar(['0','4','8'],[primary[k]['ratio_mean_gradient_to_current'] for k in keys],color='#529b7d')
    axes[2].axhline(1,color='black',ls='--');axes[2].set(xlabel='Snapshot round',ylabel='Mean gradient norm / current weighted ReN')
    fig.savefig(out/'certified_audit.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(primary=primary['all'],fp32_head=secondary['all'],validation=validation,sensitivity=sensitivity,shares=shares),indent=2))

if __name__=='__main__':main()
