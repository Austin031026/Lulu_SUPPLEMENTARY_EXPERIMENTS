"""Standalone report for the two semantic audits (no training)."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

def main():
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);x=p.parse_args();out=Path(x.output_dir).resolve()
    read=lambda n:json.loads((out/n).read_text())
    spec=read('outcome_specificity.json');util=read('utility_results.json');match=read('matching.json');over=read('outcome_overlap.json');protocol=read('continuation_protocol.json')
    parser_audit=read('parser_disagreement_audit.json')
    s=spec['summary'];u=util['group_metrics'];rows=[]
    labels={'mean_lambda':'E[λ]','positive_fraction':'P(λ>0)','above_025_fraction':'P(λ>0.25)','above_05_fraction':'P(λ>0.5)'}
    for key,label in labels.items():
        a=s[key];scale=1 if key=='mean_lambda' else 100;unit='' if key=='mean_lambda' else '%';deltaunit='' if key=='mean_lambda' else ' pp'
        rows.append(f"| {label} | {a['gold']*scale:.5f}{unit} | {a['wrong']*scale:.5f}{unit} | {a['gold_minus_wrong']*scale:+.5f}{deltaunit} | [{a['paired_rollout_cluster_ci95'][0]*scale:+.5f}, {a['paired_rollout_cluster_ci95'][1]*scale:+.5f}] |")
    round_rows=[]
    for ri,details in spec['by_round_exploratory'].items():
        d=details['mean_lambda'];ci=d['ci95']
        round_rows.append(f"| {ri} | {d['gold']:.5f} | {d['wrong']:.5f} | {d['difference']:+.5f} [{ci[0]:+.5f}, {ci[1]:+.5f}] |")
    utility_rows=[]
    for g in ['high','medium','zero']:
        a=u[g]['legacy'];b=u[g]['strict_final']
        ci=lambda z:f"{100*z['effect']:+.3f} [{100*z['ci95'][0]:+.3f}, {100*z['ci95'][1]:+.3f}]"
        utility_rows.append(f"| {g} | {u[g]['states']} | {u[g]['mean_lambda']:.4f} | {u[g]['mean_changed_mass']*100:.3f}% | {ci(a)} | {ci(b)} |")
    c=util['contrasts']['legacy_high_minus_zero'];cs=util['contrasts']['strict_final_high_minus_zero']
    if c['ci95'][0]>0 and cs['ci95'][0]>0:
        utility_judgment='高 λ 组显示了更大的单步 Teacher 干预收益（两种评分的 high−zero 区间均高于零）。但正确答案特异性未通过，因此这更支持一个可能有用的局部分布代理量，尚不能支持“recognition of correct outcome”的解释。'
    elif c['ci95'][1]<0:
        utility_judgment='高 λ 组的单步 Teacher 干预收益低于零 λ 组；该结果不支持把 alignment 当作更有价值 correction 的选择信号。'
    else:
        utility_judgment='本次未检出高 λ 组有更大单步 Teacher 干预收益的可靠证据。置信区间及有限重复次数不允许把它解释为“所有局部 correction 都无用”。'
    balance=max(abs(v['high_vs_zero_smd_matched_sd']) for v in match['balance'].values())
    text=f'''# Outcome specificity 与 downstream utility 审计

结论：**当前不启动 λ-ReN 训练，也不实现新的 mixture objective。** 正确答案与格式匹配错误答案没有显示出明确的 alignment 差异。{utility_judgment}

本轮只运行这两个审计，没有重新训练、没有 Vanilla OPD baseline，也没有改动现有 checkpoint。内部名称使用 **probability-alignment weight**。

## 1. 正确答案特异性

沿用之前已固定的 48 条 rollout（Round 0/4/8 各 16 条），在完全相同的 10,433 个 reasoning positions 上比较 gold 与 wrong hindsight。每条 rollout 的 wrong answer 来自另一道训练题，数值不等于 gold；正负号格式、字符长度、答案 token 数、完整 hindsight prompt token 数均匹配。随机选择不依赖 probe 分数。

C/H/T hidden states 复用缓存，仅增加 W 的 Student forward。Student 使用 FP32 master weights + BF16 autocast，缓存 C/H/W hidden tensor 为 FP32；Teacher backbone 与缓存 hidden tensor 为 BF16。LM-head matmul 与 softmax 均为 FP32，关闭 TF32。原 gold FP32 λ 重算的最大误差为 {spec['maximum_cached_gold_lambda_reconstruction_error']:.1g}。公式为 `λ=clip(<qT−pC,pH−pC>/(||qT−pC||²+1e−8),0,1)`。

位置按采样概率回权，每条 rollout 内按 reasoning token 平均，再对 prompt 等权。以下 CI 使用 4,000 次配对 rollout-cluster bootstrap，并按 snapshot round 分层；不把 token 当独立样本。

| 指标 | Gold | Wrong | Gold−Wrong | 差值 95% CI |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

补充描述：gold/wrong λ 的加权相关系数为 {over['weighted_pearson']:.4f}，`λ>0` 判定一致率为 {100*over['active_agreement']:.2f}%；gold 高 λ（>0.25）位置中，{100*over['wrong_high_given_gold_high']:.2f}% 在 wrong 条件下也为高 λ。这个条件比例只是描述，不是独立的显著性检验。

**未满足用户要求的 outcome-specificity premise。** “出现稀疏且方向一致的 geometry”本身不能证明 recognition 有正确性语义。结果与一般 privileged-context shift 相容，但这一对照并未单独识别其全部机制；不能宣称已经证明它纯粹是噪声。

各 snapshot 的探索性均值对照如下（每组仅 16 rollout；CI 未作多重比较校正）：

| Round | Gold E[λ] | Wrong E[λ] | 差值 [95% CI] |
|---|---:|---:|---:|
{chr(10).join(round_rows)}

整体差值不能排除 snapshot 间异质性；本轮要求的明确、稳定 specificity 并未显示。

## 2. 固定强度的单步干预

冻结 {match['triplets']} 组三元组，共 {match['states']} states，来自 {match['original_rollouts']} 条原始 rollout。分层为 high λ>0.25、medium 0<λ≤0.25、zero λ=0，每组 72 states。{match['same_rollout_triplets']} 个三元组全部在同一条原始 rollout 内，因而题目、snapshot、原始 capped/completed 状态相同。

匹配 Teacher KL、Student entropy、Teacher TV、最大 token 概率、原 Student 实际 token 概率、绝对位置；位置 bin 相同，任意两者位置差≤768 tokens，连续特征差≤1 个全局 SD，每条 rollout 最多 6 states。high−zero 最大匹配后标准化差异为 **{balance:.4f}**。匹配不读取 wrong probe 或后续 intervention outcome。其余两两比较也低于 0.054。详见 `matching.json` 与 `matching_all_pair_balance.json`。

每个 state 固定 η=0.2：首 token 对照分布为 pC，干预为 `(1−η)pC+ηqT`。首 token 使用完整分布（temperature=1），此后全部使用原 Student thinking 采样：temperature=0.6、top_p=0.95、top_k=20；两臂使用相同续写种子。Teacher 与 gold/wrong answer 不进入 continuation prompt。

为避免约 97% 的共同首 token 概率质量浪费长生成，采用精确的共同质量消去：

- `m=η·TV(pC,qT)`；
- `rC=(pC−qT)+/TV`，`rI=(qT−pC)+/TV`；
- 原问题的收益差恰为 `A=m·(E[success|rI]−E[success|rC])`。

每个 state 在差异部分做两组 paired repetitions，共 {util['generation_jobs']} 条 continuation。首 token 的两臂支持集不相交；common-mass 的贡献在期望中严格为零。**实际报告乘回 m 的原始 mixture 收益差；条件残差两臂准确率不能当成原 Student/mixture 的总体准确率。** 干预强度 η 不依赖 λ；同时匹配 TV，三组平均 m 接近。

总回答预算为 32,768 tokens，包含已存在的 rollout 前缀和强制首 token；另外使用 `min(32768−position−1,40960−Lprompt−position−1−128)` 限制剩余生成。题目与前缀没有截断。测量使用 FP32 head；普通 continuation 使用 vLLM BF16 推理。输出 head 分类采样的归一化采用 FP64，减少小 residual 的舍入误差。

| λ 组 | States | 平均 λ | 平均 m | Legacy 收益 pp [95% CI] | Strict-final 收益 pp [95% CI] |
|---|---:|---:|---:|---:|---:|
{chr(10).join(utility_rows)}

- **High−zero，Legacy：{100*c['effect']:+.3f} pp，95% CI [{100*c['ci95'][0]:+.3f}, {100*c['ci95'][1]:+.3f}] pp。**
- **High−zero，Strict-final：{100*cs['effect']:+.3f} pp，95% CI [{100*cs['ci95'][0]:+.3f}, {100*cs['ci95'][1]:+.3f}] pp。**

Legacy 使用现有 math parser 对完整 response 评分；Strict-final 只对最后一个 `</think>` 后的文本评分，没有闭合 thinking 时判失败。两种评分分歧 {util['generation']['legacy_strict_disagreements']} 条。Verifier error 数为 {len(util['verifier_errors'])}。

评分分歧的进一步核对：{parser_audit['legacy_wrong_final_correct']}/{parser_audit['disagreements']} 条都是 **Legacy 错、final-only 对**。根因并不是 boxed：这 {parser_audit['answer_phrase_only_in_reasoning']} 条在 reasoning 中出现了 `the answer is`，而最后使用 `Answer: 数字`。Legacy 的 `he answer is` 分支会把后续整段 reasoning 加最终解答一起交给 `strip_string`；其中不加限制的 `convert_word_number` 又把整段文字里的英文数词转成了小整数（{parser_audit['word_number_conversion_changes_to_legacy_prediction']} 条重放确认）。例如最终答案 `Answer: 504` 被解析为 `6`，`Answer: 1033` 被解析为 `6` 或 `1`。这是**答案提取的误判**，不是模型给出了错误的最终答案。完整核对见 `parser_disagreement_audit.json`。本次保留事先规定的两套指标；两者都不支持高 λ 有更大的 correction utility，未因结果选择评分方式。这个 56/864 发生率只属于本次条件采样，不可直接用来修正之前的 benchmark 分数；今后解释未 boxed 的回答时，应优先检查 final-only 分数。

续写产生 {util['generation']['total_continuation_tokens']:,} 个新 tokens，hit-cap {util['generation']['hit_cap']}/{util['generation_jobs']}。上述 hit-cap 属于差异部分的条件采样，不能直接和普通 benchmark hit-cap 比较。

## 3. 统计范围与结论边界

CI 以 **39 条原始 rollout** 为 cluster（按 snapshot 分层），三元组、两臂与重复均一同重采样。216 states 不是 216 个独立题目。本结果针对匹配得到的 common-support 状态，不能直接外推所有 training states，也不能跨 benchmark 外推。

每 state 两次配对只足以估计聚合收益及其不确定性，无法稳定标注每个 state 的真实 `A>0`；因此不报告以噪声标签计算的 classification AUC。“固定 η”也不意味着所有 state 的总变差干预相同，故另外匹配并报告 TV/m。

`utility_results.json` 额外列出 within-triplet λ 和旧 w 的探索性 utility slope；小样本、匹配选择及 Monte Carlo 噪声决定了它不能替代独立验证，更不能作为训练优劣的因果结论。

即使 continuation utility 为正，也不证明 parameter-level learnability 或 general capability 不退化。当前第一项 premise 已不满足，故不推进 4/8-round 训练。此次证据也不能反过来证明 Teacher-trajectory selection 必然有效。

## 4. 执行与复现

续写与分析从 15:16:40 到 15:45:55 UTC，约 29 分钟（包含模型加载、编译和失败恢复）。完成后 8 张 GPU 均已释放。

使用现有 vLLM 0.9.0，8 个独立单卡 Student worker，连续批处理、prefix cache、chunked prefill；32 个最大并发序列，4096 batched tokens。三个旧 snapshot 只读加载，不加载 32B backbone；Teacher logits 仅由缓存 hidden state 与 FP32 output head 得到。

首轮并发启动遇到 vLLM/Torch compilation-cache hash 报错。保留成功生成的 worker；失败 worker 在生成前退出，恢复时隔离并关闭 vLLM compile cache，采样方案与随机种子不变。修复记录在 `compile_recovery.json`，没有按 outcome 重试或筛选结果。

验证通过：概率分解、零差异情形、上下文预算等 3 项单元测试；864 条冻结输入逐条检查 prefix 与预算，详见 `validation.json`。

原始配对：`pair_events.jsonl`；冻结状态：`matched_states.jsonl`；冻结 generation 输入与预算：`continuation_jobs.jsonl`；结果：`continuations/task_*.jsonl`；逐 state 估计：`state_utility.csv`；聚合与 CI：`utility_results.json`；gold/wrong 分析：`outcome_specificity.json`。源码快照和 SHA256：`code/`、`analysis_provenance.json`。

![两项审计汇总](semantic_audit_summary.png)
'''
    (out/'REPORT.md').write_text(text)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    names=['E[lambda]','P(lambda>0)','P(lambda>0.25)','P(lambda>0.5)'];keys=list(labels)
    xx=np.arange(4)
    axes[0].bar(xx-.18,[s[k]['gold'] for k in keys],width=.36,label='Gold',color='#3377aa')
    axes[0].bar(xx+.18,[s[k]['wrong'] for k in keys],width=.36,label='Wrong',color='#dd8844')
    axes[0].set_xticks(xx,names,rotation=20);axes[0].set_title('Outcome specificity (48 rollouts)');axes[0].legend();axes[0].set_ylabel('Prompt-balanced statistic')
    for offset,metric,label,color in [(-.08,'legacy','Full-response parser','#3377aa'),(.08,'strict_final','Final-only parser','#dd8844')]:
        mean=np.array([u[g][metric]['effect'] for g in ['high','medium','zero']])*100
        low=np.array([u[g][metric]['ci95'][0] for g in ['high','medium','zero']])*100
        high=np.array([u[g][metric]['ci95'][1] for g in ['high','medium','zero']])*100
        axes[1].errorbar(np.arange(3)+offset,mean,yerr=[mean-low,high-mean],fmt='o',capsize=4,label=label,color=color)
    axes[1].axhline(0,color='grey',lw=1);axes[1].set_xticks(range(3),['High lambda','Medium','Zero']);axes[1].set_ylabel('Original mixture success difference (pp)');axes[1].set_title('Fixed eta=0.2 intervention; cluster 95% CI');axes[1].legend(fontsize=8)
    fig.savefig(out/'semantic_audit_summary.png',dpi=180);plt.close(fig)
    provenance={}
    for name in ['analyze_semantic_utility.py','analyze_outcome_specificity.py','audit_utility_parser.py','finalize_semantic_audit.py','report_semantic_utility.py']:
        src=Path(__file__).with_name(name);shutil.copy2(src,out/'code'/name)
    for name in ['benchmark_parser.py','math_parser.py']:
        src=Path(__file__).parents[1]/'lulu'/name;shutil.copy2(src,out/'code'/'lulu'/name)
    for f in sorted((out/'code').rglob('*.py')):provenance[str(f.relative_to(out))]=hashlib.sha256(f.read_bytes()).hexdigest()
    for name in ['continuation_jobs.jsonl','matched_states.jsonl','pair_events.jsonl','utility_results.json','outcome_specificity.json','parser_disagreement_audit.json','validation.json','REPORT.md']:
        provenance[name]=hashlib.sha256((out/name).read_bytes()).hexdigest()
    (out/'analysis_provenance.json').write_text(json.dumps(provenance,indent=2));print(out/'REPORT.md')
if __name__=='__main__':main()
