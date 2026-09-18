"""Join fixed audits and the three diagnostic arms; no checkpoint selection."""
from pathlib import Path
import argparse,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from evaluate_lulu import load_complete_rows,summarize_rows,paired_comparison
from run_decisive import write_json

def load_evaluation_rows(root,name,dataset,expected_examples):
    folder=root/('base_dev' if name=='base' else 'evaluation')
    plan=json.loads((folder/'eval_plan.json').read_text())
    return load_complete_rows(folder/name/dataset,len(plan['devices']),expected_examples)

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment-dir',required=True);a=p.parse_args();root=Path(a.experiment_dir)
    plan=json.loads((root/'experiment_plan.json').read_text());ep=json.loads((root/'evaluation/eval_plan.json').read_text());source=Path(plan['source_16k'])
    scores={};rows={};bench=[b['name'] for b in ep['benchmarks']];external=[b for b in bench if b!='dapo_dev128'];rng=np.random.default_rng(20260917)
    for name in ['base',*plan['arms']]:
        scores[name]={};rows[name]={}
        for b in ep['benchmarks']:
            ds=b['name'];n=b['expected_examples'];assert n is not None
            if name=='base' and ds!='dapo_dev128':rr=[json.loads(x) for x in (source/f'reference_32k/base/{ds}/rows.jsonl').read_text().splitlines()]
            else:rr=load_evaluation_rows(root,name,ds,n)
            assert len(rr)==n
            if name!='base':
                for x,y in zip(rows['base'][ds],rr):
                    assert all(x[k]==y[k] for k in ['prompt_index','prompt_sha256','generation_seed','ground_truth'])
            scores[name][ds]=summarize_rows(rr);rows[name][ds]=rr
    comparisons={}
    for base,new in [('base',n) for n in plan['arms']]+[('A_control_ref','B_vanilla'),('A_control_ref','C_ren'),('B_vanilla','C_ren')]:
        by={};draws=[]
        for ds in bench:
            r=paired_comparison(rows[base][ds],rows[new][ds]);n=r['paired_examples'];up=r['rescues'];down=r['degradations'];z=rng.multinomial(n,[up/n,down/n,1-(up+down)/n],size=10000);boot=(z[:,0]-z[:,1])/n
            r['ci95']=np.quantile(boot,[.025,.975]).tolist();by[ds]=r
            if ds in external:draws.append(boot)
        comparisons[new+'_vs_'+base]=dict(benchmarks=by,external_macro_delta=float(np.mean([by[b]['accuracy_delta'] for b in external])),external_macro_ci95=np.quantile(np.mean(draws,axis=0),[.025,.975]).tolist())
    training={}
    old=Path(plan['source_semantic_audit']).parent
    old_ids=[]
    for i in range(4):
        old_ids.append(sorted(json.loads(x)['source_id'] for p in (old/f'train/rollouts/round_{i:04d}').glob('shard-*.jsonl') for x in p.read_text().splitlines()))
    for name in plan['arms']:
        dest=root/'arms'/name/'train';ms=[json.loads(p.read_text())[0] for p in sorted((dest/'metrics').glob('round_*.json'))];assert len(ms)==4 and not any(m['skipped_update'] for m in ms)
        assert [m['reasoning_ablation'] for m in ms]==[plan['arms'][name]]*4
        for i in range(4):
            rr=[json.loads(x) for p in (dest/f'rollouts/round_{i:04d}').glob('shard-*.jsonl') for x in p.read_text().splitlines()]
            assert len(rr)==64 and all(r['snapshot_round']==i and r['response_tokens']<=8192 for r in rr)
            assert sorted(r['source_id'] for r in rr)==old_ids[i]
        training[name]=dict(rounds=ms,total_generated_tokens=sum(m['response_tokens'] for m in ms),total_round_seconds=sum(m['round_seconds'] for m in ms),clipped_updates=sum(m['grad_norm']>1 for m in ms),mean_hit_cap=float(np.mean([m['stability']['hit_cap_fraction'] for m in ms])))
    audits={prec:json.loads((root/f'audit_{prec}_signed/results.json').read_text()) for prec in ['bf16','fp32']};cos=json.loads((root/'gradient_audit/summary.json').read_text())
    result=dict(scores=scores,external_macro={n:float(np.mean([scores[n][b]['accuracy'] for b in external])) for n in scores},paired=comparisons,training=training,audits=audits,historical_checkpoint_cosines=cos)
    write_json(root/'comparison.json',result)
    lines=['# 当前 bounded absolute ReN：specificity、utility 与三组短训练','',
        '已完成缓存审计、固定批次全参数梯度夹角、三组独立Base起步的8k×4轮训练，以及固定32k sampled评估。没有训练新的outcome-specific gate，没有调整LR/weight normalization/控制项系数。','',
        '## 1. 当前g的gold/wrong specificity','',
        '48条既定rollout、10433个reasoning位置；正确/错误答案格式和长度匹配。每条rollout内按抽样概率回权，再对prompt等权。区间为4000次按snapshot分层的rollout-cluster bootstrap，条件于每条rollout的一个wrong donor。BF16输出头是训练对应精度；FP32是数值对照，均复用同一批C/H/W/T hidden states。','',
        '| Head | E[gG] | E[gW] | E[wG] | E[wW] | w差值95%CI | corr(wG,wW) | 高权重重合 |','|---|---:|---:|---:|---:|---|---:|---:|']
    for prec,z in audits.items():
        sg=z['specificity']['g'];sw=z['specificity']['w'];ov=z['overlap'];lines.append(f'| {prec} | {sg["gold"]:.6g} | {sg["wrong"]:.6g} | {sw["gold"]:.6g} | {sw["wrong"]:.6g} | {sw["ci95"]} | {ov["weighted_correlation"]:.4f} | {ov["high_overlap_given_gold"]:.2%} |')
    lines+=['','高权重分组是缓存gold位置的prompt-balanced第90百分位，仅用于审计，不是训练阈值。完整active fraction、weight mass、overlap、逐rollout指标见audit_*_signed/results.json。','',
        '## 2. 复用固定Teacher干预的utility','',
        '复用216 states / 39 original rollouts的864条已完成续写，不重新生成。每个state的效应为η=0.2 mixture干预收益，已经乘回changed mass；不把条件residual准确率当成总体准确率。原states按lambda选择和匹配，改按w分组不保证平衡；主要关联采用原triplet固定效应+Teacher KL、Student entropy、position、TV、最大/实际token概率协变量，按原rollout聚类重采样。回归只能控制记录下的特征，不能证明因果效用或外推全部训练状态。','',
        '| Head | adjusted high−zero utility (pp) | 95%CI(pp) | 每1SD w的adjusted utility(pp) | 95%CI(pp) |','|---|---:|---|---:|---|']
    for prec,z in audits.items():
        u=z['utility'];coef=u['adjusted_high_minus_zero'];lines.append(f'| {prec} | {100*coef["coefficients"][0]:+.4f} | {[100*v for v in coef["ci95"][0]]} | {100*u["adjusted_effect_per_w_sd"]:+.4f} | {[100*v for v in u["adjusted_ci95_per_w_sd"]]} |')
    lines+=['','同时计算s=[DW−DH]+与w_spec=s/(1+s)，见各audit的specificity_score：均值、稀疏度及controlled utility关联。同时报告未截正的DW−DH及反向w_spec作为对照：即使完全对称的噪声，截正后均值也可能大于零，不能把E[s]>0当成正确性特异性的证据。一个wrong donor无法检验换donor稳定性；FP32/BF16对照只涉及数值稳健性，不能替代该检验。没有将w_spec接入训练。','',
        '## 3. 固定已有轨迹上的梯度夹角','',cos['scope'],'',
        '原训练第4/8/12次更新前的参数分别是checkpoint3/7/11，已未保留，不能冒充精确历史重放。这里固定16条Base已有轨迹，比较保存的更新后checkpoint0/4/8/12，重新计算各自C/H与ReN权重；全参数、完整audit batch，范数在clip/Adam之前，reference含原0.1系数。','',
        '| Checkpoint after updates | R−C cosine | R−ref cosine | C−ref cosine |','|---|---:|---:|---:|']
    for x in cos['results']:
        cc=x['component_gradient_cosines'];lines.append(f'| {x["completed_updates"]} | '+' | '.join('undefined (zero norm)' if cc[k] is None else f'{cc[k]:+.5f}' for k in ['reasoning__control','reasoning__reference','control__reference'])+' |')
    lines+=['','cos<0表示在这个batch上梯度方向相反，不直接证明一方在阻碍有益泛化；cos>0也不保证答案正确。','',
        '## 4. 三组8k×4轮结果','',
        '训练固定四个Student worker；eval分片数由各次实际eval_plan读取，可使用不同数量的空闲GPU，题目和request seed不变。同Base、同前4轮64题prompt schedule、同optimizer/LR=1e−6、同control/reference系数与归一化。每轮重新on-policy rollout；权重分别0/1/g/(1+g)。每组256个prompt exposure。Vanilla沿用相同reasoning-token→rollout→prompt平均，不使用旧的无reference backend。各组的trajectory是自己策略采样，后续state分布可以不同。','',
        '| Dataset | N | Base | A control/ref | B Vanilla | C bounded ReN |','|---|---:|---:|---:|---:|---:|']
    for b in bench:lines.append(f'| {b} | {scores["base"][b]["examples"]} | '+' | '.join(f'{scores[n][b]["accuracy"]*100:.2f}%' for n in scores)+' |')
    lines+=['| External five-set macro | — | '+' | '.join(f'{result["external_macro"][n]*100:.2f}%' for n in scores)+' |','',
        'DAPO dev128在本次训练和结果之前按固定SHA顺序抽取，与2048训练池不相交；primary为这个预先冻结的sampled dev，外部五集继续作为迁移诊断。三组只评round4，没有按测试分数挑checkpoint。Base外部回答复用现有32k前缀缓存，仅dev128新生成。','',
        '| Contrast | External macro delta(pp) | 95% paired question CI(pp) |','|---|---:|---|']
    for name,x in comparisons.items():lines.append(f'| {name} | {x["external_macro_delta"]*100:+.3f} | {[v*100 for v in x["external_macro_ci95"]]} |')
    lines+=['','这里比较完整training system；将w设为1同时改变梯度幅度和位置分配，Adam/clipping也可能改变有效步长，不能单独识别gate排序质量。不得凭单seed、4步、小dev的点估计断言所有OPD或ReN无效。','',
        '## 5. 本轮训练的梯度与稳定性','',
        '| Arm | Round | Lreason | Lcontrol | reference penalty | preclip norm | R−C cosine | R−ref cosine | C−ref cosine |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name,info in training.items():
        for m in info['rounds']:
            cc=m.get('component_gradient_cosines',{})
            values=['—' if cc.get(k) is None else f'{cc[k]:+.4f}' for k in ['reasoning__control','reasoning__reference','control__reference']]
            lines.append(f'| {name} | {m["completed_updates"]} | {m["ren_loss"]:.5g} | {m["answer_stop_loss"]:.5g} | {m["reference_penalty"]:.5g} | {m["grad_norm"]:.4f} | '+' | '.join(values)+' |')
    lines+=['','A的reasoning梯度为零，所以涉及R的cosine不定义，不能记作“正交”。三组保留全部失败/capped rollout；rho诊断在A/B仍计算counterfactual bounded ReN，但实际应用权重为0/1；以reasoning_ablation、applied_reasoning_weight_sum及真实loss/gradient解释更新。','',
        '具体配对rescues/degradations、区间、各组cap/token成本和完整loss分量见comparison.json。此报告的scope与命令、输入/源码SHA、dev题号在experiment_plan.json固定。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'complete','external_macro':result['external_macro'],'dev_accuracy':{n:scores[n]['dapo_dev128']['accuracy'] for n in scores},'report':str(root/'REPORT.md')},indent=2))
if __name__=='__main__':main()
