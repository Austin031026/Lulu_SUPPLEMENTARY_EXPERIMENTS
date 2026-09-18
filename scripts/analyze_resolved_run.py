"""Summarize persisted diagnostics without any model inference."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--train-dir',required=True);a=p.parse_args()
    root=Path(a.train_dir);output=root/'analysis';output.mkdir(exist_ok=True)
    rows=[]
    for path in sorted((root/'diagnostics').glob('round_*/summary.json')):
        scores=json.loads(path.read_text());index=scores['round']
        file=root/'metrics'/f'round_{index:04d}.json'
        metrics=json.loads(file.read_text())[0] if file.exists() else {}
        rollouts=[]
        for shard in (root/'rollouts'/f'round_{index:04d}').glob('shard-*.jsonl'):
            rollouts.extend(json.loads(line) for line in shard.read_text().split('\n') if line.strip())
        weights=scores['concentration']['raw_weight']
        verified=[r for r in rollouts if isinstance(r.get('correct'),bool)]
        rows.append({'round':index+1,'completed_updates':metrics.get('completed_updates'),
            'causal_kl':scores['causal_kl_mean'],'hindsight_kl':scores['hindsight_kl_mean'],
            'mean_weight':scores['mean_weight'],'active_fraction':weights['positive_fraction'],
            'effective_positions':weights['effective_positions'],
            'weight_top1pct_mass':weights['top_fraction_mass']['0.01'],
            'weight_top10pct_mass':weights['top_fraction_mass']['0.1'],
            'objective_loss':metrics.get('objective_loss'),'grad_norm':metrics.get('grad_norm'),
            'rollout_accuracy':sum(r['correct'] for r in verified)/len(verified) if verified else None,
            'verifier_coverage':len(verified)/len(rollouts) if rollouts else None,
            'hit_cap_fraction':sum(r['truncated'] for r in rollouts)/len(rollouts) if rollouts else None,
            'mean_response_tokens':sum(r['response_tokens'] for r in rollouts)/len(rollouts) if rollouts else None,
            'round_seconds':metrics.get('round_seconds'),'checkpoint_seconds':metrics.get('checkpoint_seconds')})
    if not rows:return
    with (output/'training_summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for ax,keys,title in [(axes[0,0],['causal_kl','hindsight_kl'],'Full-vocabulary mismatch'),
                          (axes[0,1],['active_fraction','weight_top10pct_mass'],'Weight support and concentration'),
                          (axes[1,0],['rollout_accuracy','hit_cap_fraction'],'Rollout metadata (different prompts each round)'),
                          (axes[1,1],['grad_norm'],'Gradient norm before clipping')]:
        for key in keys:ax.plot([r['round'] for r in rows],[r[key] for r in rows],label=key)
        ax.set_title(title);ax.set_xlabel('Completed round');ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(output/'training_diagnostics.png',dpi=160);fig.savefig(output/'training_diagnostics.pdf');plt.close(fig)
    fig,ax=plt.subplots(figsize=(6,4),layout='constrained')
    path=root/'diagnostics/round_0000/concentration.csv'
    if path.exists():
        with path.open() as stream:curve=list(csv.DictReader(stream))
        for key in ('causal_kl_mass','old_alpha_mass','resolved_weight_mass'):
            ax.plot([float(r['top_fraction']) for r in curve],[float(r[key]) for r in curve],label=key)
        ax.plot([0,1],[0,1],'k--',alpha=.4);ax.set_xlabel('Top fraction of reasoning positions');ax.set_ylabel('Cumulative score mass')
        ax.set_title('Round 1: concentration before update');ax.legend(fontsize=8);ax.grid(alpha=.2)
        fig.savefig(output/'first_round_concentration.png',dpi=160);fig.savefig(output/'first_round_concentration.pdf')
    plt.close(fig)


if __name__=='__main__':main()
