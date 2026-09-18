# Outcome specificity and downstream utility

This workflow evaluates probability-alignment weights without modifying any model or training objective. It reuses saved C/H/T hidden states from `certified_correction_audit_20260916`; that historical directory name does not imply certified semantic correctness.

1. `scripts/probe_outcome_specificity.py` prepares one format/length-matched wrong-answer prompt per audited rollout, then scores the original fixed positions. Only the wrong-context Student backbone forward is new. C/H/T states and the Teacher output head are reused. All probe head matmuls and softmax operations use FP32, with TF32 disabled.
2. `scripts/analyze_outcome_specificity.py` computes prompt-balanced paired gold-minus-wrong statistics and rollout-cluster bootstrap confidence intervals.
3. `scripts/prepare_utility_states.py` freezes high/medium/zero matched states without consulting wrong-answer scores or continuation outcomes. `--tokens --mode residual` freezes the paired first-token interventions.
4. `scripts/run_utility_continuations.py` runs eight independent vLLM replicas, with continuous batching, prefix caching and chunked prefill. Restarting the manager adopts its own surviving workers and skips completed shards. Compilation caches are isolated per task. It does not kill unrelated processes.
5. `scripts/analyze_semantic_utility.py` requires complete, unique job coverage and no unresolved verifier errors. It estimates original fixed-eta mixture success differences and rollout-cluster intervals.
6. `scripts/audit_utility_parser.py` diagnoses discrepancies between the two predeclared scorers without changing frozen outcomes.
7. `scripts/report_semantic_utility.py` writes the standalone report, plot and source/result hashes.

All scripts require `--output-dir`. Initial probe preparation additionally requires `--source-audit`; actual probe execution uses `--run`. Input preparation refuses to overwrite frozen assignments. Use the experiment's existing `trl` environment and local Hugging Face cache in offline mode.

## Estimand and efficient sampling

For every selected state, fix `eta=0.2`. Compare a first token drawn from `pC` against `(1-eta)*pC+eta*qT`; afterward use the same ordinary Student sampling kernel. First-token distributions are full-vocabulary distributions at temperature one. Continuations use temperature 0.6, top_p 0.95, top_k 20, with paired random seeds.

The default residual estimator removes the exactly cancelling common probability mass:

```
m = eta * TV(pC, qT)
rC = positive_part(pC - qT) / TV(pC, qT)
rI = positive_part(qT - pC) / TV(pC, qT)
A = m * (E[success | rI] - E[success | rC])
```

This is an identity, not an approximation to the treatment effect. Two paired residual repetitions per state give 864 continuations for 216 states. Report the effect after multiplying by `m`. The conditional residual-arm accuracies are not overall Student or mixture accuracies.

The optional `--mode direct` uses 32 maximal-coupled pairs per state, cancelling identical first-token pairs analytically. It has correct original marginals but generally much lower power at the same generation budget. Do not mix these protocols after inspecting results.

## Length, scoring and statistical limits

The total response horizon is 32768 including the original generated prefix and forced first token. The remaining budget is also bounded by `40960 - original_prompt_length - prefix_length - 128`. Inputs are never truncated. Gold and wrong answer fields are scoring metadata only and never enter continuation prompts.

The primary verifier is the existing full-response math parser. The predeclared secondary metric parses only text after the final `</think>`; missing closure counts as failure. Verifier errors are retained and block final analysis until resolved.

Bootstrap units are original rollouts, stratified by snapshot; matched states, arms and repetitions remain together. Two repetitions per state do not establish a state's true utility sign, so the report does not treat noisy per-state signs as ground-truth classification labels. Positive local continuation effects would still not prove parameter-level learnability or benchmark improvement. Training is a separate decision gated on the two scientific premises.
