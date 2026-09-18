# Action-level shared positive correction

Implementation: `lulu/shared.py`, method `ren_shared`. Launch:

```bash
python scripts/run_shared_correction.py --run --detach
```

For frozen round Student C/H and answer-blind Teacher T:

- r = [min(T,H) - C]+; m = sum(r).
- donor = [C-T]+; M = sum(donor).
- q = C+r-(m/M)donor, with q=C when M=0.
- Lreason = mean_prompt mean_rollout mean_reasoning_tokens KL(stopgrad(q) || live Student).

No Top-K target approximation, scalar state gate, active-token normalization or
correction-mass normalization. All failed/capped rollouts remain. Control is the
existing Teacher KL global-token mean; reference is the existing reverse KL to
frozen Base, coefficient0.1. Full parameters, one update then fresh rollouts.

This pilot fixes 8k × 4 rounds,64 prompts/round drawn from the same2048 pool,
LR1e-6, Qwen thinking sampling0.6/0.95/20. Five Student workers, one H/reference,
two TP32B Teacher workers. Targets are reconstructed from CPU hidden caches in
128-position chunks; no dense length×vocabulary cache. H/T scoring overlaps
Student rollout work. The previously tested TP communication/recovery remains.

Every round logs m, M, m/M, target KL, normalization/box/TV errors and position
arrays. R/control/ref exact full-batch gradient norms and cosines at updates1/4.
Latest optimizer checkpoint every update, retention at2/4. Final4 is fixed for
external vLLM eval; intermediate2 gets disjoint dev128 only. Common32k eval,
context40960 minus prompt length and128 margin; prompts never truncated.
Each external dataset <=199 examples. Prior Base/A/B/C outputs are reused.

Probability bounds are coordinatewise, not a single scalar convex mixture.
Only recipient increases are hindsight-supported; donor decreases need not be.
TV(q,C)=m concerns the target, not the realized shared-parameter optimizer step.
Zero-mass states retain KL to round C and can anchor gradients from other states.
This is an experimental hypothesis; prior scalar-gate negatives neither prove
nor refute useful action-level correction.
