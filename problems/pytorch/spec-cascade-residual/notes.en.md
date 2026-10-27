# Solution Notes · Cascaded Residual

## What the residual means physically

`rem` is "the probability mass the target still wants to say but no draft has claimed yet". At level 1, `rem` is \(p\) itself; each level may only take its overlap with `rem`, that is \(\min(q_l, rem)\), and the take must be subtracted immediately. Whatever `rem` remains after all levels is covered by no draft and goes back to the target.

This rule settles "multiple drafts overlap" in one shot: no matter how similar the \(L\) drafts are, mass consumed by an earlier level is never counted again, and `Σ accepted_mass ≤ 1` always holds.

## Why a per-level min, not a per-level product

One might model stacked drafts as "independent verifications", each passing with some probability, and multiply. But the real semantics of speculative decoding is **filling a hole in series**—one token is emitted at a position, level 1 already explains 0.8 of the mass, and level 2 can only fight over the remaining 0.2. Hence take-minimum accumulation rather than a product of probabilities.

## The fallback decides whether there is bias

- **Resample from \(p\)** (this problem): `out = Σacc + |rem|·p`. The residual mass is spread in proportion to \(p\) while the hit mass is already fully counted, so tokens the drafts cover well are inflated (\(p(x)\) becomes \((1+|rem|)p(x)\)) and uncovered tokens only receive \(|rem|p(x)\)—the distribution tilts toward wherever the drafts are accurate.
- **Resample from the residual**: `out = Σacc + rem/|rem| · |rem| = Σacc + rem ≡ p`. That is strict rejection sampling, preserving the distribution exactly, at the cost of normalizing the residual before sampling.

This problem outputs the first variant so the bias becomes computable and observable: whenever `target_mass > 0`, `out` differs from \(p\).

## The effect of level order

Per token the hit mass equals \(\min(p(x), \sum_l q_l(x))\)—successive minimums equal a single truncation—so `out` does not depend on order. Yet the credit each level receives is decided entirely by order: with the same set of drafts, running the broad-coverage level first leaves later levels only the leftovers. Operationally that means level order should follow hit rate and cost, putting the cheap and accurate drafts first.

## Boundaries

- \(L=0\): `out = p`, `accepted_mass` is empty, `target_mass = 1`—a pure target sample.
- Some level with \(q_l = p\): that level consumes everything and all later levels contribute 0.
- Row normalization is a hard constraint: an unnormalized distribution destroys the "mass" semantics of the min and must be rejected rather than silently accepted.

## Related problems

- `inf-spec-decoding-rejection`: single-draft rejection sampling, the distribution-exact side.
- `spec-sequoia-sample-verify`: sampling verification over multiple branches at one position, with a dynamically shrinking residual.
- `spec-medusa-tree-verify`: typical acceptance on a tree, replacing rejection with a threshold rule.
