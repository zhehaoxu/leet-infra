# PRM Best-of-N Aggregation Ledger: Four Step-Score Reduction Rules

## Problem

A Process Reward Model (PRM, Lightman et al. 2023, arXiv:2305.20050 "Let's Verify Step by Step")
scores **every step** of a candidate solution instead of only the final answer. In a best-of-N
setting you must compare N candidates from these variable-length step-score sequences, which forces
a choice of **reduction rule** that collapses each sequence into a scalar — and whatever the rule
is, that is what the ranking trusts:

- `last` looks only at the final step: local, easily overturned by a single late step;
- `max` looks at the best step: optimistic, one highlight wins;
- `min` looks at the worst step: pessimistic, one weak step eliminates;
- `product` multiplies all steps: any step near zero crushes the whole chain — precisely why the
  paper prefers product: the cost of one hidden failure step in a long chain is exponentially
  amplified.

> To keep judging **deterministic**, step scores arrive directly as numeric lists; the tie rule is
> pinned as "aggregate descending, ties broken by the smaller index".

## Rules

- Implement `solve(step_scores, answer_scores=None, rule="product", top_m=1)`:
  - `step_scores: list[list[float]]`, per-candidate step-score sequences (variable lengths, at
    least 1 step each);
  - `answer_scores: None or list[float]`, per-candidate final-answer scores; when provided,
    `agg[i] = step aggregation[i] × answer_scores[i]`;
  - `rule`: one of `"last" | "max" | "min" | "product"`;
  - `top_m: int ≥ 1`.
- After aggregation, sort by `(-agg[i], i)`:
  - `best`: index of the largest aggregate (ties to the smaller index);
  - `top`: the first `top_m` indices (descending, ties to the smaller index);
  - `tie`: whether the maximum is shared (`agg.count(max) > 1`, strict counting).
- The ledger also returns `agg` (per-candidate aggregates), `n_steps` (per-candidate step counts),
  and `min_len` (shortest candidate's step count).
- Validation (`ValueError`): `rule` not among the four; `top_m` not a positive integer (bool
  included); empty `step_scores`; a candidate with no steps; `answer_scores` length mismatch.

## Example

```
Input: step_scores = [[0.9, 0.5, 0.8], [0.7, 0.9], [1.0, 0.1, 0.1, 0.1]], rule = "product"
  product: [0.36, 0.63, 0.001]
  best = 1 (0.63 > 0.36 > 0.001), top(3) = [1, 0, 2]
  note: under max the best would be 2 (a 1.0 highlight step); under last it would be 1
Output: agg = [0.36, 0.63, 0.001], best = 1, top = [1, 0, 2], tie = false
```

## Constraints

- Pure Python suffices (torch not needed); scores are floats in `[0, 1]`
- `top_m` may exceed N (then `top` is simply all candidates ranked); `bool` is a subclass of `int`
  and must be explicitly excluded
- All four rules must be implemented under their paper names; an unknown `rule` must raise before
  any aggregation happens
