# Walkthrough: Lion Sign-Based Momentum Update

## Source
The Lion paper: *Lion: Symbolic Discovery of Optimization Algorithms*, arXiv:2302.06675 (Google, 2023). The algorithm skeleton follows the `Lion` implementation in kozistr/pytorch_optimizer (the eager per-param path, with `weight_decouple=True`).

## The mechanism in one sentence
Update direction = \(\operatorname{sign}(\beta_1 m + (1-\beta_1) g)\), i.e. "take the sign of the interpolation between the old momentum and the current gradient"; afterwards the momentum keeps EMA-ing with a different decay rate \(\beta_2\). Because of the sign, each coordinate moves strictly by \(\pm lr\) in this step (the sign decides the direction), in sharp contrast to Adam's per-coordinate scaling with \(\hat m/(\sqrt{\hat v}+\epsilon)\).

## Division of labor between the two betas (pitfall)
- \(\beta_1\) (default 0.9) is used only to **compose this step's direction** \(u\); it is never written back into the state.
- \(\beta_2\) (default 0.99) is the decay rate that actually updates the momentum buffer \(m\).
- Critical order: first use the **old** \(m\) to compute \(u=\operatorname{sign}(\beta_1 m + (1-\beta_1)g)\), then execute \(m \leftarrow \beta_2 m + (1-\beta_2)g\). If you update \(m\) first and only then compute the direction, the result is wrong. In the reference implementation this is `update = m.lerp(g, 1-beta1).sign_()` followed by `m.lerp_(g, 1-beta2)`.

`lerp(a, b, w) = a + (b-a)w = (1-w)a + wb`, hence `m.lerp(g, 1-beta1) = beta1*m + (1-beta1)*g`.

## Decoupled weight decay
Same form as AdamW: `p ← p·(1 − lr·wd)`, applied before the direction is computed. This problem has no `fixed_decay` (i.e. the decay is scaled by lr), matching the paper's default.

## Comparison setup
The external reference is an equivalent vectorized recomputation that explicitly applies the sign and the two EMAs each step. `sign(0)=0`, so a coordinate whose interpolation is exactly 0 does not move. Parameters with `grad is None` are skipped (weight decay is not applied either). Everything is float32; the two-stage formulation and the vectorized recomputation agree element by element (the example values measure maxdiff 0).

## Complexity
Per parameter per step \(O(\text{numel})\), with a single extra state tensor \(m\) — about half of Adam's memory.
