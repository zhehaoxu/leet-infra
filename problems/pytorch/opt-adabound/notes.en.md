# In Depth: AdaBound Dynamic Learning-Rate Clamping

## Source
AdaBound paper: *AdaBound: Adaptive Bound Learning Rate Methods*, arXiv:1902.11010 (Microsoft, ICLR 2019). The algorithm skeleton follows `AdaBound` from kozistr/pytorch_optimizer (the per-param path with `ams_bound=False, foreach=False`, `weight_decouple=True`).

## Mechanism in one sentence
Adam's effective per-coordinate learning rate is \(\eta_t/(\sqrt{\hat v}+\epsilon)\). AdaBound clamps it inside the interval \([\alpha_t,\beta_t]\), whose endpoints converge onto the same `final_lr` from both sides as \(t\) grows: early on the interval is extremely wide (equal to Adam), later it collapses to a point (equal to fixed-learning-rate SGD).

## Implementation points (pitfalls)
1. **What gets clamped is "denominator / step size", not the step size itself**: the library first computes `update = de_nom / step_size` (i.e. \(\text{denom}/\text{step}\), the **reciprocal** of the effective learning rate), then `clamp_(lower, upper)`, then `*= m`, then `p -= update`. This is equivalent to restricting the per-coordinate learning rate to lie within the bounds. Do not apply the clamp to \(m\) or to the final displacement.
2. **Dynamic bounds**: `lower = flr*(1 − 1/(γt+1))` (the lower bound rises from 0 toward flr), `upper = flr*(1 + 1/(γt))` (the upper bound falls from ∞ toward flr). `flr = final_lr * lr / base_lr`; `base_lr` is the initial lr captured at construction (here lr is constant, so flr = final_lr).
3. **Numerical-stability constant 1e-15**: the library adds `exp_avg_sq_eps=1e-15` to \(v\) before the square root (i.e. \(\sqrt{v+1e^{-15}}\)), and `eps=1e-8` is added afterwards. Do not drop either of the two.
4. **The step size is folded into the denominator**: `step_size = lr·sqrt(1−β2^t)/(1−β1^t)` merges both bias corrections with lr; because the value is divided by `denom` before clamping, the denominator uses the **uncorrected** \(\sqrt{v+1e^{-15}}\).
5. Decoupled weight decay happens **before the moment updates** and is applied to every parameter that has a gradient; parameters with `grad is None` are skipped entirely.

## Cross-check convention
The external reference uses **the same sequence of tensor primitives as the reference solution** (`lerp_ / mul_+addcmul_ / add(1e-15).sqrt_().add_(eps) / div_(step) / clamp_ / mul_(m) / sub_`) recomputed independently in vectorized form — keeping one and the same implementation path so that the rounding of fused operators, once amplified by `/step_size`, cannot shift the result (measured: 60 seeds × 25 steps, element-wise maxdiff=0). This is the only problem in the collection that deliberately aligns primitives, because AdaBound's clamping amplifies 1e-8-level rounding to 1e-6-level, and an independent algebraic recomputation would push right up against the judge tolerance.

## Complexity
\(O(\text{numel})\) per parameter per step, with two extra states \((m, v)\), the same order as Adam.
