# In Depth: AdaBelief Belief-Based Second Moment

## Source
AdaBelief paper: *AdaBelief Optimizer: Adapting Stepsizes by the Belief in Observed Gradients*, arXiv:1904.03502 (NeurIPS 2018/2019). The algorithm skeleton follows `AdaBelief` from kozistr/pytorch_optimizer (the per-param path with `rectify=False, ams_bound=False, foreach=False`, `weight_decouple=True, fixed_decay=False`).

## Mechanism in one sentence
Replace Adam's second moment \(\mathbb{E}[g^2]\) with the second moment of the "gradient prediction error" \(\mathbb{E}[(g-m)^2]\) — the belief. If the gradient is predictable the error is small → small denominator → effective step size amplified; if the gradient is chaotic the error is large → large denominator → step size shrinks.

## Key ordering (pitfalls)
1. **Update \(m\) first, then compute the residual with the updated \(m\)**: `diff = g - m` (\(m\) has already been lerped). Using the old \(m\) degenerates into a different algorithm.
2. Add \(\epsilon\) once inside the accumulation of \(s\): `s = β2·s + (1−β2)·diff² + eps`; then add `+ eps` again in the final denominator. Both places, matching the library implementation.
3. Bias correction applies only to the \(\sqrt{s}\) of the denominator (divide by \(\sqrt{1-β_2^t}\)); the numerator \(m\) is not corrected separately — instead \(1/(1-β_1^t)\) is folded into `step_size = lr/(1-β1^t)`. Equivalent form: \(p \mathrel{-}= \text{step\_size}\cdot m/(\sqrt{s}/\sqrt{\hat β_2}+\epsilon)\).

## The non-rectified branch
With `rectify=False` there is no RAdam-style SMA rectification; the adaptive update is applied directly, so there is an update from step 1 (unlike RAdam's no-op first steps). This problem fixes `rectify=False`.

## Cross-check convention
The external reference recomputes independently in vectorized form (each step explicitly doing lerp, residual square, and both bias corrections). All float32; per-param and vectorized agree element-wise (measured maxdiff 0 on the example values; over random multi-step runs of 20 steps × 50 seeds the max deviation is ≈2.4e-7, far below the judge tolerance).

## Complexity
\(O(\text{numel})\) per parameter per step, with two extra states \((m, s)\), the same order as Adam.
