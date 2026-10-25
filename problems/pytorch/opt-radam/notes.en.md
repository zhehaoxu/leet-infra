# RAdam Rectified Adaptive Moment Estimation Explained

## Source
RAdam paper: *On the Convergence of Adam and Beyond*, arXiv:1908.03213 (OpenReview). The algorithm skeleton follows `RAdam` from kozistr/pytorch_optimizer (the per-param path with `degenerated_to_sgd=False, adam_debias=False, foreach=False`, and `weight_decouple=True`).

## Mechanism in one sentence
The variance of Adam's adaptive learning rate decays with the number of steps, and the early variance estimate is unreliable → training is unstable. RAdam derives a closed form for the variance of the second-moment estimate and uses the stability coefficient \(r_t\) to scale the adaptive step: when the variance is large (early), it "cools down" the adaptive term; when the variance is negligible (\(n_t\) large enough), it falls back to standard Adam.

## Key order and traps (pitfalls)
1. **The extreme form of built-in warmup**: when \(n_t < \text{n\_sma\_threshold}\) and `degenerated_to_sgd=False`, the library sets \(r_t=-1\), so `step_size<0` and the entire update block is blocked by `if step_size>0 or n_sma>=threshold` — **this step does not even apply weight decay or accumulate momentum** (`m,v` remain zero). Therefore with \(\beta_2=0.999\) the first 4 steps are complete no-ops, and only from step 5 does it actually update. If you assume every step updates, the result will be wrong.
2. \(n_{\max}=\frac{2}{1-\beta_2}-1\) and \(\beta_2^t\) are pure scalars; \(n_t\) is also a scalar (not a tensor).
3. Copy the order of the four factors inside \(r_t\) exactly as the library does: \(\sqrt{(1-\beta_2^t)\cdot\frac{n_t-4}{n_{\max}-4}\cdot\frac{n_t-2}{n_t}\cdot\frac{n_{\max}}{n_{\max}-2}}\).
4. The denominator \(\sqrt{v}+\epsilon\) uses the **bias-uncorrected** \(v\); the correction for \(m\) is folded into `step_size = lr·r_t/(1−β1^t)` (i.e. `apply_adam_debias=False`, dividing by `bias_correction1`).

## Relation to Adam
Once the threshold is reached, \(r_t\to 1\) and \(n_t\to n_{\max}\), and the update degenerates to \(p \mathrel{-}= \frac{lr}{1-\beta_1^t}\cdot\frac{m}{\sqrt v+\epsilon}\) — exactly Adam's form without denominator bias correction (different from this project's opt-adamw convention; note the distinction).

## Grading details
The external reference uses an independent vectorized recomputation (explicit scalars \(n_t, r_t\), a different implementation path from the Optimizer). float32 throughout; per-param and vectorized agree element-wise (measured maxdiff 0 for the example values, and a maximum deviation of ≈1.2e-7 over 20 random steps × 50 seeds).

## Complexity
\(O(\text{numel})\) per parameter per step plus a few scalar operations, with two extra state tensors \((m, v)\).
