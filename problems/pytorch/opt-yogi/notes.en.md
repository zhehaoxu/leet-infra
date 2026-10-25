# Yogi Sign-controlled Second Moment Explained

## Source
Zaheer, Reddi, Sachan, Rawhi, Kumar, *Adaptive Learning Rate Strategies for Non-convex and Stochastic Optimization* (arXiv:1807.01874). The reference implementation follows `Yogi` from kozistr/pytorch-optimizer (`weight_decouple=True`, `adam_debias=False`, `maximize=False`).

## Why change the second-moment update
Adam's \(v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2\) lets a single very large \(g^2\) make \(v\) jump in one step (keeping that coordinate's learning rate small for a long time). Yogi observes that if the update is written as "move \(v\) **up or down** along the direction of \(\operatorname{sign}(v-g^2)\), with magnitude \(g^2\) and step \((1-\beta_2)\)":

\[ v_t = v_{t-1} - (1-\beta_2)\,\operatorname{sign}(v_{t-1}-g_t^2)\, g_t^2 \]

then \(v\) changes by at most \((1-\beta_2)g_t^2\) each time, and the direction is determined by which of \(v\) and \(g^2\) is larger — "stickier" than Adam's soft average, and resistant to bursty gradients. It can be shown to preserve the upper-bound property of \(v\) with respect to \(g^2\) (\(v\) is always an upper-bound estimate of the historical \(g^2\)).

## Three pitfalls in matching the reference solution
1. **Nonzero initial state**: \(m_0=v_0=\texttt{initial\_accumulator}\) (default \(10^{-6}\)), not 0. Writing 0 deviates from the first few steps' \(v\).
2. **The sign uses \(v\) before the update**: \(\operatorname{sign}(v_{t-1}-g^2)\) must be computed before \(v\) itself changes — i.e. first compute `v - g2`, then `sign_`, then `addcmul_` into `v`.
3. **Order of denominator operators**: \(\sqrt{v}/\sqrt{bc_2}+\epsilon\), consistent with PyTorch (`v.sqrt().div_(bc2sq).add_(eps)`, where `bc2sq=√(1−β2^t)`). \(bc_1\) only affects the step size \(lr/bc_1\).

## Grading details
The grader uses an independent vectorized recomputation (`_oracle_yogi`): using formulations different from the solution (`b1*m+(1-b1)*g`, `v-(1-b2)*sign(v-g2)*g2`, `sqrt(v)/sqrt(bc2)+eps`, etc.) compared under multiple shapes and multiple steps of random gradients with `allclose(atol=1e-5, rtol=1e-5)`. Parameters with `grad is None` skip even the weight decay — dedicated cases cover this.

## Complexity
Same as Adam: \(O(\text{numel})\) per parameter per step, with two state tensors \(m,v\).
