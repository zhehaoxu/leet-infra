# In Depth: Generalized Advantage Estimation (GAE)

## Source
Schulman, Moritz, Levine, Jordan, Abbeel, *High-Dimensional Continuous Control Using Generalized Advantage Estimation* (arXiv:1506.02438). GAE is defined as \(\hat A_t=\sum_{l=0}^{\infty}(\gamma\lambda)^l\delta_{t+l}^V\), implemented with the backward recursion \(\hat A_t=\delta_t+\gamma\lambda\hat A_{t+1}\) in a single sweep; it is the standard way to compute advantages in PPO training.

## Core points and pitfalls
1. **Backward recursion**: it must be computed from \(t=T-1\) backward, `last = delta + gamma*lam*last`; a forward accumulation is wrong.
2. **The TD error uses V_{t+1}**: \(\delta_t=r_t+\gamma V_{t+1}-V_t\), with \(V_T=0\) at the end (do not bootstrap the value after termination). Guard the out-of-range index with `t+1<T`.
3. **return = advantage + value**: the value regression target is \(R_t=\hat A_t+V_t\), not \(\hat A_t\) itself.
4. **The two ends of λ**: λ=0 → one-step TD (low variance, high bias); λ=1 (γ=1) → Monte Carlo (unbiased, high variance). The middle is a trade-off.

## Cross-check convention
The reference solution recurses backward for advantages, with returns=advantages+values. The judging `_oracle` first materializes the whole `delta` list, then accumulates in reverse with `acc=delta[t]+gamma*lam*acc` (a different structure from the reference solution); the two tensors are compared separately with `allclose(atol=1e-5,rtol=1e-5)`. Extra cases: λ=0 degenerating to the one-step TD, the identity returns-advantages==values, and the T=1 boundary.

## Complexity
\(O(T)\), a single backward sweep.
