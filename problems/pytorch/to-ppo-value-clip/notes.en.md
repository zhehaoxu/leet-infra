# In Depth: PPO Clipped Value Loss

## Source
Schulman et al., *Proximal Policy Optimization* (arXiv:1707.06347) introduced the clipped objective; applying the same clipped MSE to the value function is the standard engineering implementation (the `vp clipped` variant in OpenAI Baselines / CleanRL PPO). Confining \(V_\theta\) to the old value ±ε and then taking the squared error against the target \(R\), followed by taking the max of the two errors, gives the conservative loss.

## Core idea and pitfalls
1. **max, not min**: unlike the policy clip, which uses `min` (a pessimistic lower bound), the value clip takes `max(unclipped, clipped)` — it would rather overestimate the loss than let an out-of-bounds update be rewarded. Getting this backwards turns it into an optimistic clip.
2. **Clip around V_old**: the interval is \([V_\text{old}-\epsilon, V_\text{old}+\epsilon]\), not around the returns.
3. **Reduces to MSE**: when the new values are all inside the interval, clipping changes nothing and the two terms are equal → plain MSE; this is a self-check that "clipping did not trigger".
4. The target is the GAE `returns` (R = A + V), not the advantage.

## Cross-check methodology
The reference solution vectorizes `maximum((vn-r)^2,(clip(vn,vo±eps)-r)^2).mean()`. The judging `_oracle` computes the clip element-wise with Python `min(max(...))` and then takes `max(...)` of the two squared errors before averaging (different structure, numerically equivalent), with `allclose(atol=1e-5,rtol=1e-5)`. There are also dedicated assertions for "large ε triggers no clipping == plain MSE" and "out-of-bounds samples are pulled back".

## Complexity
\(O(N)\) element-wise.
