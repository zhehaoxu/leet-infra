# Alignment Training: PPO Clipped Value Loss

## Problem

Besides clipping the policy to a trust region, PPO commonly applies a "clipped MSE" to the value function update as well, to keep a single step from pulling \(V_\theta\) too far away from the old value \(V_\text{old}\) (symmetric to the policy clip, and it stabilizes value head training). Given the regression target \(R\) (the GAE return), the per-element loss takes the **larger** of the "unclipped" and "clipped" squared errors:

\[ \mathcal{L}^V = \frac1N\sum_i \max\!\Big((V_{\theta,i}-R_i)^2,\ \big(\operatorname{clip}(V_{\theta,i},\,V_{\text{old},i}-\epsilon,\,V_{\text{old},i}+\epsilon)-R_i\big)^2\Big) \]

The clip confines the new value to \([V_\text{old}-\epsilon,\,V_\text{old}+\epsilon]\); taking the max means that a direction which "goes out of bounds and has a larger error" is not let off the hook (conservative update).

## Rules

Inputs are `values_new` (\(V_\theta\)), `values_old` (\(V_\text{old}\)), `returns` (\(R\)), all of shape \((N,)\), plus the scalar `epsilon`. Return a 0-dim float32 scalar tensor = the mean clipped value loss. Do not use an off-the-shelf RL library.

## Example

`epsilon=0.2`, \(V_\theta=[0.6,1.2,-0.3,0.9]\), \(V_\text{old}=[0.5,1.0,-0.2,0.8]\), \(R=[0.7,1.5,-0.5,0.8]\):

```
clip(V_new, V_old±0.2) = [0.6, 1.2, -0.3, 0.9]   (nothing is out of bounds here, so clipped == new)
per-term max((V_new-R)^2,(clip-R)^2) = [0.01, 0.09, 0.04, 0.01]
loss = mean = 0.0375
```

## Constraints

- CPU float32; \(\epsilon>0\).
- Take `max` of the two squared errors (not `min`, and not the clipped one alone).
- When every \(V_\theta\) lies inside \([V_\text{old}\pm\epsilon]\), the clipped term equals the unclipped one and the loss reduces to plain MSE.
