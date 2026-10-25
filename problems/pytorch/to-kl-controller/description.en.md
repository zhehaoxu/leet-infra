# Alignment Training: PPO Adaptive KL Coefficient Controller

## Problem

In RLHF/PPO the KL penalty coefficient \(\beta\) cannot be a fixed value: too large locks the policy near the reference model so it cannot learn, too small lets the policy run away and language quality collapse. Ziegler et al. (arXiv:1909.08593) and HuggingFace TRL's `AdaptiveKLController` use a **proportional controller** to adjust \(\beta\) online:

Each control cycle compares the observed mean KL \(d\) against the target \(d_\text{tgt}\), computes the **proportional error**, and clips it to \([-0.2, 0.2]\):

\[ e = \mathrm{clip}\!\left(\frac{d}{d_\text{tgt}} - 1,\ -0.2,\ 0.2\right) \]

then smoothly rescales \(\beta\) by "how many steps \(n\) this cycle took and the controller horizon \(H\)":

\[ \beta' = \beta \cdot \left(1 + e\cdot\frac{n}{H}\right) \]

Intuition: if the actual KL is **above** the target (\(e>0\)), **raise** \(\beta\) to strengthen the penalty; if it is **below** the target (\(e<0\)), **lower** \(\beta\) to loosen the constraint. The \(\pm0.2\) clip prevents a single step from changing the coefficient too abruptly.

## Rules

Given the scalar `beta` (the current KL coefficient), `current_kl` (the measured mean KL \(d\) this cycle), `target_kl` (the target \(d_\text{tgt}\)), `horizon` (\(H>0\)), and `n_steps` (the number of steps this cycle \(n\)). Compute and return the new KL coefficient \(\beta'\) (a 0-dim float32 scalar) according to the formula above. Do not use an off-the-shelf RLHF library.

## Example

`beta=0.1`, `current_kl=0.3`, `target_kl=0.2`, `horizon=1000`, `n_steps=1`:

```
e = clip(0.3/0.2 - 1, -0.2, 0.2) = clip(0.5, -0.2, 0.2) = 0.2
beta' = 0.1 * (1 + 0.2 * 1/1000) = 0.1 * 1.0002 = 0.100020
```

Measured KL above the target → the coefficient is raised.

## Constraints

- CPU float32; `target_kl>0`, `horizon>0`.
- When \(d=d_\text{tgt}\), \(e=0\) and \(\beta\) is unchanged.
- When \(|d/d_\text{tgt}-1|\ge 0.2\) the error is clipped (saturated); a larger deviation no longer changes the result.
