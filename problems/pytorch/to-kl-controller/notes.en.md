# In Depth: PPO Adaptive KL Coefficient Controller

## Source
The adaptive KL control in the appendix of Ziegler et al., *Fine-Tuning Language Models from Human Preferences Using RL* (arXiv:1909.08593); HuggingFace TRL `AdaptiveKLController.update`: `proportional_error = clip(current/target - 1, -0.2, 0.2); mult = 1 + proportional_error * n_steps / horizon; value *= mult`. This problem focuses on this controller itself.

## Core idea and pitfalls
1. **It is the ratio `current/target - 1`, not the difference `current - target`**: the controller adjusts by relative deviation, independent of units; using the absolute difference directly would overshoot when the target KL is small.
2. **Clip the error to ±0.2 first, then multiply by \(n/H\)**: the order is clip before multiplication, to prevent a single-cycle explosion; multiplying first and clipping later gives a different result.
3. **\(n/H\) is the smoothing factor**: more steps and a smaller horizon make the adjustment more aggressive; as \(n/H\to0\) the coefficient barely changes.
4. **Direction**: KL over the target (\(d>d_\text{tgt}\)) raises \(\beta\) to add penalty, KL under the target lowers \(\beta\) — getting the sign backwards makes the system diverge.

## Cross-check methodology
The reference solution is `beta * (1 + clip(current_kl/target_kl - 1, -0.2, 0.2) * n_steps/horizon)`. The judging `_oracle` computes the clipping by hand with pure Python `max(-0.2, min(0.2, ...))` and does not reuse torch.clip (different structure, numerically equivalent), with `abs(got-exp) < 1e-5`. There are also assertions for "coefficient unchanged when \(d=d_\text{tgt}\)", "over/under-shooting push in opposite directions", and "large deviations saturate (a bigger KL gives the same result)".

## Complexity
\(O(1)\) scalar computation.
