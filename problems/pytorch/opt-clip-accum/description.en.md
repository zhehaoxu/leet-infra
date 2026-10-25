# Optimizer: Gradient Accumulation + Global Norm Clipping + Momentum SGD

## Problem

When memory is short we use **gradient accumulation**: run several consecutive micro-batches, sum their gradients, and then treat the sum as one large batch for a single update. Large-model training almost always enables **gradient clipping** as well: whenever the global gradient norm over all parameters concatenated exceeds a threshold, everything is shrunk by the same factor so exploding gradients cannot derail training. Stack the classic **momentum SGD** on top of the two and you get the combined mechanism this problem asks you to implement.

This problem asks you to **subclass `torch.optim.Optimizer`** and wrap "accumulate → average → global norm clipping → momentum update" into a single `step()`. The judge feeds gradients micro-batch by micro-batch and compares the final parameters.

## Rules

- Factory function `solve(params, lr, momentum=0.0, accum_steps=1, max_norm=1.0)`, returning an instance of a `torch.optim.Optimizer` subclass whose `step()` updates `params` in place.
- Each parameter keeps two state tensors: the accumulator \(a=0\) and the momentum buffer \(b=0\); the scheduler level keeps a micro-step counter \(m\) (initially \(0\)).
- Every `step()` (one micro-batch):
  1. For each parameter \(p\) **that has a gradient** (gradient \(g\)): \(a \leftarrow a + g\).
  2. \(m \leftarrow m+1\). If \(m <\) `accum_steps`, **do not update the parameters this time** and return immediately (keep accumulating).
  3. Otherwise (a full large batch is ready):
     - Effective gradient (mean) \(\bar g_p = a_p / \texttt{accum\_steps}\).
     - **Global norm** (over all participating parameters): \(\|\bar g\|=\sqrt{\sum_p \|\bar g_p\|_2^2}\).
     - Clipping coefficient \(c=\min\!\left(\dfrac{\texttt{max\_norm}}{\|\bar g\|+10^{-6}},\ 1\right)\) (same convention as `torch.nn.utils.clip_grad_norm_`: shrink only when the threshold is exceeded, otherwise leave untouched).
     - Clipped gradient \(\hat g_p = \bar g_p \cdot c\).
     - Momentum: \(b_p \leftarrow \text{momentum}\cdot b_p + \hat g_p\); update \(p \leftarrow p - lr\cdot b_p\).
     - Reset: set \(a_p\leftarrow 0\) for every participating parameter; \(m\leftarrow 0\).
- Parameters with `grad is None` are neither accumulated, nor included in the global norm, nor updated. Everything is float32; calling a ready-made optimizer or `clip_grad_norm_` is forbidden.

## Example

Two parameters \(p_1\in\mathbb R^{3\times2},\,p_2\in\mathbb R^{4}\) (initialized with `randn` under `torch.manual_seed(7)`), `lr=0.1`, `momentum=0.9`, `accum_steps=4`, `max_norm=1.0`. Feed 12 micro-batch gradients (each step, for every parameter, `randn_like * 2.0`), i.e. after 3 effective updates:

```
p1 ≈ [[-0.211246,  0.827588],
      [ 0.988395, -0.952307],
      [ 1.620651, -0.932862]]
p2 ≈ [-0.524546, 1.210246, 0.033766, -1.750957]
```

(The gradients are scaled up by \(2.0\) and the global norm far exceeds 1.0, so every update is clipped to norm ≈ 1 before accumulating momentum.)

## Constraints

- The order accumulate, average, clip, momentum **must** be as above: average first, then compute the global norm, then clip.
- Must agree parameter by parameter with the reference implementation (see notes.md for the comparison setup).
- Supports mixing 1D and multi-dimensional parameters; with `accum_steps=1` it degenerates to "clip + momentum update on every step".
