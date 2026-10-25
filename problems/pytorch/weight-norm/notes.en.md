# WeightNorm Reparameterization In Depth

## Core idea

Split the weight tensor $w$ into a **direction** and a **length**, parameterized independently:

$$ w = g\cdot\frac{v}{\|v\|_{\text{dim}}} $$

$g$ (weight_g) is the scalar / per-output-channel scale, and $v$ (weight_v) determines the direction. The
original parameter $w$ is no longer optimized directly; instead $(g,v)$ are optimized.

## Why

- Decoupling the weight's "magnitude" from its "direction" improves the condition number, so SGD converges
  faster and tolerates a larger learning rate — this is what OpenAI GPT-1 used, and what the
  `torch.nn.utils.weight_norm` commonly wrapped around today's diffusion models provides.
- During training the effective weight $w$ is recomputed from $(g,v)$ at every step; the forward pass is
  identical to an ordinary linear/convolutional layer.

## Implementation and pitfalls

- The norm is **L2**, taken along the given `dim` (convolutions often use `dim=1` or per-output-channel; this
  problem defaults to `dim=0`), and `keepdim=True` broadcasts it back to the original shape.
- **Division-by-zero protection**: when a slice of $v$ is all zeros, $\|v\|=0$, so `clamp_min(1e-12)` is
  needed; otherwise `0/0=NaN`. This is the hidden test point of the problem.
- $g$, $v$ and the output share the same shape; $g$ may be a scalar or a vector along dim, aligned via
  broadcasting.

## Decoupling from SpectralNorm / WeightNorm

`torch.nn.utils.weight_norm` is the official wrapper for this reparameterization (note that its `dim`
semantics differ from the older version); this problem only requires computing the effective weight by hand
from the formula, with no backward pass involved.
