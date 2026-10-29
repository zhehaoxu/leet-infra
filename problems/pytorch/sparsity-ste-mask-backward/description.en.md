# Sparsification: STE and Soft-Threshold Reparameterization Backward for Pruned Fine-Tuning

## Problem

When fine-tuning a pruned model, the forward pass uses the **masked** weights \(w\odot m\) — but what should the backward pass do? A binary mask \(m=\mathbb{1}[|w|\ge\tau]\) has zero derivative with respect to \(w\), so a literal backward pass would starve every pruned weight of gradient and make "revival" impossible. Production code solves this in one of two ways:

- **STE (straight-through estimator)**: hard mask in the forward pass, treat the mask as a constant in the backward pass, i.e. push \(g\odot m\) straight through. Kept positions train normally and pruned positions get exactly zero gradient. This is the standard pruning fine-tune recipe; the price is a deliberate forward/backward mismatch (the very bias discussed by Bengio et al. 2013).
- **WSN (soft threshold weight reparameterization)**: replace the mask by the continuous relaxation \(m=\mathrm{clamp}(|w|/\tau,0,1)\). The forward \(y=w\cdot m\) is then the quadratic \(w|w|/\tau\) for \(|w|<\tau\), and the backward carries both the \(m\) term and the mask's own derivative: for \(|w|<\tau\) we get \(\mathrm{d}y/\mathrm{d}w = 2|w|/\tau\) (times the upstream gradient), degenerating to 1 for \(|w|\ge\tau\). Kusupati et al. (ICML 2021) use this to turn "prune or not" into a learnable threshold problem.

The upstream gradient `grad_out` is handed to you, so you only have to produce the forward output, the mask and the weight gradient under each recipe — no autograd needed.

## Rules

Given `weight` (float32, any shape), scalar `tau > 0`, upstream gradient `grad_out` (float32, same shape) and `mode`:

- `mode="hard_ste"`:
  - mask \(m=\mathbb{1}[|w|\ge\tau]\) (the equality counts as kept);
  - forward \(y=w\odot m\);
  - gradient \(g_w=g\odot m\) (the mask is not differentiated).
- `mode="wsn"`:
  - mask \(m=\mathrm{clamp}(|w|/\tau,\,0,\,1)\);
  - forward \(y=w\odot m\);
  - gradient \(g_w=g\odot\big(m+\frac{|w|}{\tau}\cdot\mathbb{1}[|w|<\tau]\big)\).

Return `(y, grad_w, mask)`, all the input's shape and float32. `torch.autograd` is forbidden.

## Examples

```
weight   = [0.5, -1.5, 2.0, -0.25]
tau      = 1.0
grad_out = [1, 2, 3, 4]

hard_ste:
  mask  = [0, 1, 1, 0]           # |w| = [0.5, 1.5, 2.0, 0.25], keep those >= 1
  y     = [0, -1.5, 2, 0]
  grad  = [0, 2, 3, 0]

wsn:
  mask  = [0.5, 1.0, 1.0, 0.25]  # clamp(|w|/1.0, 0, 1)
  y     = [0.25, -1.5, 2, -0.0625]
  grad  = [1*(0.5+0.5), 2*1, 3*1, 4*(0.25+0.25)] = [1, 2, 3, 2]
```

## Constraints

- CPU float32; `tau > 0`; `grad_out` is given, autograd is off limits.
- `hard_ste` keeps \(|w|=\tau\) (mask 1) and passes the gradient through there.
- `wsn` saturates to mask 1 with gradient exactly equal to the upstream value for \(|w|\ge\tau\), and uses the coefficient \(2|w|/\tau\) for \(|w|<\tau\).
- Both modes must satisfy \(y=w\odot m\) in the forward pass.
