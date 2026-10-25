# Optimizer: Tiger Sign-of-Momentum Update

## Problem

Adaptive optimizers have to maintain a second moment, which doubles memory usage. Sign-based methods (such as Lion and SignSGD) take a different route: **use the sign of the update direction directly** to normalize each step's magnitude to \(1\), relying only on momentum to smooth the direction. Tiger is one of the most minimal members of this family — it keeps only **a single momentum buffer**, uses the sign of the momentum as the update direction, and pairs it with decoupled weight decay. It often appears as the "companion optimizer" for non-2D parameters (biases, normalization layers) in the Muon family.

This problem asks you to **subclass `torch.optim.Optimizer`** and implement Tiger, with `step()` updating each parameter in place. The grader feeds a fixed gradient sequence and compares the final parameters.

## Rules

- Factory function `solve(params, lr=1e-3, beta=0.965, weight_decay=0.01)`, returning an instance of a `torch.optim.Optimizer` subclass whose `step()` updates `params` in place.
- Each parameter maintains only one momentum \(m\), initialized to \(0\). At step \(t\), with gradient \(g\):
  1. Decoupled weight decay (wd≠0, done first): \(p \leftarrow p\,(1-lr\cdot\text{wd})\)
  2. Momentum: \(m \leftarrow \beta\, m + (1-\beta)\, g\)
  3. Sign update: \(p \leftarrow p - lr\cdot \operatorname{sign}(m)\)
- **No second moment, no bias correction**: the update magnitude is always \(lr\) (because the sign is taken), and the momentum starts at 0. Parameters with `grad is None` are skipped entirely. float32 throughout, ready-made optimizers forbidden.

## Examples

A single parameter \(p=[1.0,\,-2.0,\,0.5]\), `lr=0.01`, `beta=0.965`, `weight_decay=0.01`, fed three steps of gradients:

```
g₁ = [ 0.10, -0.20,  0.30]
g₂ = [-0.40,  0.05,  0.20]
g₃ = [ 0.15, -0.10, -0.25]
after three steps p ≈ [1.009701, -1.969403, 0.469853]
```

## Constraints

- dtype float32, in-place update; supports a mix of 1D and multi-dimensional parameters
- Must match the reference implementation step by step (grading details in notes.md)
