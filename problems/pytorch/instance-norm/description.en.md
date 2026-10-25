# InstanceNorm

## Problem

Instance Normalization normalizes **each sample and each channel separately** over the spatial dimensions—the standard component in style transfer that erases
each image's own contrast information:

$$
y = \frac{x - \mu_{n,c}}{\sqrt{\sigma_{n,c}^2 + \epsilon}}\,\gamma_c + \beta_c,\qquad
\mu_{n,c},\sigma_{n,c}^2 = \text{mean/var over spatial dims } (2,3,\dots)
$$

Key difference from BatchNorm: the statistics are **neither across samples nor across channels**; difference from LayerNorm: the reduction is over the spatial
dimensions rather than the feature dimension.

## Rules

- Implement `solve(x, weight=None, bias=None, eps=1e-5)`, where x has shape `(N, C, *)` (e.g. `(N,C,H,W)` or `(N,C,L)`).
- The statistic dimensions are all spatial dims `dim >= 2`; broadcast after `keepdim=True`.
- `weight`/`bias` may be `None` (equivalent to an identity affine transform); when not `None` their shape is `(C,)`, broadcast per channel.
- The variance uses the **biased** convention; the result matches `F.instance_norm(x, None, None, w, b, use_input_stats=True)`.

## Examples

```
Input: x = [[[1,2],[3,4]], [[5,6],[7,8]]]   # (N=2, C=2, L=2), no affine parameters
sample 0 channel 0: [1,2] → μ=1.5, σ²=0.25
Output: [[[-0.99998, 0.99998], [-0.99998, 0.99998]],
       [[-0.99998, 0.99998], [-0.99998, 0.99998]]]   # each (sample, channel) normalized independently
```

## Constraints

- float32/float64; the number of spatial elements per (sample, channel) is ≥ 2
- A constant channel (σ²=0) must not produce NaN
