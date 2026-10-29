# Diffusion Training: SD3 Timestep-Sampling Density

## Problem Statement

SD3 training does not draw timesteps uniformly. It first draws a continuous position \(\sigma\in[0,1]\) (then multiplies by `num_train_timesteps` to get an integer timestep). `compute_density_for_timestep_sampling` offers three schemes for that position:

- `logit_normal`: \(z\) is standard normal and the position is \(\sigma(\mu+\varsigma z)\). \(\mu\) shifts the bulk (negative values toward the noisy side) and \(\varsigma\) controls concentration.
- `mode`: \(u\) is uniform on \(U(0,1)\) and the position is \(1-u-s\left(\cos^2\frac{\pi u}{2}-1+u\right)\), where \(s\) is `mode_scale`. For \(s>0\) this squeezes a uniform density into a shape that is high at both ends and low in the middle; SD3 training scripts use \(s=1.29\).
- `uniform`: the position is \(u\) itself.

Given a scheme and **already-drawn samples**, compute the corresponding continuous positions. Randomness is supplied explicitly by the caller, so the problem itself is deterministic.

## Rules

Compute element-wise:

| scheme | input samples | output position |
| --- | --- | --- |
| `logit_normal` | \(z\sim N(\mu,\varsigma^2)\) | \(1/(1+e^{-(\mu+\varsigma z)})\) |
| `mode` | \(u\sim U(0,1)\) | \(1-u-s\left(\cos^2\frac{\pi u}{2}-1+u\right)\) |
| `uniform` | \(u\sim U(0,1)\) | \(u\) |

Return a 1-D float32 tensor with the same shape as `samples`.

## Example

`logit_normal` with `z = [-1, 0, 1]`, `logit_mean = 0`, `logit_std = 1`:

```
[sigmoid(-1), sigmoid(0), sigmoid(1)] = [0.268941, 0.5, 0.731059]
```

`mode` with `u = [0, 0.25, 0.5, 0.75, 1]`, `mode_scale = 1.29`:

```
[1.0, 0.616416, 0.5, 0.383584, 0.0]
```

## Constraints

- `scheme` must be one of `logit_normal`, `mode`, `uniform` (the source silently falls back to uniform for unknown strings; here it must raise `ValueError`).
- `samples` must be a 1-D float tensor with no nan / inf; `mode` and `uniform` additionally require values in \([0,1]\).
- `logit_normal` requires a finite `logit_mean` and `logit_std >= 0`; `mode` requires a finite `mode_scale >= 0`.
- Raise `ValueError` when any of the above fails.
- Degenerate properties: for `mode`, \(u=0\) maps to exactly 1, \(u=1\) to exactly 0, \(u=0.5\) to exactly 0.5, and the map is symmetric about the center, \(v(u)+v(1-u)=1\); `mode_scale=0` reduces it to \(1-u\). For `logit_normal`, \(z=0\) gives \(\sigma(\mu)\) and `logit_std=0` makes the whole batch the same constant.
