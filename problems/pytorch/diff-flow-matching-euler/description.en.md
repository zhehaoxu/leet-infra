# Diffusion: Euler Integration of a Rectified-Flow Velocity Field

## Problem

Flow Matching / Rectified Flow stop predicting noise and learn a **velocity field** \(v_\theta(x,t)\), phrasing "noise to data" as an ordinary differential equation from \(t=1\) to \(t=0\):

\[ \frac{dx}{dt}=v_\theta(x,t) \]

The ideal RF path is a straight line, \(x_t=(1-t)x_0+t\epsilon\), whose matching velocity is the constant \(v=\epsilon-x_0\). Sampling is explicit Euler on that field:

\[ t_k=t_{\text{start}}+k\cdot\frac{t_{\text{end}}-t_{\text{start}}}{n},\qquad x_{k+1}=x_k+(t_{k+1}-t_k)\,v_\theta(x_k,t_k) \]

Note that first-order Euler is **exact** for a constant velocity (with a straight path every number of steps stays on the line). That is fundamentally different from DDPM's stochastic differencing, and it is why RF can sample in very few steps.

## Rules

Given `x_init` (the state at \(t=t_{\text{start}}\)), `velocity_fn(x, t)`, the step count `n_steps`, and the endpoints `t_start` / `t_end` (with `t_end < t_start`), return `(trajectory, x_final)`: `trajectory` holds `n_steps + 1` float32 tensors including the start, and `x_final` is its last element. The velocity may be a same-shaped tensor or a scalar (broadcast).

## Examples

\[ x_{\text{init}}=1.0,\ v(x,t)=-x,\ n=4,\ t:\ 1\to0\quad(\Delta t=-0.25) \]

```
trajectory = [1.0, 1.25, 1.5625, 1.953125, 2.44140625]
x_final    = 2.44140625
```

## Constraints

- `n_steps` must be a positive integer; `t_start` / `t_end` must be finite reals with `t_end < t_start`; `velocity_fn` must be callable.
- A velocity tensor must match the input shape, otherwise raise `ValueError`.
- **Degeneracies**: for a constant velocity \(v\equiv c\), \(x_{\text{final}}=x_{\text{init}}+(t_{\text{end}}-t_{\text{start}})c\); on the RF straight path, starting from \(x_1=\epsilon\) recovers \(x_0\) exactly in a single step, and just as exactly in many.
- The first trajectory entry must still be the start (the implementation may not overwrite an already stored state).
