# Huber Loss (Huber Loss / Smoothed Absolute Loss)

## Problem

Huber loss is the robust loss for regression: for small residuals it behaves like MSE (quadratic, smooth, easy to optimize), while for large residuals it switches to linear (like MAE, not dominated by outliers). A threshold $\delta$ controls the switch point, balancing convergence quality against noise robustness.

## Rules

- Implement `solve(pred, target, delta=1.0, reduction="mean")`, element-wise with $z=|\text{pred}-\text{target}|$:
  $$ \mathcal L = \begin{cases}\tfrac12 z^2 & z\le\delta\\[3pt] \delta\,(z-\tfrac12\delta) & z>\delta\end{cases} $$
- At $z=\delta$ the two segments are **continuous in both value and first derivative** (both are $\tfrac12\delta^2$, both slopes are $\delta$), which is what "smooth" means.
- `reduction` takes `"mean"` / `"sum"` / `"none"`; an unknown value raises `ValueError`.
- The result must match `F.huber_loss(pred, target, delta=delta, reduction=...)`.

## Examples

```
Input: pred = [1.5, -1.0, 0.3, 4.0]
      target = [1.0, 0.5, 0.2, 1.0]
      absolute residual z = [0.5, 1.5, 0.1, 3.0]
delta=1.0, reduction="none" → [0.125, 1.0, 0.005, 2.5]
delta=1.0, reduction="mean" → 0.9075
delta=0.5, reduction="mean" → 0.5325
(terms with z>delta use the linear branch: at 0.5, = delta*(z-0.5*delta))
```

## Constraints

- `pred` supports float32 / float64; `delta > 0`
- Do not use `F.huber_loss` / `F.smooth_l1_loss`; write the piecewise form by hand (with `torch.where`)
