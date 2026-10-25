# Solution Notes · CV Auxiliary Loss

## Trade-off against the Switch aux loss

- Switch: $E \cdot \sum f_e P_e$, whose minimum is $= k$ (when perfectly balanced); its scale depends on $E$.
- CV: $\mathrm{CV}^2$, whose minimum is $\approx 0$ (when perfectly balanced); **dimensionless** and independent of $E$.

CV is better suited to "sharing one set of hyperparameters across models with different expert counts"; Switch is better suited to tuning a single model.

## Why penalize both importance and load

Penalizing only load: `load` is a zeroth-order quantity with no gradient (not differentiable). Penalizing only importance: the soft distribution may be uniform while the hard dispatch is still skewed. Summing the two squares injects both the "soft, differentiable signal" and the "hard, actual-load signal" into the gradient.

## std uses the biased form (unbiased=False)

The OLNN paper and common implementations all use $\sqrt{\frac{1}{E}\sum (x - \bar{x})^2}$; the biased version makes CV=0 when $E=1$, matching the intuition that "a single expert is necessarily balanced".

## eps=1e-6

Denominator protection: `load.mean()` and `importance.mean()` are always positive ($kT/E$, $T/E$), so eps only matters when $T=0$ or $k=0$. It is retained to align exactly with mainstream implementations.
