# Huber Loss Explained

## Definition

Let the residual be $z=|p-t|$ and the threshold $\delta$:

$$ \mathcal L_\delta(z)=\begin{cases}\tfrac12 z^2 & z\le\delta\\ \delta\,(z-\tfrac12\delta) & z>\delta\end{cases} $$

Small errors follow MSE, large errors follow the linear branch. The gradient with respect to $p$: when $z\le\delta$ it is $=(p-t)$, otherwise $=\delta\,\mathrm{sign}(p-t)$ (clipped to $\pm\delta$).

## Why it is designed this way

- The MSE gradient grows linearly with the residual: a single outlier (with a huge $z$) can skew the whole batch's gradient.
- The MAE gradient is constantly $\pm1$: robust, but non-differentiable near 0, and it penalizes fully even in the small-residual region, so convergence is slow.
- Huber takes the middle ground: quadratic inside (second-order information near the optimum, fast convergence) and linear outside with the gradient saturating at $\delta$ (outliers no longer amplified). $\delta$ is the knob for "how large a residual you are willing to tolerate before treating it as an outlier".

## The meaning of $\delta$ and convention pitfalls

- Large $\delta$ → more like MSE; small $\delta$ → more like MAE (shrinking it toward 0 gives the L1 scale).
- **Huber ≠ Smooth L1**: `F.smooth_l1_loss` is the $\delta=1$ special case of Huber, but its linear segment is written as $z-0.5$ (i.e. $\delta(z-\tfrac12\delta)$ with $\delta=1$). The two agree at $\delta=1$; for $\delta\ne1$ only Huber is tunable, so do not mix up the default thresholds of the two APIs.
- The piecewise boundary uses $z\le\delta$ for the quadratic branch (consistent with `F.huber_loss`); the function and its derivative are continuous on both sides, so which side gets the equality does not change the numerical result.

## Implementation

`torch.where(z<=delta, 0.5*z*z, delta*(z-0.5*delta))` does it in one vectorized pass, then reduce according to `reduction`.

## Family

Fair / Cauchy / LogCosh / Charbonnier are all robust-loss variants that are "quadratic inside and sub/super-quadratic outside"; Huber is commonly used for bounding-box regression in object detection, physical-quantity regression, and noisy regression.

## Related

mse-loss is its inner region; L1/MAE is the approximation of its outer region; the reduction convention is the same as in all the loss problems.
