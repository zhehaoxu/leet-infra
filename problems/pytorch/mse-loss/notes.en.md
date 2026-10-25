# MSE Loss in Detail

## Definition

$$ \mathcal L_{\text{MSE}}=\frac1n\sum_{i=1}^{n}(p_i-t_i)^2 \quad(\text{mean});\qquad \sum_i(p_i-t_i)^2\quad(\text{sum}) $$

The squared error per element, then reduced according to `reduction`. The gradient is $\partial\mathcal L/\partial p_i = \frac{2}{n}(p_i-t_i)$ (mean).

## Why Squared

- **Smooth and differentiable**: gradients exist everywhere, which favours optimization; the penalty is light near zero and heavy far away.
- **Corresponds to maximum likelihood under Gaussian noise**: if the target is $t=p+\varepsilon,\ \varepsilon\sim\mathcal N(0,\sigma^2)$, minimizing MSE is equivalent to MLE.
- **Sensitivity to outliers** is a double-edged sword: one bad annotation noticeably inflates the loss - use Huber/L1 when robustness is needed (see huber-loss).

## Convention Pitfalls

- **There is no $\frac12$ coefficient**. Some textbooks write $\frac12(p-t)^2$ (to keep the gradient clean), but PyTorch's `F.mse_loss` does not add $\frac12$. The judge aligns with `F.mse_loss`, so do not supply a $\frac12$ yourself.
- `mean` divides by the **total number of elements** `numel` (including the batch and feature dimensions), not the number of samples; `reduction="none"` keeps the full shape.
- Broadcasting: `target` may be a scalar or a broadcastable shape; `(pred-target)` is broadcast first and then squared.

## Convention of the reduction Family

All loss problems (MSE/BCE/CE/Huber) share the same `reduction∈{mean,sum,none}` convention, which matches the semantics of the `reduction` argument of `torch.nn.functional.*`; an unknown value raises `ValueError`.

## Related

- The L1 loss $=\frac1n\sum|p-t|$ (more stable against outliers);
- Huber follows MSE for small residuals and L1 for large ones, taking the best of both;
- MAE / MSLE and the rest are all variants of this expression (changing the exponent or taking logarithms first).
