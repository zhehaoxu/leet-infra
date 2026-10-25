# BatchNorm (Forward) Explained

## Definition

Normalize independently for each feature channel along the **batch dimension** (usually dim=0), then apply the affine transform:

$$ \hat{x}=\frac{x-\mu_B}{\sqrt{\sigma_B^2+\epsilon}},\qquad y=\gamma\hat{x}+\beta $$

$\mu_B,\sigma_B^2$ are the mean/variance of the current batch, and $\gamma$ (weight), $\beta$ (bias) are per-channel learnable parameters.

## Training vs inference: the biggest trap in this problem

The two conventions are **different**, do not mix them up:

- **Training (train)**: use the **current batch**'s $\mu_B,\sigma_B^2$ (a biased estimate is fine; torch uses unbiased=False for the forward statistics by default).
- **Inference (eval)**: use the **moving averages** running_mean/running_var accumulated during training, independent of the current batch size (even a single sample works).

If the judge passes `training=True`, compute statistics over the current batch; if it passes running_* with `training=False`, use them. Choosing the wrong convention makes every number wrong.

## Dimensions and reductions

- For $(N,C,H,W)$: $\mu,\sigma$ have shape $(C,)$, and the reduction must **average over N, H, W** (dim=(0,2,3)), then reshape $(C,)$ to $(1,C,1,1)$ for broadcasting. Reducing only over the batch and forgetting H/W is a common mistake.
- BN in convolutions is often written as "sync BN": with multiple GPUs, $\mu,\sigma$ are aggregated across devices; this problem is single-GPU and does not involve that.

## Where eps goes

$\epsilon$ is **inside the square root**: $\sqrt{\sigma^2+\epsilon}$, preventing division by zero when the variance is zero; the default is $10^{-5}$. Putting it outside the root is a different formula.

## Reference implementation for comparison

Use `F.batch_norm(x, running_mean, running_var, weight, bias, training, momentum, eps)`; note that the passed-in running_* are still updated when `training=True` (but we only take the forward output).
