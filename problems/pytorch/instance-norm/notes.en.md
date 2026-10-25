# InstanceNorm (Forward) in Detail

## Definition

Normalize **each sample and each channel separately** over the spatial dimensions:

$$ \hat{x}_{n,c}=\frac{x_{n,c}-\mu_{n,c}}{\sqrt{\sigma_{n,c}^2+\epsilon}}\cdot\gamma_c+\beta_c $$

The statistics are computed only over $(H,W)$ and are shared neither across samples nor across channels.

## Where it is used

- Style transfer (the core of AdaIN is per-channel IN followed by mean/variance alignment), GAN generators. IN removes each image's own contrast/brightness statistics, which suits "content–style" disentanglement.
- Relation to GN: IN is equivalent to GN with $G=C$ (each channel forms its own group).

## Dimension and parameter pitfalls

- For $(N,C,H,W)$: $\mu,\sigma$ have shape $(N,C)$, reduction dim=(2,3), reshape to $(N,C,1,1)$ for broadcasting.
- With `affine=True`, $\gamma,\beta$ have shape $(C,)$ (shared per channel, not per sample); `track_running_stats` is generally False (IN usually keeps no running stats; both training and inference compute per sample on the fly).

## Matching the reference solution

`F.instance_norm(x, running_mean, running_var, weight, bias, use_input_stats=True, momentum, eps)`—note that IN always uses the input's own statistics.
