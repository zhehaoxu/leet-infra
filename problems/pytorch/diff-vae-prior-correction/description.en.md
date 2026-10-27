# Diffusion: VAE Latent Shift/Scale and Prior Correction

## Problem

Latent diffusion does not work on pixels but on VAE latents, and the two are separated by an **affine change of units**. SDXL uses a single `scaling_factor`; SD3 / Flux add a `shift_factor` (the latent mean is not zero):

\[ \text{encode}:\quad z=\big((x-\text{shift})\cdot\text{scale}-\mu_c\big)/\sigma_c,\qquad \text{decode}:\quad x=\frac{z\cdot\sigma_c+\mu_c}{\text{scale}}+\text{shift} \]

Here \(\mu_c,\sigma_c\) are **per-channel** prior-correction statistics (they whiten the latent toward \(N(0,1)\) so the noise prior used in training matches the distribution actually fed to the model at inference). The order matters: encoding subtracts `shift` before multiplying by `scale`, while decoding multiplies by \(\sigma_c\), divides by `scale`, and only then adds `shift`.

## Rules

Given `latents` (at least 2-D, dim 0 is the batch and dim 1 the channel), `scaling_factor` (must be positive), `shift_factor`, optional `channel_mean` / `channel_std` (one value per channel or a scalar; either both or neither; \(\sigma_c>0\)) and the direction switch `inverse`, return a float32 tensor. `inverse=False` is the decode direction (latent → pixel-space scale) and `inverse=True` the encode direction.

## Examples

\[ z=\begin{bmatrix}0.5&-1.0\\2.0&0.0\end{bmatrix},\ \text{scale}=0.18215,\ \text{shift}=0.0609,\ \mu_c=[0.2,-0.3],\ \sigma_c=[1.5,0.5] \]

```
with stats -> [[ 5.276382, -4.331085], [17.628838, -1.586094]]
no stats   -> [[ 2.805890, -5.429081], [11.040861,  0.060900]]
```

## Constraints

- `scaling_factor>0` and finite; `shift_factor` finite; `channel_std` must be strictly positive.
- `channel_mean` / `channel_std` come as a pair; their length must equal the channel count (or be a single-element scalar); 2-D statistics are rejected.
- **Identity and round trip**: with `scale=1, shift=0` and no statistics both directions are the identity; for arbitrary parameters `encode(decode(z))` must recover \(z\) element by element.
- **Per channel**: a channel of \(z=0\) maps to \(\mu_c/\text{scale}+\text{shift}\); the within-channel differences are amplified by \(\sigma_c/\text{scale}\).
