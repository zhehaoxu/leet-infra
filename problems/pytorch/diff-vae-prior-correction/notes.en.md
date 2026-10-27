# In Depth: VAE Latent Shift/Scale and Prior Correction

## Origin

SDXL (Podell et al., arXiv:2307.01952) normalizes VAE latents to unit scale with `scaling_factor=0.13025`; SD3 / Flux switch to `scaling_factor=0.3611` and `shift_factor=0.1159` with a linear correction of the latent statistics; Waltersdorfer et al. (ICLR 2024) analyze the encode-decode inconsistency caused by a latent distribution that does not match the \(N(0,1)\) prior and give the linear whitening recipe.

## Core points and pitfalls

1. **The order is fixed**: encoding does \((x-\text{shift})\cdot\text{scale}\) and decoding does \(z/\text{scale}+\text{shift}\). Swapping shift and scale, or adding shift before dividing, shifts \(x\) by roughly \(\text{shift}\cdot(\text{scale}-1)\) — the judge pins this down with the single point `scale=0.5, shift=0.25`.
2. **Per-channel statistics live on dim 1**: \(\mu_c,\sigma_c\) have shape \((C,)\) and must be reshaped to \((1,C,1,1)\) when broadcasting; broadcasting on the last axis silently permutes channels. A scalar statistic means "the same value on every channel".
3. **`scaling_factor` must be positive**: it is a scale factor; negatives are not mathematically impossible but flip the semantics, so the interface rejects them.
4. **The statistics come as a pair**: supplying a mean without a std is meaningless and must raise (not silently default to std=1).
5. **The round trip is the strongest assertion**: for any `scale/shift/mean/std`, `encode(decode(z))` must recover \(z\), which also checks that decoding multiplies by \(\sigma_c\) while encoding divides by it.
6. **Numerical range**: decoded values are about \(1/\text{scale}\approx 5.5\times\) larger, still comfortable for float32; accumulating intermediates in float16 would introduce visible error, so everything runs in float64 internally.

## Verification approach

The reference solution computes in float64 and casts to float32 at the exit. The judge's `_oracle` looks up statistics per \((batch,channel)\) and recomputes both directions in python floats, adding four structural groups: the example values (with and without statistics), round trips over five parameter combinations, the single-point order check (`scale=0.5`, `shift=0.25`) plus the default identity, and the per-channel zero mapping with its within-channel amplification. Tolerance `allclose(atol=1e-6)` (relaxed to `1e-5` for round trips).

## Complexity

Element-wise \(O(N)\) with no reductions (the statistics are supplied).
