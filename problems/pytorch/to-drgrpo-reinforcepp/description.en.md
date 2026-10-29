# Dr. GRPO and REINFORCE++: Two Normalizations of the Same Rollouts

## Problem

GRPO-family methods agree on "sample a group of answers per prompt and compute relative advantages within the group"; all the disagreement is in **normalization**. This problem asks you to compute two sets of advantages and two losses on the same batch of rollouts, putting the differences side by side:

- **Dr. GRPO** (Liu et al. 2025, *Understanding R1-Zero-Like Training: A Critical Perspective*, arXiv:2503.20783) identifies two biases in GRPO: normalizing per response length causes **length bias** (each token in a longer answer gets a smaller weight), and normalizing by the in-group std causes **difficulty bias** (gradients of easy/hard groups are amplified). Its fix removes both: the advantage is only group-centered, and the loss uses a single constant denominator $B\cdot G\cdot T$ over all tokens ($T$ is the tensor width $L_\text{max}$, not each answer's valid length).
- **REINFORCE++** (Hu 2025, *REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models*, arXiv:2501.03262) keeps whitening but places the statistics over the **whole batch** (global mean plus global population std), and averages each answer over its **own valid token count** before averaging over answers.

Both use symmetric PPO clipping $\mathrm{clip}(\rho,1-\epsilon,1+\epsilon)$ with the surrogate $\min(\rho A,\ \mathrm{clip}\cdot A)$; they differ only in where normalization happens and over what range.

## Rules

Given rewards $(B,G)$, per-token logprobs $(B,G,T)$ for the current/behaviour policies, a same-shaped boolean mask, and $\epsilon\in(0,1)$:

1. **Advantages**
   - $A_\text{dr}[b,i]=r[b,i]-\frac{1}{G}\sum_j r[b,j]$ (group-centered, no std division);
   - let $\mu,\sigma$ be the mean and **population** std (`unbiased=False`) of all $B\cdot G$ rewards, then
     $A_\text{rp}[b,i]=(r[b,i]-\mu)/\sigma$; if $\sigma\le10^{-12}$ the whole batch's advantages are 0.
2. **Per-token importance ratio** $\rho=\exp(\text{logp}-\text{logp\_old})$, clipped $\bar\rho=\mathrm{clamp}(\rho,1-\epsilon,1+\epsilon)$, surrogate $=\min(\rho A,\ \bar\rho A)$.
3. **Dr. GRPO loss**: $L_\text{dr}=-\dfrac{\sum_{b,i,t}\text{mask}\cdot\text{surrogate}}{B\cdot G\cdot T}$. The denominator includes padding positions, so padding dilutes the loss.
4. **REINFORCE++ loss**: $L_\text{rp}=-\dfrac{1}{BG}\sum_{b,i}\dfrac{\sum_t\text{mask}\cdot\text{surrogate}}{|o_{b,i}|}$. The denominator is each answer's valid length, so padding has no effect.

Return `adv_drgrpo`, `adv_reinforcepp` (each $(B,G)$ float64) and `loss_drgrpo`, `loss_reinforcepp` (each 0-dimensional float64).

## Examples

```
rewards = [[1.0, 0.0]]          G = 2
logp    = [[[ 0.30, -0.30,  0.00],
            [ 0.05,  0.00, -0.05]]]     logp_old = all zeros, mask all True
ε = 0.2

A_dr = [[ 0.5, -0.5]]        A_rp = [[ 1.0, -1.0]]
ρ    = [[[1.349859, 0.740818, 1.0], [1.051271, 1.0, 0.951229]]]
surrogate (Dr., A=±0.5) = [[0.6, 0.370409, 0.5], [-0.525636, -0.5, -0.475615]]
L_dr = -(-0.030842) / (1·2·3) = 0.005140
L_rp = -mean( 1.470409/3, -1.501251/3 ) = 0.010280
```

The REINFORCE++ loss is exactly twice the Dr. GRPO one here: the global std is 0.5, so whitening doubles the advantages, while both answers have the same length and thus the same divisor.

## Constraints

- All four outputs are float64; float32 inputs are promoted internally. `adv_*` are $(B,G)$, `loss_*` are 0-dimensional.
- Appending a fully masked padding column shrinks $L_\text{dr}$ to $3/4$ while leaving $L_\text{rp}$ unchanged — the most direct observable difference between the two normalizations.
- Identical rewards make $\sigma=0$; REINFORCE++ advantages and both losses are then 0 (tolerance $10^{-12}$, no division by zero).
- Invalid input raises `ValueError`: non-tensor / non-float / NaN·Inf / shape mismatch / $G<2$ / $T<1$ / mask not bool or mismatched / an empty answer / $\epsilon\notin(0,1)$.
- CPU, deterministic, no random number calls.
