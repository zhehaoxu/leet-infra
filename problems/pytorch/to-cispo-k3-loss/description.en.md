# CISPO and K3: Stop-Gradient Clipped Weights + KL Estimation

## Problem

**CISPO** (MiniMax-M1, MiniMax 2025, *MiniMax-M1: Scaling Test-Time Compute Efficiently with Lightning Attention*, arXiv:2506.13585) repurposes PPO clipping. PPO's surrogate is $\min(\rho A,\ \mathrm{clip}(\rho)A)$: the ratio sits inside the objective, so a clipped token's gradient vanishes entirely (the `min` selects the constant term), wasting many tokens in long-sequence training. CISPO moves the clip — the clipped ratio becomes a **stop-gradient scalar weight** multiplying $\log\pi$:

$$L_\text{CISPO}=-\frac{1}{N_\text{valid}}\sum_{t}\text{mask}\cdot\mathrm{sg}\big(\mathrm{clamp}(\rho_t,1-\epsilon_\text{low},1+\epsilon_\text{high})\big)\,A\,(\log\pi_\theta)_t .$$

Every token now keeps a **bounded** gradient (weight within $[1-\epsilon_\text{low},1+\epsilon_\text{high}]$) and none is discarded; and $\mathrm{sg}(\cdot)$ ensures the gradient flows only through $\nabla\log\pi$, never through "the ratio itself being optimized" as an extra term.

The second component is the **K3 KL estimator** (Schulman's k3, from his blog *Approximating KL Divergence*): with samples from the current policy $\pi$, use $r=\pi_\text{ref}/\pi$ to form
$$k_3=r-\log r-1\ \ge 0,$$
an unbiased, per-sample non-negative estimator of $\mathrm{KL}(\pi\,\|\,\pi_\text{ref})$ with lower variance than $-\log r$ — commonly used to monitor or penalize deviation from the reference model.

This problem gives per-token logprobs, reference logprobs, a mask, and sequence advantages; compute the CISPO loss, K3 KL, and clipping fraction.

## Rules

Given $\text{logp},\text{logp\_old},\text{ref\_logp}$ all $(N,T)$, a same-shaped boolean mask (at least one True), advantages $(N,)$, $\epsilon_\text{low}\in[0,1)$, $\epsilon_\text{high}\ge0$:

1. **Importance ratio** $\rho_t=\exp(\text{logp}_t-\text{logp\_old}_t)$.
2. **Clipped weight** $w_t=\mathrm{clamp}(\rho_t,1-\epsilon_\text{low},1+\epsilon_\text{high})$, **with stop-gradient** (treated as a constant).
3. **Policy loss** $L=-\dfrac{\sum_t\text{mask}\cdot w_t\,A\,(\log\pi)_t}{N_\text{valid}}$, where $(\log\pi)_t$ is `logp`; $A$ is broadcast per answer across tokens.
4. **K3**: $\text{log\_r}=\text{ref\_logp}-\text{logp}$ (reference minus current), $k_3=e^{\text{log\_r}}-\text{log\_r}-1$, $\text{kl\_k3}=\dfrac{\sum_t\text{mask}\cdot k_3}{N_\text{valid}}$. To keep $e^{\text{log\_r}}$ finite, require $\max(\text{ref\_logp}-\text{logp})\le20$, else `ValueError`.
5. **Clipping fraction**: the proportion of valid tokens whose $\rho$ lies **strictly outside** $[1-\epsilon_\text{low},1+\epsilon_\text{high}]$ (exactly on the boundary does not count).

Return `loss`, `kl_k3`, `clip_frac`, all 0-dimensional float64.

## Examples

```
logp     = [[ 0.1, -0.2,  0.0], [ 0.3,  0.0,  0.5]]
logp_old = all zeros
ref_logp = [[ 0.0,  0.0,  0.0], [ 0.4,  0.0,  0.0]]
mask     = [[True, True, True], [True, True, False]]
adv      = [1.0, -0.5]        ε_low = ε_high = 0.2

ρ = [[1.105171, 0.818731, 1.0], [1.349859, 1.0, ·]]
w = [[1.105171, 0.818731, 1.0], [1.2,      1.0, ·]]     # row 2's first token is capped at 1.2
loss = -(1.105171·0.1 + 0.818731·(-0.2) + 1.2·0.3·(-0.5)) / 5 = 0.046646
k₃ = exp(ref-logp) - (ref-logp) - 1 = [0.004837, 0.021403, 0, 0.005171]
kl_k3 = 0.031411 / 5 = 0.006282
clip_frac = 1/5 = 0.2      # only ρ=1.349859 is out of range
```

Gradient check: $\partial L/\partial(\log\pi_t)=-w_t A/N_\text{valid}$, exactly 0 at masked tokens; because $w$ is stop-gradient, an unclipped token gains no extra $\rho A$ term.

## Constraints

- All three outputs are 0-dimensional float64; float32 inputs are promoted internally.
- Stop-gradient is the crux: without `sg`, unclipped positions gain an extra $\rho A/N$ in the gradient, detectable by an autograd cross-check.
- The K3 direction is `ref - logp` (reference minus current); it is per-token non-negative and 0 when the two coincide.
- `clip_frac` uses strict inequalities; a $\rho$ exactly on the boundary is not counted.
- Invalid input raises `ValueError`: non-tensor / non-float / NaN·Inf / shape mismatch / $N$ or $T<1$ / mask not bool or mismatched / no valid token / advantage shape or dtype mismatch / $\epsilon_\text{low}\notin[0,1)$ / $\epsilon_\text{high}<0$ / $\text{ref}-\text{logp}>20$.
- CPU, deterministic, no random number calls.
