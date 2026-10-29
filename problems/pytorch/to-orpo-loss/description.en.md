# ORPO: Reference-Free Odds-Ratio Preference Optimization

## Problem

**ORPO** (Hong et al. 2024, *ORPO: Monolithic Preference Optimization without Reference Model*, arXiv:2403.07691) squeezes supervised fine-tuning and preference alignment into a **single** training stage: no reference model, no separate "SFT then align" pipeline — directly on preference data it simultaneously raises the likelihood of the chosen answer and separates the odds of chosen versus rejected.

DPO uses the reference model's logprob as an anchor to limit drift; ORPO takes a different route — the **SFT term on the chosen answer itself** acts as the regularizer, and an **odds ratio** replaces the probability ratio in the preference term. The odds $p/(1-p)$ are more sensitive as $p$ approaches 1: once the model already assigns high probability to the chosen answer, differences in odds get amplified and keep supplying gradient — exactly the gap left by sigmoid-type preference losses saturating on "already-correct" pairs.

This problem gives four **already length-normalized** sequence-level quantities (two sequence logprobs plus two lengths); you only have to assemble the SFT term and the odds term.

## Rules

Given a batch of length $N\ge 1$:

1. **Length-normalize first**: $\ell_w=\text{logp}_w/|\text{y}_w|$, $\ell_l=\text{logp}_l/|\text{y}_l|$. Both must be strictly negative, otherwise $p=e^{\ell}$ is not a valid probability and $\log(1-p)$ is undefined.
2. **Log-odds** must use the stable form
   $$\mathrm{logodds}(\ell)=\ell-\log\!\big(1-e^{\ell}\big)=\ell-\mathrm{log1p}(-e^{\ell}),$$
   not "compute $p$, then $\log\frac{p}{1-p}$" (which loses all precision as $p\to 1$).
3. **SFT term** (NLL on the preferred answer only): $L_\text{SFT}=-\frac{1}{N}\sum \ell_w$.
4. **Odds term**: $L_\text{OR}=-\frac{1}{N}\sum \log\sigma\big(\mathrm{logodds}(\ell_w)-\mathrm{logodds}(\ell_l)\big)$, with a stable log-sigmoid.
5. **Total loss**: $L=L_\text{SFT}+\lambda\,L_\text{OR}$, $\lambda\ge 0$ defaulting to $0.1$; no $\beta$ factor and no reference term.

Return a 0-dimensional float64 scalar, accumulating in float64 throughout.

## Examples

```
logp_w = [-2.0, -1.5]   |y_w| = [2, 3]
logp_l = [-3.0, -2.4]   |y_l| = [3, 4]
λ = 0.1

length norm: ℓ_w = [-1.0, -0.5]      ℓ_l = [-1.0, -0.6]
log-odds:    logodds(ℓ_w) = [-0.541325, 0.432752]
             logodds(ℓ_l) = [-0.541325, 0.195870]
odds diff:   d = [0.0, 0.236882]
SFT term:    -mean(ℓ_w) = 0.75
odds term:   -mean(logsigmoid(d)) = 0.637426
total loss:  0.75 + 0.1 × 0.637426 = 0.813743
```

Note: if sample 2's odds difference were mistaken for $\ell_w-\ell_l=0.1$, the odds term would be a different number — the gap is far above $10^{-3}$ and immediately gives away a wrong implementation.

## Constraints

- Return a 0-dimensional float64 scalar; inputs may be float32 but everything must be promoted to float64 internally.
- $\lambda=0$ degenerates strictly to SFT; equal **mean** logprobs on both sides make the odds difference 0 and the odds term exactly $\ln 2$.
- Sensitive to length normalization: spreading the same sequence logprob over a longer answer changes $\ell$ and hence both terms.
- Invalid input raises `ValueError`: non-tensor / non-1-D / non-float logp / non-integer lengths / mismatched shapes / length < 1 / NaN·Inf / $\lambda<0$ / length-normalized value $\ge 0$.
- CPU, deterministic, no random number calls.
