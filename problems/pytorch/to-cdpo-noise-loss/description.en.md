# cDPO: Preference Loss with Label Noise

## Problem

Real preference data is not clean: annotators mis-click, and `chosen`/`rejected` may be swapped. **cDPO** (conservative DPO, proposed in Eric Mitchell's 2023 blog post *A note on DPO with noisy preferences & relationship to IPO*; a theoretical analysis of noisy preferences appears in Ray Chowdhury et al. 2024, *Provably Robust DPO: Aligning Language Models with Noisy Feedback*, arXiv:2403.00409) handles this with plain **label smoothing**: assume each preference label is flipped with probability $\varepsilon$, so the hard labels $\{1,0\}$ become soft labels $\{1-\varepsilon,\varepsilon\}$ and DPO's negative log-likelihood becomes

$$\mathcal{L}_\text{cDPO}=-\mathbb{E}\Big[(1-\varepsilon)\log\sigma(\beta\Delta)+\varepsilon\log\sigma(-\beta\Delta)\Big],\qquad \Delta=(\text{logp}_w-\text{ref}_w)-(\text{logp}_l-\text{ref}_l).$$

The structure of DPO is unchanged; it merely keeps a pull back toward the opposite direction on "correctly separated" samples, curbing overfitting to possibly wrong labels. At $\varepsilon=0$ it reduces exactly to DPO.

This problem gives four **sequence-level** logprobs; assemble the expression above.

## Rules

Given a batch of length $N\ge 1$ with four sequence-level logprobs of shape $(N,)$ and one shared dtype:

1. Per sample $\Delta=(\text{logp}_w-\text{ref}_w)-(\text{logp}_l-\text{ref}_l)$, logit $x=\beta\Delta$.
2. Loss $=\frac{1}{N}\sum\big[-(1-\varepsilon)\log\sigma(x)-\varepsilon\log\sigma(-x)\big]$.
3. Both log-sigmoids must use the **stable** form (equivalent to `F.logsigmoid`); never compute $\sigma$ then take log.
4. $\varepsilon\in[0,0.5]$, $\beta>0$; return a 0-dimensional float64 scalar.

## Examples

```
logp_w = [-1.0]   ref_w = [-1.2]
logp_l = [-3.0]   ref_l = [-2.8]
β = 0.5, ε = 0.1

Δ = (-1.0 + 1.2) - (-3.0 + 2.8) = 0.4
x = β·Δ = 0.2
logσ(0.2)  = -0.598139
logσ(-0.2) = -0.798139
loss = -(0.9 × -0.598139 + 0.1 × -0.798139) = 0.618139
```

For comparison, the same inputs with $\varepsilon=0$ give $0.598139$ — the noise term raises it, which is exactly "keeping a reverse pull on already-separated samples". If $\Delta=0$ ($x=0$), the loss is $\ln 2$ for every $\varepsilon$.

## Constraints

- Return a 0-dimensional float64 scalar; float32 inputs are promoted to float64.
- $\varepsilon=0$ must match DPO numerically; at $\varepsilon=0.5$ the soft labels are symmetric and the loss is $-\frac12(\log\sigma(x)+\log\sigma(-x))$.
- When already separated ($x>0$), increasing $\varepsilon$ must strictly increase the loss.
- Invalid input raises `ValueError`: non-tensor / non-1-D / non-float / shape or dtype mismatch / empty batch / NaN·Inf / $\beta\le 0$ / $\varepsilon\notin[0,0.5]$.
- CPU, deterministic, no random number calls.
