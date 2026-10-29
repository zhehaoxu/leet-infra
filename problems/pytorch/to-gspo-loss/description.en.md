# GSPO: Sequence-Level Importance Ratio

## Problem

**GSPO** (Zheng et al. 2025, *Group Sequence Policy Optimization*, arXiv:2507.18071) redefines the importance ratio in large-model RL: PPO/GRPO use a **per-token** $\rho_t=\pi_\theta(o_t)/\pi_\text{old}(o_t)$ with per-token clipping; GSPO treats the whole answer as one action and uses a **length-normalized sequence ratio**

$$s_i=\Big(\frac{\pi_\theta(y_i|x)}{\pi_\text{old}(y_i|x)}\Big)^{1/|y_i|}=\exp\!\Big(\frac{1}{|o_i|}\sum_t\log\frac{\pi_\theta(o_{i,t}|x,o_{i,<t})}{\pi_\text{old}(o_{i,t}|x,o_{i,<t})}\Big),$$

with one clipping decision shared by the whole answer.

Two motivations. First, **variance**: token-level ratio variance accumulates with sequence length and a few extreme tokens make the gradient jump; the sequence ratio is the geometric mean of token ratios, so a single token can only move $s$ by $1/|o|$ — naturally smooth. Second, **MoE stability**: token-level PPO's derivation implicitly assumes routing is consistent across steps, but MoE routing drifts, making token ratios fluctuate widely; the sequence ratio averages that away and makes RL on MoE far more stable.

This problem gives per-token logprobs, a mask, and sequence-level advantages; compute the GSPO loss and sequence ratios.

## Rules

Given $\text{logp}/\text{logp\_old}$ of shape $(N,T)$, a same-shaped boolean mask (at least one True per answer), sequence advantages $A$ of shape $(N,)$, and clipping radius $\epsilon\in(0,1)$ (the paper uses $3\times10^{-4}$):

1. **Average first, then exponentiate**: $\log s_i=\dfrac{\sum_t\text{mask}\cdot(\text{logp}-\text{logp\_old})}{|o_i|}$, $s_i=e^{\log s_i}$. Not "exponentiate per token then average", nor "exponentiate the raw sum".
2. **One clipped ratio per answer**: $\bar s_i=\mathrm{clamp}(s_i,1-\epsilon,1+\epsilon)$, objective $\text{obj}_i=\min(s_i A_i,\ \bar s_i A_i)$.
3. **Average over answers**: $L=-\dfrac{1}{N}\sum_i\text{obj}_i$, not over tokens.

Return `loss` (0-dimensional float64) and `ratio` ($(N,)$ float64, i.e. $s$).

## Examples

```
logp = [[ 0.10, -0.20,  0.00],
        [ 0.30,  0.00,  0.50]]      logp_old = all zeros
mask = [[True, True, True],
        [True, True, False]]        adv = [1.0, -1.0]
ε = 3e-4

answer 0: log s = (0.1 - 0.2 + 0.0)/3 = -0.033333   s = 0.967216
answer 1: log s = (0.3 + 0.0)/2        =  0.150000   s = 1.161834
obj₀ = min(0.967216×1, 0.9997×1) = 0.967216
obj₁ = min(1.161834×(-1), 1.0003×(-1)) = -1.161834
loss = -(0.967216 - 1.161834)/2 = 0.097309
```

Answer 0 has $s<1$ with positive advantage, so `min` takes the unclipped term; answer 1 has negative advantage and `min` likewise takes the more negative unclipped term — the clipping upper bound does nothing on the negative side, inherent to the PPO-style surrogate.

## Constraints

- `loss` 0-dimensional, `ratio` $(N,)$, both float64; float32 inputs are promoted internally.
- Masked positions never enter the log-ratio: changing masked logprobs arbitrarily leaves $s$ and the loss unchanged.
- Length normalization happens **before** exponentiation: the same total delta spread over 3 tokens versus 1 token yields different $s$.
- Average over answers, not tokens: answers of length 3 and 1 carry the same weight.
- Invalid input raises `ValueError`: non-tensor / non-float / NaN·Inf / logp not 2-D / shape mismatch / $N$ or $T<1$ / mask not bool or mismatched / an empty answer / advantage shape or dtype mismatch / $\epsilon\notin(0,1)$.
- CPU, deterministic, no random number calls.
