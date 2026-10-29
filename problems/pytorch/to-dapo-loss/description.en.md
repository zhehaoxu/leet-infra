# DAPO: Clip-Higher, Dynamic Sampling, and Token-Level Loss

## Problem

**DAPO** (Yu et al. 2025, *DAPO: An Open-Source LLM Reinforcement Learning System at Scale*, arXiv:2503.14476) is an engineering overhaul of GRPO in four places, three of which live in the loss:

1. **Clip-Higher**: split PPO's symmetric clipping into two independent radii $\epsilon_\text{low}$ and $\epsilon_\text{high}$, usually with $\epsilon_\text{high}>\epsilon_\text{low}$. Symmetric clipping also limits how far a low-probability token's probability can rise, collapsing policy entropy too early; a looser upper bound gives a few critical tokens more room.
2. **Dynamic sampling (group filtering)**: for groups whose in-group accuracy $acc_b$ is 0 or 1, all advantages are 0, so they contribute nothing to the gradient yet still cost a forward pass. DAPO simply drops them, keeping only groups with $0<acc_b<1$.
3. **Token-level loss**: instead of "average within each answer, then average over answers", pool **all valid tokens of the kept groups** and take one global average. Every token in a long answer then weighs the same as in a short one, with no reweighting by length.

This problem gives per-token logprobs, a mask, and 0/1 correctness rewards; compute the loss in the three steps above.

## Rules

Given $\text{logp}/\text{logp\_old}$ of shape $(B,G,T)$, a same-shaped boolean mask, rewards $(B,G)$ strictly 0/1, $\epsilon_\text{low}\in[0,1)$, $\epsilon_\text{high}\ge0$:

1. **In-group advantage** (GRPO convention): $A[b,i]=\big(r[b,i]-\bar r_b\big)/\mathrm{std}_b$, with population std; if that group's std is 0 the whole group gets $A=0$.
2. **Group filtering**: $acc_b=\bar r_b$; keep groups with $0<acc_b<1$; output `keep` ($(B,)$ bool).
3. **Asymmetric clipping**: $\rho=\exp(\text{logp}-\text{logp\_old})$, $\bar\rho=\mathrm{clamp}(\rho,1-\epsilon_\text{low},1+\epsilon_\text{high})$, surrogate $=\min(\rho A,\ \bar\rho A)$.
4. **Global token average**: $L=-\dfrac{\sum_{b,i,t}\text{valid}\cdot\text{surrogate}}{N_\text{valid}}$ with $\text{valid}=\text{mask}\wedge\text{keep}$; if $N_\text{valid}=0$ then $L=0$ (no NaN). `num_tokens` is $N_\text{valid}$.

Return `loss` (0-dimensional float64), `adv` ($(B,G)$ float64), `keep` ($(B,)$ bool), `num_tokens` (0-dimensional int64).

## Examples

```
rewards = [[1.0, 0.0],      # acc = 0.5 -> kept
           [1.0, 1.0]]      # acc = 1.0 -> dropped
logp = [[[ 0.20, -0.10,  0.05], [ 0.00,  0.10, -0.20]],
        [[ 0.05,  0.05,  0.05], [-0.05,  0.00,  0.10]]]
mask = [[[1,1,1], [1,0,1]],
        [[1,1,1], [1,1,1]]]
ε_low = 0.2, ε_high = 0.28

A = [[ 1.0, -1.0], [0.0, 0.0]]
keep = [True, False]      num_tokens = 5
group 0: the positive-advantage answer has ρ = [1.221403, 0.904837, 1.051271], all within [0.8, 1.28], sum = 3.177511
         the negative-advantage answer (one padding) has ρ = [1.0, ·, 0.818731], surrogate = [-1.0, ·, -0.818731], sum = -1.818731
L = -(3.177511 - 1.818731) / 5 = -0.271756
```

Group 1 also has 6 valid tokens, but it is all-correct with zero advantages and is filtered out; `num_tokens` counts only 5.

## Constraints

- Four fixed keys; `adv` float64, `keep` bool, `num_tokens` int64, `loss` 0-dimensional float64.
- All-correct / all-wrong groups must be dropped; if the whole batch is dropped, the loss is 0 with `num_tokens=0`, never NaN.
- Loosening the upper bound must make the surrogate of a positive-advantage sample larger (more negative loss); on the negative-advantage side the $\min$ keeps the more negative unclipped term.
- Invalid input raises `ValueError`: non-tensor / non-float / NaN·Inf / shape mismatch / $G<2$ / $T<1$ / rewards not 0/1 / mask not bool or mismatched / an empty answer / $\epsilon_\text{low}\notin[0,1)$ / $\epsilon_\text{high}<0$.
- CPU, deterministic, no random number calls.
