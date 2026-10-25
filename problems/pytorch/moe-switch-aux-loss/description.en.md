# MoE Routing: Switch Load-Balancing Auxiliary Loss

## Problem

MoE routers tend to "collapse" onto a few popular experts. Switch Transformer proposed a **differentiable** auxiliary loss that presses in two directions at once:
- make "the fraction actually dispatched $f_e$" uniform;
- make "the average probability output by the router $P_e$" uniform.

$$ \mathcal{L}_{\text{aux}} = E \cdot \sum_{e=1}^{E} f_e \cdot P_e $$

## Rules

Implement `solve(logits, k)`, where `logits` has shape $(T, E)$:

1. `probs = softmax(logits, dim=-1)`.
2. Take the $k$ largest entries of `probs` in each row → `selected (T, k)` long.
3. $f_e = \dfrac{\text{count}(e \in \text{selected})}{T}$ (**note the denominator is $T$, not $T \cdot k$**, so $\sum_e f_e = k$).
4. $P_e = \dfrac{1}{T}\sum_t \text{probs}[t,e]$ ($\sum_e P_e = 1$).
5. $\text{aux} = E \cdot \sum_e f_e P_e$.

Return a float scalar.

> Consistent with HF `transformers.models.mixtral.load_balancing_loss_func` (the attention_mask=None branch).

## Examples

```
logits = [[2,0,0,0],
          [0,2,0,0],
          [0,0,2,0],
          [0,0,0,2],
          [1,1,0,0],
          [0,1,1,0]], k=1

aux = 1.025673
```

Extreme contrasts:
- All 4 tokens select the same expert: aux ≈ **3.48**;
- Each expert receives 1 token and each row's softmax is a sharp peak: aux = **1.0** (the lower bound).

## Constraints

- Everything is CPU float32.
- Ties are broken in favor of the smaller expert index (consistent with `torch.topk`).
