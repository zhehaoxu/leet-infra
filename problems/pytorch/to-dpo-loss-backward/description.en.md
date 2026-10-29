# Analytic Backward Pass of the DPO Loss

## Problem

Everyone knows the DPO forward loss (Rafailov et al. 2023, *Direct Preference Optimization: Your Language Model is Secretly a Reward Model*, arXiv:2305.18290):

$$L=-\frac{1}{B}\sum_b\log\sigma(x_b),\qquad x_b=\beta\Delta_b,\qquad \Delta_b=(s_w-\text{ref}_w)-(s_l-\text{ref}_l),$$

where $s_w,s_l$ are the **sequence-level** log probabilities of the chosen/rejected answers (per-token logprobs summed over valid positions). Usually we delegate the backward pass to `autograd`; but for fused kernels, custom RL loops, or verifying a framework's gradients, the **analytic gradient** is an irreplaceable reference.

This problem asks: given per-token logprobs and masks, return the loss and the analytic gradient w.r.t. the per-token logprobs on both sides. The answer is closed-form — one chain rule — but the signs, the $1/B$ scaling, and the masked positions must be exact.

## Rules

Given $\text{logp\_w},\text{logp\_l}$ of shape $(B,T)$, reference sequence logprobs $\text{ref}_w,\text{ref}_l$ of shape $(B,)$, two boolean masks $(B,T)$ (at least one True per side per sample), and $\beta>0$:

1. **Sequence log probabilities**: $s_w=\sum_t\text{mask}_w\cdot\text{logp}_w$, and likewise $s_l$.
2. **Implicit reward difference**: $\Delta=(s_w-\text{ref}_w)-(s_l-\text{ref}_l)$, $x=\beta\Delta$.
3. **Forward**: $L=\frac{1}{B}\sum_b\mathrm{softplus}(-x_b)$ (equivalent to $-\log\sigma(x)$; use the stable form).
4. **Backward**: $\dfrac{\partial L}{\partial x_b}=-\sigma(-x_b)$, hence
   $$\frac{\partial L}{\partial\Delta_b}=-\frac{\beta\,\sigma(-x_b)}{B},\qquad
   \frac{\partial L}{\partial\text{logp\_w}[b,t]}=\frac{\partial L}{\partial\Delta_b}\,\text{mask}_w[b,t],\qquad
   \frac{\partial L}{\partial\text{logp\_l}[b,t]}=-\frac{\partial L}{\partial\Delta_b}\,\text{mask}_l[b,t].$$
   The signs are opposite on the two sides: raising the rejected probability lowers $\Delta$ and raises the loss.

Return `loss` (0-dimensional float64), `grad_logp_w`, `grad_logp_l` (each $(B,T)$ float64).

## Examples

```
logp_w = [[-0.5, -0.4, -0.6], [-0.2, -0.3, -0.1]]     ref_w = [-1.5, -1.0]
logp_l = [[-0.7, -0.8, -0.9], [-0.4, -0.5, -0.6]]     ref_l = [-2.4, -1.5]
mask_w = all True
mask_l = [[True, True, False], [True, True, True]]
β = 0.1

sample 0: s_w = -1.5, s_l = -1.5
          Δ = (-1.5 + 1.5) - (-1.5 + 2.4) = -0.9      x = -0.09
sample 1: s_w = -0.6, s_l = -1.5
          Δ = (-0.6 + 1.0) - (-1.5 + 1.5) =  0.4      x =  0.04

loss = mean(softplus(0.09), softplus(-0.04)) = mean(0.739159, 0.673347) = 0.706253
dL/dΔ = -β·σ(-x)/B = [-0.026124, -0.024500]
grad_logp_w = [[-0.026124]×3, [-0.024500]×3]
grad_logp_l = [[ 0.026124,  0.026124, 0.0], [0.024500]×3]   # masked positions are exactly 0
```

## Constraints

- All three outputs are float64; `loss` is 0-dimensional, the two gradients are $(B,T)$.
- Masked positions have gradient **exactly 0**, not some shared constant; changing masked logprobs must not affect the loss or any other position's gradient.
- The $1/B$ mean scaling must appear in the gradients: putting the same sample into a larger batch shrinks each token's gradient.
- At $\Delta=0$, $x=0$, $L=\ln2$ and $\partial L/\partial\Delta=-\beta/(2B)$ (proportional to $\beta$).
- Invalid input raises `ValueError`: non-tensor / logp not 2-D or shape mismatch / non-float / NaN·Inf / reference not $(B,)$ or dtype mismatch / mask not bool or mismatched / a sample with no valid token on one side / $\beta\le0$.
- CPU, deterministic, no random number calls; autograd may be used to cross-check, but the returned values must be the analytic expressions.
