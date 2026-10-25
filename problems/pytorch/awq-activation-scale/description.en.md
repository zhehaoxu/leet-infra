# AWQ: Activation-Aware Per-Channel Scaling

## Problem

**AWQ** (Activation-aware Weight Quantization, Lin et al., 2024, [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)) is used for **weight quantization** such as W4A16: not all channels of the weights are equally important — **weights connected to large-activation channels** influence the output more and their quantization error hurts more. AWQ assigns each input channel a scaling factor $s_j$ (proportional to that channel's activation magnitude raised to the power $\alpha$); before quantization it "magnifies" the weights of important channels so that, at a fixed bit width, they receive more quantization levels and smaller error; during dequantization it divides by $s_j$ again. $\alpha$ is chosen by a small grid search.

## Rules

Given weights $W$ of shape $(O, C)$, the per-input-channel mean activation magnitude `x_bar` of shape $(C,)$, a candidate exponent grid `alphas`, and the quantization bit width `bits`. For a given $\alpha$:

1. Raw importance $r_j = \text{x\_bar}_j^{\alpha}$; **normalize** $s_j = r_j / \operatorname{mean}_k r_k$ (so the scaling does not change the overall magnitude).
2. Quantize after scaling: $\tilde{W} = W \odot s$ (multiply by column), apply **per-tensor symmetric RTN** to $\tilde{W}$ ($q_{max}=2^{b-1}-1$, scale $=\max|\tilde{W}|$, round to nearest then dequantize) to get $\widetilde{W}_q$.
3. Scale back: $\hat{W} = \widetilde{W}_q / s$ (divide by column).
4. **Output error (diagonal approximation)**: $\mathcal{L}(\alpha)=\sum_{o,j} \text{x\_bar}_j^2\,(W_{oj}-\hat W_{oj})^2$.

Choose $\alpha^\*$ over `alphas` that minimizes $\mathcal{L}$ (on ties, take the one that **appears first** in the grid). Return `(best_alpha, W_hat)`, where `W_hat` has shape $(O,C)$.

## Examples

```
W = [[2.0, 1.0, 0.10, 3.0],
     [1.0, 2.0, 0.12, 2.0],
     [0.5, 1.0, 0.08, 1.0]]
x_bar = [0.2, 0.3, 9.0, 0.1]   # channel 2 has huge activations but tiny weights → needs protection
alphas = [0.0, 0.25, 0.5, 0.75, 1.0],  bits = 4

best_alpha = 1.0
s = x_bar / mean(x_bar) = [0.0833, 0.1250, 3.7500, 0.0417]
W_hat ≈ [[2.3143, 1.0286, 0.1029, 3.0857],
         [0.7714, 2.0571, 0.1200, 1.5429],
         [0.7714, 1.0286, 0.0857, 1.5429]]
```

(Channel 2 is magnified by $\times 3.75$ before quantization, so after scaling back its weight quantization error is significantly lower than at $\alpha=0$.)

## Constraints

- CPU float32; `best_alpha` is a `float`, `W_hat` is compared with `allclose(atol=1e-4)`.
- Per-tensor RTN rounds to **nearest, with `.5` to even** (matching `torch.round`).
- Every element of `x_bar` is $>0$; the normalization denominator is the arithmetic mean of $r$.
