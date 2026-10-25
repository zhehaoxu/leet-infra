# LLM.int8(): Mixed-precision Split on Outlier Columns

## Problem Description

**LLM.int8()** (Dettmers et al., 2022, [arXiv:2208.07339](https://arxiv.org/abs/2208.07339)) is the earliest scheme for handling the "emergent feature dimension": in matrix multiplication, a very small number of **input channels** persistently produce huge activations, which inflate their quantization error enough to destroy accuracy. The method cuts the weights in two along the input channels — **outlier columns** are kept at full FP16 precision, while the remaining "normal columns" are INT8-quantized — then the two halves are multiplied separately and added. Equivalently: keep the outlier columns' weights **verbatim** and quantize only the rest.

## Rules

Given weights $W$ of shape $(O, C)$, outlier threshold $\tau > 0$, and bit-width `bits` (default 8).

1. **Outlier column decision**: column $j$ is an outlier column iff there exists some $o$ with $|W_{oj}| > \tau$, i.e. $\max_o |W_{oj}| > \tau$. This yields `outlier_mask` of shape $(C,)$ (`bool`).
2. **Quantize normal columns**: apply **per-tensor symmetric RTN** to all **non-outlier columns** — range scale $=\max\{|W_{oj}|: j\ \text{not outlier}\}$, $q_{max}=2^{bits-1}-1$, $q=\operatorname{round}(W/\text{scale}\cdot q_{max})$ clipped to $[-q_{max}-1,\,q_{max}]$, dequantized as $q/q_{max}\cdot\text{scale}$.
3. **Keep outlier columns verbatim** (treated as full-precision FP).

Return `(W_recon, outlier_mask)`, where $W_{recon}$ has shape $(O,C)$: outlier columns equal the original $W$, and normal columns hold their dequantized values.

## Example

```
W = [[0.1, 0.2, 9.0, 0.3],
     [0.4, 0.5, 8.0, 0.6],
     [0.7, 0.8, 7.0, 0.9]]
tau = 6.0, bits = 8

Column absmax: [0.7, 0.8, 9.0, 0.9]  → only col2 > 6 → outlier_mask = [F, F, T, F]
scale of normal columns {0,1,3} = max normal-column absmax = 0.9
col2 kept verbatim: [9.0, 8.0, 7.0]
Remaining columns 8-bit quantized (nearly lossless):
W_recon ≈ [[0.0992, 0.1984, 9.0, 0.2976],
           [0.3969, 0.5031, 8.0, 0.6024],
           [0.7016, 0.8008, 7.0, 0.9000]]
```

## Constraints

- CPU float32; `outlier_mask` is compared with `torch.equal`, `W_recon` with `allclose(atol=1e-4)`.
- If all columns are outliers, there are no normal columns to quantize and `W_recon` equals `W`; if the scale is 0 (all normal columns are 0), treat it as 1.
- Rounding is round-to-nearest, with `.5` going to even (consistent with `torch.round`).
