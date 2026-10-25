# SpQR: Sparse Outlier Weight Retention + Per-Row Quantization

## Problem

LLM.int8() exempts outliers by **whole column**, which is far too coarse — a column usually contains only one or two genuinely large weights, and the rest get dragged along in FP, wasting bit width for nothing. **SpQR** (Dettmers et al., 2023, [arXiv:2306.03078](https://arxiv.org/abs/2306.03078)) refines the exemption granularity down to the **individual weight**: only the very few elements whose absolute value exceeds the threshold are stored at high precision (FP16) as a **sparse matrix**, and only the remaining dense part gets low-bit quantization. Because the outliers are sparse (a tiny fraction), storing them in a format such as CSR costs almost nothing, while the dense bulk can be squeezed down to 3–4 bits.

## Rules

Given weights $W$ of shape $(O, C)$, an outlier threshold $\tau$, and a bit width `bits` (default 8).

1. **Outlier mask** (element-wise): $\text{mask}_{oj}=|W_{oj}|>\tau$. These positions count as sparse outliers and are **kept as-is** (equivalent to the FP16 branch that does not quantize).
2. **Per-row quantization of the dense bulk**: for each row $o$, take the **non-outlier** elements of that row and apply per-row symmetric RTN:
   - row scale $s_o=\max_{j:\neg\text{mask}_{oj}}|W_{oj}|$ (skip the row if it has no non-outlier elements);
   - $q_{max}=2^{bits-1}-1$;
   - $q=\mathrm{clamp}(\mathrm{round}(W_{oj}/s_o\cdot q_{max}),\,-q_{max}-1,\,q_{max})$, dequantization $\hat W_{oj}=\dfrac{q}{q_{max}}s_o$.
3. Write the original values back at the outlier positions.

Return the tuple `(recon, mask)`, of shapes $(O, C)$ float and $(O, C)$ bool respectively.

- Unlike LLM.int8() (which decides by whole-column absmax), SpQR decides **element-wise** with $|W_{oj}|>\tau$, a finer granularity.
- The quantization range is **per-row**, not per-tensor: every row has its own scale, reducing the accuracy loss caused by magnitude differences between rows.

## Examples

```
W = [[0.1, 9.0, 0.2],
     [0.4, 0.5, 8.0]],  tau = 6.0, bits = 8

mask = [[False, True,  False],
        [False, False, True ]]        # 9.0 and 8.0 are outliers, kept as-is

row0 non-outliers {0.1, 0.2}: s=0.2 → 0.1→0.1008, 0.2→0.2
row1 non-outliers {0.4, 0.5}: s=0.5 → 0.4→0.4016, 0.5→0.5

recon ≈ [[0.1008, 9.0, 0.2],
         [0.4016, 0.5, 8.0]]
```

## Constraints

- CPU float32; `recon` compared with `torch.allclose(atol=1e-4)`, `mask` with `torch.equal`.
- Round to nearest, `.5` goes to even (consistent with `torch.round`).
- The mask is decided element-wise; outlier elements take no part in their row's scale computation and are never modified.
