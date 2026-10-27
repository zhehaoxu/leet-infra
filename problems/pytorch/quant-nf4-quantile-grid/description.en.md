# Quantization: NF4 Quantile Codebook Construction and Block Encoding

## Problem

**NF4** (4-bit NormalFloat, Dettmers et al., NeurIPS 2023, arXiv:2305.14314) is not a floating-point format but a **16-value lookup codebook**: assuming the weights are standard normal, split \([-1,1]\) into **equal-probability** bins and use each bin's quantile as its reconstruction value. This minimizes per-value reconstruction error on normal weights (information-theoretically optimal), which is why QLoRA compresses a 65B base to 4 bits with it.

How the codebook is built (the `create_normal_map(offset=0.9677083, use_extra_value=True)` convention in bitsandbytes) is this problem's first judged point. The positive half takes **8** probability points and the negative half takes **7**, plus an exact \(0\) — so the 16 values are **asymmetric** about 0 (one extra on the positive side):

\[
p_i^{+}=\text{offset}+\frac{0.5-\text{offset}}{8}\,i\ (i=0,\dots,7),\qquad
p_i^{-}=\text{offset}+\frac{0.5-\text{offset}}{7}\,i\ (i=0,\dots,6)
\]

\[
v^{+}_i=\Phi^{-1}(p_i^{+}),\qquad v^{-}_i=-\Phi^{-1}(p_i^{-})
\]

where \(\Phi^{-1}(p)=\sqrt2\,\operatorname{erfinv}(2p-1)\). Sort the 16 values \(\{v^+_i\}\cup\{-v^-_i\}\cup\{0\}\), then divide by the largest absolute value to normalize into \([-1,1]\) — this yields \(T_0<\dots<T_{15}\) with \(T_0=-1\), \(T_7=0\), \(T_{15}=1\).

Encoding follows QLoRA's **block scaling**: every 64 weights (consecutive columns of a row) share one absmax scale \(s_{i,\lfloor j/64\rfloor}=\max_j|W_{ij}|\); within a block let \(u=W_{ij}/s\) and take the nearest codebook entry

\[
c_{ij}=\arg\min_k |u-T_k|,\qquad \hat W_{ij}=T_{c_{ij}}\,s_{i,\lfloor j/64\rfloor}
\]

Ties go to the smaller code (same as `argmin`). If a block's absmax is 0, the whole block gets code 7 (\(T_7=0\)) and reconstructs to 0.

## Task

Implement `solve(w, offset)`:

- Input: `w` (float32 `(M, N)`, `N` a multiple of 64), `offset` (scalar in \((0.5,1)\)).
- Output `(codebook, codes, deq)`: `codebook` is `(16,)` float32; `codes` is `(M, N)` int64; `deq` is `(M, N)` float32 (dequantized reconstruction).

**Conventions**: compute internally in float64, return `codebook` and `deq` as float32; tolerance \(10^{-5}\). `bitsandbytes`, `torch.quantile` and `numpy.quantile` are forbidden.

## Example

```
w = shape (1, 64), all 0.5; w[0,0]=1.0, w[0,1]=-1.0, w[0,2]=0.95, offset=0.9677083
absmax=1 → u = w itself
codes[0,:4] = [15, 0, 15, 12]      # 1.0 and 0.95 both nearest to 1.0; 0.5 nearest to 0.4407 (code 12)
deq[0,:4]   = [1.0, -1.0, 1.0, 0.4407098]
```

## Constraints

- `0.5 < offset < 1`, otherwise raise `ValueError`; `N % 64 == 0`.
- The 16 codebook values are strictly increasing and **not** required to be symmetric about 0 (8 on the positive side, 7 on the negative side).
- Scaling a block by 2 leaves the codes unchanged and multiplies the reconstruction by 2 exactly (block-scaling linearity).
