# Quantization: RTN vs GPTQ Output MSE and Per-Group Error Quantiles

## Description

**GPTQ** (Frantar et al., arXiv:2210.17323) and naive **RTN (Round-To-Nearest)** use the same per-tensor symmetric quantization grid, but GPTQ compensates the quantization error of each column to the remaining columns via the inverse Hessian. For the same weight \(W\) and input \(X\), compute both RTN and GPTQ and report:

1. **Output MSE**
   \[\text{MSE}=\frac{1}{b\cdot d_\text{out}}\|XW^{\mathsf T}-X\hat W^{\mathsf T}\|_F^2.\]
2. **Per-group error quantiles**: split columns into groups of size `group_size`, compute the mean absolute error per group
   \[e_g=\frac{1}{d_\text{out}\cdot\text{group_size}}\sum_{(i,j)\in g}|W_{ij}-\hat W_{ij}|,\]
   and return the \(0.25,0.5,0.75,0.9\) quantiles.

Quantization rule (shared by RTN and GPTQ):

\[\text{scale}=\max|W|,\quad q_{ij}=\operatorname{clamp}\left(\operatorname{round}\left(\frac{W_{ij}}{\text{scale}}\cdot q_\max\right), -q_\max-1, q_\max\right),\quad \hat W_{ij}=\frac{q_{ij}}{q_\max}\cdot\text{scale},\]

where \(q_\max=2^{\text{bits}-1}-1\).

The GPTQ column update rule is the same as in `gptq-column-quantize`:

\[H^{-1}=(H+\lambda I)^{-1},\quad W_{:,k+1:C}\mathrel{-}=\frac{\varepsilon_i}{H^{-1}_{ii}}\cdot H^{-1}_{i,k+1:C},\]

with \(\varepsilon_i=W_{:,i}-\hat W_{:,i}\).

## Task

Implement `solve(W, H, X, bits, damp, group_size)`:

- `W`: float32 \((d_\text{out}, C)\).
- `H`: float32 \((C, C)\) symmetric positive semi-definite Hessian.
- `X`: float32 \((b, C)\).
- `bits`, `damp`: scalars.
- `group_size`: column-wise group size dividing \(C\).
- Return `(mse_rtn, mse_gptq, rtn_quantiles, gptq_quantiles)`:
  - `mse_rtn`, `mse_gptq` are float32 scalars;
  - `rtn_quantiles`, `gptq_quantiles` are float32 tensors of length \(4\), giving the \(0.25,0.5,0.75,0.9\) quantiles.

## Hints

- RTN simply rounds each column; GPTQ updates the unquantized columns after rounding with the error.
- Use `torch.quantile` on the flattened group statistics.
- GPTQ output MSE is usually not larger than RTN (tests will assert strict inequality).
